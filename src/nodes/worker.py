import numpy as np
import json
import time

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from copy import deepcopy

import torch
import torch.nn as nn
import torch.distributed as dist

from src.configurations import PartitioningConfiguration
from src.configurations.defaults import LAYERS_TO_PARTITION

from src.models import Model

from src.logger import logger

from typing import Dict, Union, TextIO

import src.google_speech_data_loader as speech_dataset

class Worker:
    def __init__(self,
                 rank: int,
                 backend: str,
                 init_method: str,
                 world_size: int,
                 m_config: Union[TextIO, Dict] = None,
                 p_config: Union[TextIO, Dict] = None) -> None:
        
        self.rank = rank
        self.backend = backend
        self.init_method = init_method
        self.world_size = world_size
        
        self.coefficient = 1 / (world_size - 1)
        
        # To get updated during training
        self.training_round_time = 0.0
        
        self.m_config = m_config
        self.p_config = p_config
        
        if isinstance(self.p_config, TextIO):
            self.p_config = json.loads(self.p_config)
            
        self.p_config = PartitioningConfiguration(self.p_config).__dict__
        
        # Beggaii
        self.repartition_iter = self.p_config["repartition_iter"]
        
        self._start_process()
        
        self._init_model()
        self.train()
        
        
    def _start_process(self):
        dist.init_process_group(backend=self.backend,
                                init_method=self.init_method,
                                rank=self.rank,
                                world_size=self.world_size)
        
        logger.info(f"Node with rank {self.rank} initialized.")
        
    
    def _init_model(self):
        if isinstance(self.m_config, TextIO):
            self.m_config = json.loads(self.m_config)

        updated_config = deepcopy(self.m_config)
        partitioned_layers = [layer.__name__.lower() for layer in LAYERS_TO_PARTITION]
        
        is_consequence = False
        last_linear = None
        last_batchnorm = None
        for i, layer in enumerate(updated_config["layers"]):
            if layer["type"] in partitioned_layers:
                if layer["type"] == "linear":
                    if is_consequence:
                        updated_config["layers"][i]["out_features"] = int(layer["out_features"] * self.coefficient)
                        updated_config["layers"][i]["in_features"] = int(layer["in_features"] * self.coefficient)
                    else:
                        updated_config["layers"][i]["out_features"] = int(layer["out_features"] * self.coefficient)
                    last_linear = i
                elif layer["type"] == "batchnorm1d":
                    updated_config["layers"][i]["num_features"] = int(layer["num_features"] * self.coefficient)
                    last_batchnorm = i
                is_consequence = True
            elif layer["type"] != "activation":
                is_consequence = False
                
        if last_linear is not None:
            updated_config["layers"][last_linear]["out_features"] = self.m_config["layers"][last_linear]["out_features"]
        if last_batchnorm is not None:
            updated_config["layers"][last_batchnorm]["num_features"] = self.m_config["layers"][last_batchnorm]["num_features"]
            
        self.model = Model(updated_config)

        logger.info(f"Model initiated on node with rank {self.rank}.")
        
        
    def _send_training_round_time(self):
        dist.send(torch.tensor(self.training_round_time, dtype=torch.float32), dst=0)

    
    def _receive_coefficient(self):
        tensor = torch.zeros(1, dtype=torch.float32)
        dist.recv(tensor, src=0)
        
        self.coefficient = tensor

    
    def destroy(self):
        dist.destroy_process_group()
        
        logger.info(f"Node with rank {self.rank} destroyed.")
        
    
    def _send_learned_partitions_from_workers(self):
        batchnorm_layers = [layer_i for layer_i, layer in enumerate(self.model.model) if layer.__class__.__name__.lower() == "batchnorm1d"]
        for layer_i, layer in enumerate(self.model.model):
            if layer.__class__ in LAYERS_TO_PARTITION and layer_i != batchnorm_layers[-1]:
                # print(f"Sending layer {layer_i} from worker {self.rank}: {layer.weight.shape}")
                if layer.weight is not None:
                    dist.send(layer.weight.data.detach(), dst=0)
                if layer.bias is not None:
                    dist.send(layer.bias.data.detach(), dst=0)                
            
            
    def _receive_partitions_from_master(self):
        for layer_i, layer in enumerate(self.model.model):
            # print(f"Receiving layer {layer} {layer_i} in worker {self.rank}")
            if hasattr(layer, 'weight') and layer.weight is not None:
                dist.recv(layer.weight.data, src=0)
            if hasattr(layer, 'bias') and layer.bias is not None:
                dist.recv(layer.bias.data, src=0)
        
        
    def train(self):
        learning_rate = self.m_config["learning_rate"]
        num_epochs = self.m_config["num_epochs"]
        batch_size = self.m_config["batch_size"]
        
        loss_fn = self.model.get_loss_function()
        optimizer = self.model.get_optimizer()
        
        train_set = speech_dataset.train_dataset()
        test_set = speech_dataset.test_dataset()
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,
                                drop_last=True)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,
                                drop_last=False)
        
        self.model.train()
        
        self.training_round_time = time.time()
        
        # Repartitioning iterations
        for epoch_i in range(num_epochs):
            correct = 0
            for batch_i, batch in enumerate(train_loader):
                print(batch_i)
                if batch_i % self.p_config["repartition_iter"] == 0 and batch_i != 0:
                    print(f"Repartitioning in worker {self.rank}")
                    self._receive_coefficient()
                    self._init_model()
                    self._receive_partitions_from_master()
                    
                    optimizer = self.model.get_optimizer(learning_rate)
                    self.model.train()

                    self.training_round_time = time.time()
                
                X, y = batch['wav'].float(), batch['label']
                
                optimizer.zero_grad()
                
                # Pay Attention
                # batch_i += 1
                
                y_pred = self.model(X)
                import torch.nn.functional as F
                loss = F.nll_loss(y_pred, y)
                loss.backward()
                optimizer.step()
                
                train_pred = y_pred.max(1, keepdim=True)[1]
                correct = train_pred.eq(y.view_as(train_pred)).sum().item()
                
                if batch_i % 100 == 0:
                    logger.info(f"Epoch: {epoch_i}, Batch: {batch_i}, Loss: {loss.item()}, Accuracy: %{correct / y.shape[0] * 100}, Worker: {self.rank}")
                
                if (batch_i + 1) % self.p_config["repartition_iter"] == 0 or batch_i == len(train_loader) - 1:
                    self.training_round_time = time.time() - self.training_round_time
                    self._send_learned_partitions_from_workers()
                    self._send_training_round_time()
