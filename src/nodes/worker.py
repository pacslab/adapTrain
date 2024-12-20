import numpy as np
import json
import time

from copy import deepcopy

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Subset, DataLoader
from torchvision import datasets, transforms

from src.configurations import ModelConfiguration
from src.configurations import DeploymentConfiguration
from src.configurations import PartitioningConfiguration
from src.configurations.defaults import LAYERS_TO_PARTITION

from src.dataset import AdapTrainDataset

from src.models import Model

from src.logger import logger

from typing import Dict, Union, TextIO


class Worker:
    def __init__(self,
                 rank: int,
                 m_config: Union[TextIO, Dict] = None,
                 p_config: Union[TextIO, Dict] = None,
                 d_config: Union[TextIO, Dict] = None,
                 dataset_path = None) -> None:
        
        if dataset_path is None:
            raise ValueError("Please provide the path to the dataset.")
        
        torch.manual_seed(1)
        
        self._d_config = DeploymentConfiguration(d_config)
        self._m_config = ModelConfiguration(m_config).__dict__
        self._p_config = PartitioningConfiguration(p_config).__dict__
        
        self.dataset_path = dataset_path

        self.rank = rank
        self.backend = self._d_config.dist_backend
        self.init_method = self._d_config.dist_url
        self.world_size = self._d_config.num_workers + 1
        
        self.dataset_path = dataset_path
        
        self.coefficient = 1 / (self.world_size - 1)
        self.all_coefs = [self.coefficient] * (self.world_size - 1)
        
        # To get updated during training
        self.training_round_time = 0.0
        
        self.m_config = m_config
        self.p_config = p_config
        
        if isinstance(self.p_config, TextIO):
            self.p_config = json.loads(self.p_config)

   
        self.p_config = PartitioningConfiguration(self.p_config).__dict__
        

        self.repartition_iter = self.p_config["repartition_iter"]
        
        self._start_process()
        
        self._init_model()

        self._train()
        
        self._destroy()
        
        
    def _start_process(self):
        dist.init_process_group(backend=self.backend,
                                init_method=self.init_method,
                                rank=self.rank,
                                world_size=self.world_size)
        
        logger.info(f"Node with rank {self.rank} initialized.")


    def _find_partition_size(self, layer_size):
        if self.rank == self.world_size - 1:
            return int(layer_size - np.sum(np.floor(np.array(self.all_coefs[:-1]) * layer_size).astype(int)))
        else:
            return int(layer_size * self.all_coefs[self.rank - 1])


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
                        updated_config["layers"][i]["out_features"] = self._find_partition_size(layer["out_features"])
                        updated_config["layers"][i]["in_features"] = self._find_partition_size(layer["in_features"])
                    else:
                        updated_config["layers"][i]["out_features"] = self._find_partition_size(layer["out_features"])

                    last_linear = i
                elif layer["type"] == "batchnorm1d":
                    updated_config["layers"][i]["num_features"] = self._find_partition_size(layer["num_features"])

                    last_batchnorm = i
                is_consequence = True

            elif layer["type"] != "activation":
                is_consequence = False

             
        if last_linear is not None:
            updated_config["layers"][last_linear]["out_features"] = self.m_config["layers"][last_linear]["out_features"]

        if last_batchnorm is not None:
            updated_config["layers"][last_batchnorm]["num_features"] = self.m_config["layers"][last_batchnorm]["num_features"]
            
        self.model = Model(updated_config)

        logger.info(f"Model initiated on node with rank {self.rank}")
        
        
    def _send_training_round_time(self):
        dist.send(torch.tensor(self.training_round_time, dtype=torch.float32), dst=0)

    
    def _receive_coefficient(self):
        tensor = torch.zeros(self.world_size - 1, dtype=torch.float32)
        dist.recv(tensor, src=0)

        self.all_coefs = tensor
        self.coefficient = tensor[self.rank - 1]

    
    def _destroy(self):
        dist.destroy_process_group()
        
        logger.info(f"Node with rank {self.rank} destroyed.")
        
    
    def _send_learned_partitions_to_controller(self):
        batchnorm_layers = [layer_i for layer_i, layer in enumerate(self.model.model) if layer.__class__.__name__.lower() == "batchnorm1d"]
        for layer_i, layer in enumerate(self.model.model):
            if layer.__class__ in LAYERS_TO_PARTITION and layer_i != batchnorm_layers[-1]:
                if layer.weight is not None:
                    dist.send(layer.weight.data, dst=0)

                if layer.bias is not None:
                    dist.send(layer.bias.data, dst=0)                
            
            
    def _receive_partitions_from_controller(self):
        for layer_i, layer in enumerate(self.model.model):
            if hasattr(layer, 'weight') and layer.weight is not None:
                dist.recv(layer.weight.data, src=0)
            if hasattr(layer, 'bias') and layer.bias is not None:
                dist.recv(layer.bias.data, src=0)
        
        
    def _train(self):
        learning_rate = self.m_config["learning_rate"]
        num_epochs = self.m_config["num_epochs"]
        batch_size = self.m_config["batch_size"]
        log_interval = self.p_config["log_interval"]
        
        loss_fn = self.model.get_loss_function()
        optimizer = self.model.get_optimizer()
        
        train_set = AdapTrainDataset(X_path=f"{self.dataset_path}/train_x.npy", y_path=f"{self.dataset_path}/train_y.npy")

        subset_size = len(train_set) // (self.world_size - 1)

        indices = list(range((self.rank - 1)* subset_size, self.rank * subset_size))

        subset = Subset(train_set, indices)

        train_loader = DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,
                                drop_last=True)

        self.model.train()
        
        self.training_round_time = time.time()
        
        # Repartitioning iterations
        for epoch_i in range(num_epochs):  
            for batch_i, (X, y) in enumerate(train_loader):
                if batch_i % self.p_config["repartition_iter"] == 0:

                    if epoch_i > 0 or batch_i > 0:
                        logger.info(f"Worker {self.rank} is repartitioning.")

                        self._receive_coefficient()
                        self._init_model()
                        self._receive_partitions_from_controller()
                        
                        logger.info(f"Worker {self.rank} has received partitions.")
                        
                        optimizer = self.model.get_optimizer(learning_rate)
                        
                        self.model.train()

                        self.training_round_time = time.time()
                
                optimizer.zero_grad()
                
                y_pred = self.model(X)
                loss = loss_fn(y_pred, y)
                loss.backward()
                optimizer.step()
                
                train_pred = y_pred.max(1, keepdim=True)[1]
                correct = train_pred.eq(y.view_as(train_pred)).sum().item()
                
                if batch_i % log_interval == 0:
                    logger.info(f"Epoch: {epoch_i}, Batch: {batch_i}, Loss: {loss.item()}, Accuracy: %{correct / y.shape[0] * 100}, Worker: {self.rank}")
                
                if (batch_i + 1) % self.p_config["repartition_iter"] == 0 or batch_i == len(train_loader) - 1:
                    logger.info(f"Worker {self.rank} is sending partitions and training round time to controller.")

                    self.training_round_time = time.time() - self.training_round_time
                    self._send_learned_partitions_to_controller()
                    self._send_training_round_time()
                    
                    logger.info(f"Worker {self.rank} has sent partitions to controller.")


        logger.info(f"Training done on worker {self.rank}")