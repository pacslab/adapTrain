import random
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import numpy as np

from tqdm import tqdm

from multiprocessing import Process

from typing import TextIO, Union, Dict, List

from src.configurations import ModelConfiguration
from src.configurations import DeploymentConfiguration
from src.configurations import PartitioningConfiguration
from src.configurations.defaults import LAYERS_TO_PARTITION

from src.utils import partition_layer
from src.utils import update_layer
from src.utils import adaptive_partitioning

from src.nodes import Worker

from src.models import Model

from src.logger import logger

import src.google_speech_data_loader as speech_dataset


class Controller:
    
    def __init__(self,
                 m_config: Union[TextIO, Dict],
                 d_config: Union[TextIO, Dict],
                 p_config: Union[TextIO, Dict],) -> None:
        
        self._d_config = DeploymentConfiguration(d_config)
        self._m_config = ModelConfiguration(m_config).__dict__
        self._p_config = PartitioningConfiguration(p_config).__dict__
        
        self.raw_model = Model(m_config)
        
        print(self.raw_model)

        # partitioned_indices includes the indices of the layers to partition for each worker
        self.layers_to_partition_indices, \
            self.partitioned_indices_per_workers_per_layer, \
                self.partitioned_parameters_to_workers_per_layer = self._extract_layers_to_partition()

        self.num_of_workers = self._d_config.num_workers
        
        # Initial partition coefficients
        self.partition_coefficients = torch.tensor([1 / self._d_config.num_workers] * self._d_config.num_workers)
        
        self.training_round_times = torch.tensor([0.0] * self._d_config.num_workers)

        self.workers: List[Worker] = []
        
        self._init_nodes()
        
        # self._destroy_nodes()
        
        
    def _init_controller(self):
        dist.init_process_group(backend=self._d_config.dist_backend,
                                init_method=self._d_config.dist_url,
                                rank=0,
                                world_size=self._d_config.num_workers + 1)

        self.scheduler()


    def _launch_process(self,
                        rank: int,
                        backend: str,
                        init_method: str,
                        world_size: int) -> None:

        self.workers.append(
            Worker(
                rank=rank,
                backend=backend,
                init_method=init_method,
                world_size=world_size,
                m_config=self._m_config,
                p_config=self._p_config
            )
        )
        
    
    def _destroy_nodes(self):
        for worker in tqdm(self.workers):
            worker.destroy()
            
        logger.info("All nodes have been destroyed.")
    
    
    def _init_nodes(self):
        processes = []
        
        p = Process(target=self._init_controller, args=())
        p.start()
        processes.append(p)

        for rank in tqdm(range(1, self.num_of_workers + 1)):
            p = Process(
                target=self._launch_process,
                args= (
                        rank,
                        self._d_config.dist_backend, 
                        self._d_config.dist_url, 
                        self._d_config.num_workers + 1
                    )
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
            
        logger.info("All nodes have been initialized.")
        
    
    def _generate_partitions(self):
        self.layers_to_partition_indices, \
            self.partitioned_indices_per_workers_per_layer, \
                self.partitioned_parameters_to_workers_per_layer = self._extract_layers_to_partition()
                
        linear_layers_to_partition = [layer_i for layer_i in self.layers_to_partition_indices if self.raw_model.model[layer_i].__class__.__name__ == "Linear"]

        for i, layer_i in enumerate(self.layers_to_partition_indices):
            if self.raw_model.model[layer_i].__class__.__name__ == "Linear":
                layer_size = self.raw_model.model[layer_i].out_features
                partition_dim_0 = True if layer_i != linear_layers_to_partition[-1] else False
                partition_dim_1 = True if layer_i != linear_layers_to_partition[0] else False
            elif self.raw_model.model[layer_i].__class__.__name__ == "BatchNorm1d":
                layer_size = self.raw_model.model[layer_i].num_features
                partition_dim_0 = True
                partition_dim_1 = False
                
            layer_neurons = np.arange(layer_size)
            np.random.shuffle(layer_neurons)
            
            cursor = 0
            for worker_index in range(self.num_of_workers):
                if worker_index < self.num_of_workers - 1:
                    end_cursor = cursor + int(self.partition_coefficients[worker_index] * layer_size)
                else:
                    end_cursor = layer_size

                # Generate current indexes based on whether coefficients are equal
                current_indexes = torch.tensor(layer_neurons[cursor:end_cursor])

                # Update cursor and log the indexes
                cursor = end_cursor
                self.partitioned_indices_per_workers_per_layer[i][worker_index] = current_indexes.detach().clone()
            
            dim_0_indices = self.partitioned_indices_per_workers_per_layer[i] if partition_dim_0 else None
            dim_1_indices = self.partitioned_indices_per_workers_per_layer[i - 1] if partition_dim_1 else None
                
            partitioned_weights, partitioned_biases = partition_layer(
                self.raw_model.model[layer_i],
                True,
                False if self.raw_model.model[layer_i].__class__.__name__ == "Linear" else True,
                dim_0_indices,
                dim_1_indices
            )
            
            self.partitioned_parameters_to_workers_per_layer[i] = (partitioned_weights, partitioned_biases)
    
    
    def _receive_training_round_times(self):
        for i in range(self.num_of_workers):
            dist.recv(tensor=self.training_round_times[i], src=i + 1)
            # print(f"Received training round time from worker {i}: {self.training_round_times[i]}")

        
        # self.partition_coefficients = adaptive_partitioning(self.training_round_times, self.partition_coefficients)
        # print(self.partition_coefficients)
    
    
    def _send_coefficients_to_workers(self):
        for i in range(self.num_of_workers):
            dist.send(tensor=self.partition_coefficients[i].clone().detach(), dst=i + 1)
    
    
    def _send_partitions_to_workers(self):
        for layer_i, layer in enumerate(self.raw_model.model):
            if layer.__class__ in LAYERS_TO_PARTITION and layer_i in self.layers_to_partition_indices:
                i = self.layers_to_partition_indices.index(layer_i)
                partitioned_weights, partitioned_biases = self.partitioned_parameters_to_workers_per_layer[i]
                
                for worker_i in range(self.num_of_workers):
                    # print(f"Sending layer {layer_i} to worker {worker_i + 1}: {partitioned_weights[worker_i].shape}")
                    if layer.weight is not None:
                        dist.send(tensor=partitioned_weights[worker_i].clone().detach(), dst=worker_i + 1)
                    if layer.bias is not None:
                        dist.send(tensor=partitioned_biases[worker_i].clone().detach(), dst=worker_i + 1)
            else:
                for worker_i in range(self.num_of_workers):
                    if hasattr(layer, 'weight') and layer.weight is not None:
                        dist.send(tensor=layer.weight.data.clone().detach(), dst=worker_i + 1)
                    if hasattr(layer, 'bias') and layer.bias is not None:
                        dist.send(tensor=layer.weight.data.clone().detach(), dst=worker_i + 1)
                
    
    
    def _receive_learned_partitions_from_workers(self):
        linear_layers_to_partition = [layer_i for layer_i in self.layers_to_partition_indices if self.raw_model.model[layer_i].__class__.__name__ == "Linear"]

        for layer_i, layer in enumerate(self.raw_model.model):
            if layer.__class__ in LAYERS_TO_PARTITION and layer_i in self.layers_to_partition_indices:
                i = self.layers_to_partition_indices.index(layer_i)
                partitioned_weights, partitioned_biases = self.partitioned_parameters_to_workers_per_layer[i]
                
                for worker_i in range(self.num_of_workers):
                    # print(partitioned_weights[worker_i].shape, layer.weight.shape)
                    if layer.weight is not None:
                        dist.recv(tensor=partitioned_weights[worker_i], src=worker_i + 1)
                    if layer.bias is not None:
                        dist.recv(tensor=partitioned_biases[worker_i], src=worker_i + 1)
                        
                if layer.__class__.__name__ == "Linear":
                    partition_dim_0 = True if layer_i != linear_layers_to_partition[-1] else False
                    partition_dim_1 = True if layer_i != linear_layers_to_partition[0] else False
                elif layer.__class__.__name__ == "BatchNorm1d":
                    partition_dim_0 = True
                    partition_dim_1 = False

                        
                dim_0_indices = self.partitioned_indices_per_workers_per_layer[i] if partition_dim_0 else None
                dim_1_indices = self.partitioned_indices_per_workers_per_layer[i - 1] if partition_dim_1 else None
                
                new_layer = update_layer(layer, partitioned_weights, partitioned_biases, dim_0_indices, dim_1_indices)
                #Might be buggy
                self.raw_model.model[layer_i] = new_layer
            
            else:
                # TODO: Add support for other layers
                # Aggregating the trained non-partitioned layers from the workers
                # for worker_i in range(1, self.num_of_workers):
                #     dist.recv(tensor=self.raw_model.model[layer_i], src=worker_i)
                pass
        
    
    
    def _extract_layers_to_partition(self):
        """
        Extract the layers to partition from the model.
        
        Returns:
            list: A list containing the extracted layers to partition.
        """
        extracted_layers_indices = []
        
        for i, layer in enumerate(self.raw_model.model):
            if layer.__class__ in LAYERS_TO_PARTITION:
                extracted_layers_indices.append(i)
                
        if self.raw_model.model[extracted_layers_indices[-1]].__class__.__name__ != "Linear":
            extracted_layers_indices.pop()
                
        return (extracted_layers_indices, 
                [[[] for _ in range(self._d_config.num_workers)] for _ in range(len(extracted_layers_indices))],
                [([], []) for _ in range(len(extracted_layers_indices))])
        
        
        
    def scheduler(self):
        self._generate_partitions()
        
        num_epochs = self._m_config["num_epochs"]
        batch_size = self._m_config["batch_size"]

        train_set = speech_dataset.train_dataset()
        test_set = speech_dataset.test_dataset()
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,
                                drop_last=True)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,
                                drop_last=False)
        
        for epoch_i in range(num_epochs):
            for batch_i, (X, y) in enumerate(train_loader):
                if batch_i % self._p_config["repartition_iter"] == 0 and batch_i != 0:
                    print("Controller: Repartitioning")
                    self._send_coefficients_to_workers()
                    self._generate_partitions()
                    self._send_partitions_to_workers()
                    print("Controller: Sent partitions to workers")
                
                if (batch_i + 1) % self._p_config["repartition_iter"] == 0 or batch_i == len(train_loader) - 1:
                    print("Controller: Receiving learned partitions")
                    self._receive_learned_partitions_from_workers()
                    self._receive_training_round_times()
            
            logger.info(f"Epoch {epoch_i} finished.")
            
            self._test_model(test_loader)
      
            
    def _test_model(self, test_loader):
        import torch.nn as nn
        self.raw_model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for _, batch in enumerate(test_loader):
                data, target = batch['wav'].float(), batch['label']
                output = self.raw_model(data)
                test_loss += nn.functional.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                test_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
                test_correct += test_pred.eq(target.view_as(test_pred)).sum().item()
                test_total += target.shape[0]
            test_acc = float(test_correct) / float(test_total)
            test_loss /= float(test_total)
        logger.info(f"Accuracy: {test_acc}")