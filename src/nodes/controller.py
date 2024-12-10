import random
import torch
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

from src.nodes import Worker

from src.models import Model

from src.logger import logger


class Controller:
    
    def __init__(self,
                 m_config: Union[TextIO, Dict],
                 d_config: Union[TextIO, Dict],
                 p_config: Union[TextIO, Dict],) -> None:
        
        self._d_config = DeploymentConfiguration(d_config)
        self._m_config = ModelConfiguration(m_config)
        self._p_config = PartitioningConfiguration(p_config)
        
        self.raw_model = Model(m_config)
        
        print(self.raw_model.model)

        # partitioned_indices includes the indices of the layers to partition for each worker
        self.layers_to_partition_indices, \
            self.partitioned_indices_per_workers_per_layer, \
                self.partitioned_parameters_to_workers_per_layer = self._extract_layers_to_partition()

        self.num_of_workers = self._d_config.num_workers
        
        # Initial partition coefficients
        self.partition_coefficients = [1 / self._d_config.num_workers] * self._d_config.num_workers
        
        self.generate_partitions()

        self.workers: List[Worker] = []
        
        # self._init_nodes()
        #self._destroy_nodes()
    
        
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
                world_size=world_size
            )
        )
        
    
    def _destroy_nodes(self):
        for worker in tqdm(self.workers):
            worker.destroy()
            
        logger.info("All nodes have been destroyed.")
    
    
    def _init_nodes(self):
        processes = []

        for rank in tqdm(range(self.num_of_workers)):
            p = Process(
                target=self._launch_process,
                args= (
                        rank,
                        self._d_config.dist_backend, 
                        self._d_config.dist_url, 
                        self._d_config.num_workers
                    )
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
            
        logger.info("All nodes have been initialized.")
        
    
    def generate_partitions(self):
        for i, layer_i in enumerate(self.layers_to_partition_indices):
            if self.raw_model.model[layer_i].__class__.__name__ == "Linear":
                layer_size = self.raw_model.model[layer_i].out_features
            elif self.raw_model.model[layer_i].__class__.__name__ == "BatchNorm1d":
                print(self.raw_model.model[layer_i].num_features)
                layer_size = self.raw_model.model[layer_i].num_features
            
            self.partitioned_indices_per_workers_per_layer[i].clear()
                
            layer_neurons = np.arange(layer_size)
            np.random.shuffle(layer_neurons)
            
            cursor = 0
            for worker_index in range(self.num_of_workers):
                if i < self.num_of_workers - 1:
                    end_cursor = cursor + int(self.partition_coefficients[worker_index] * layer_size)
                else:
                    end_cursor = layer_size

                # Generate current indexes based on whether coefficients are equal
                current_indexes = torch.tensor(layer_neurons[cursor:end_cursor])

                # Update cursor and log the indexes
                cursor = end_cursor
                self.partitioned_indices_per_workers_per_layer[i].append(current_indexes)
                
            self.partitioned_parameters_to_workers_per_layer[i].clear()
            
            self.partitioned_parameters_to_workers_per_layer[i] = partition_layer(
                self.raw_model.model[layer_i],
                True,
                False if self.raw_model.model[layer_i].__class__.__name__ == "Linear" else True,
                None,
                None
            )
                
        
        
    def _test_model(self):
        pass
    
    
    def _send_model_parameters_to_workers(self):
        pass
    
    
    def _receive_model_parameters_from_workers(self):
        pass
    
    
    def _partition_model(self):
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
                
        return (extracted_layers_indices, 
                [[[] for _ in range(self._d_config.num_workers)] for _ in range(len(extracted_layers_indices))],
                [[] for _ in range(len(extracted_layers_indices))])