import numpy as np

import torch
import torch.nn as nn
import torch.distributed as dist

from src.configurations.defaults import LAYERS_TO_PARTITION_STRING

from src.models import Model

from src.logger import logger

from typing import Dict, Union, TextIO


class Worker:
    def __init__(self,
                 rank: int,
                 backend: str,
                 init_method: str,
                 world_size: int,
                 m_config: Union[TextIO, Dict] = None,
                 p_config: Union[TextIO, Dict] = None) -> None:
        # Might not need p_config for the worker
        
        self.rank = rank
        self.backend = backend
        self.init_method = init_method
        self.world_size = world_size
        
        self.init_coefficient = 1 / world_size
        
        # To get updated during training
        self.training_round_time = 0.0
        
        self.m_config = m_config
        self.p_config = p_config
        
        self._start_process()
        
        self._init_model()
        
        
    def _start_process(self):
        dist.init_process_group(backend=self.backend,
                                init_method=self.init_method,
                                rank=self.rank,
                                world_size=self.world_size)
        
        logger.info(f"Node with rank {self.rank} initialized.")
        
    
    def _init_model(self):
        updated_config = self.m_config.copy()
        
        for i, layer in enumerate(self.m_config["layers"]):
            if layer["type"] in LAYERS_TO_PARTITION_STRING:
                pass
            
        self.model = Model(self.m_config)
        
        logger.info(f"Model initiated on node with rank {self.rank}.")
        
        
    def _send_training_round_time(self):
        dist.send(torch.tensor([self.training_round_time], dtype=torch.float32), dst=0)
        
    
    def destroy(self):
        dist.destroy_process_group()
        
        logger.info(f"Node with rank {self.rank} destroyed.")
