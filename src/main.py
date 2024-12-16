import json

import torch.nn as nn
import torch
import numpy as np

from .configurations import DeploymentConfiguration
from .configurations import PartitioningConfiguration
from .logger import logger
from .nodes import Controller
from .models import Model

from src.utils import partition_layer, update_layer


def main():
    d_config = """
    {
        "num_workers": 1
    }
    """
    p_config = """
    {
        "repartition_iter": 64
    }
    """
    m_config = """
    {
    "input_channels": 1,
    "layers": [
        {
            "type": "linear",
            "in_features": 4096,
            "out_features": 4096
        },
        {
            "type": "batchnorm1d",
            "num_features": 4096
        },
        {
            "type": "activation",
            "activation": "relu"
        },
        {
            "type": "linear",
            "in_features": 4096,
            "out_features": 4096
        },
        {
            "type": "batchnorm1d",
            "num_features": 4096
        },
        {
            "type": "activation",
            "activation": "relu"
        },
        {
            "type": "linear",
            "in_features": 4096,
            "out_features": 35
        },
        {
            "type": "batchnorm1d",
            "num_features": 35
        }
    ]
}
    """
    d_config = json.loads(d_config)
    m_config = json.loads(m_config)
    p_config = json.loads(p_config)
    
    m = Model(m_config)
    
    # for layer in layers.values():
    #     with torch.no_grad():
    #         if isinstance(layer, nn.Linear):
    #             print(layer.in_features, layer.out_features)
    #         if isinstance(layer, nn.BatchNorm1d):
    #             print(layer.num_features)
    
    ctrl = Controller(m_config, d_config, p_config)
    
    
    # print(m.model[0].weight.shape)

    # # Sample n values from 1 to m
    # samples = np.random.choice(np.arange(0, 128), size=98, replace=False)
    # dim_0 = [torch.tensor(samples), torch.tensor(np.array([j for j in range(0, 128) if j not in samples]))]
    
    # samples = np.random.choice(np.arange(0, 64), size=32, replace=False)
    # dim_1 = [torch.tensor(samples), torch.tensor(np.array([j for j in range(0, 64) if j not in samples]))]
    
    # # print(m.model[0].weight)
    # a, b = partition_layer(m.model[0], True, False, dim_0, dim_1)
    
    # print(a[0].shape, a[1].shape)
    
    # a[0] = a[0].fill_(100)
    # a[1] = a[1].fill_(200)

    
    # j = update_layer(m.model[0], a, None, dim_0, dim_1)
    
    # print(j.weight)