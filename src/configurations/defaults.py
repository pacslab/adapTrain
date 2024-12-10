import logging

import torch.nn as nn

LOG_LEVEL = logging.INFO

LAYERS_TO_PARTITION = [
    nn.Linear,
    nn.BatchNorm1d
]

LAYERS_TO_PARTITION_STRING = [
    "Linear",
    "BatchNorm1d"
]