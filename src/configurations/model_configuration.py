from .configuration import Configuration

from typing import TextIO, Union, Dict


class ModelConfiguration(Configuration):
    def __init__(self, config_file: Union[TextIO, Dict]):
        super(ModelConfiguration, self).__init__(config_file)
    
    
    def _load_config_json_schema(self):
        """
        Load the JSON schema for the model configuration file.
        """
        self._config_json_schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "Model Configuration Schema",
            "type": "object",
            "properties": {
                "device": {
                "type": "string",
                "description": "Device to use for training.",
                "enum": ["cpu", "cuda"],
                "default": "cpu"
                },
                "optimizer": {
                "type": "string",
                "description": "Optimizer to use for training.",
                "enum": ["adam", "sgd"],
                "default": "sgd"
                },
                "loss_function": {
                "type": "string",
                "description": "Loss function to use for training.",
                "enum": ["cross_entropy", "nll"],
                "default": "nll"
                },
                "batch_size": {
                "type": "integer",
                "description": "Batch size for training.",
                "minimum": 1,
                "default": 128
                },
                "learning_rate": {
                "type": "number",
                "description": "Learning rate for training.",
                "minimum": 0.000001,
                "default": 0.001
                },
                "num_epochs": {
                "type": "integer",
                "description": "Number of epochs for training.",
                "minimum": 1,
                "default": 1
                },
                "input_channels": {
                "type": "integer",
                "description": "Number of input channels for the model.",
                "minimum": 1
                },
                "layers": {
                "type": "array",
                "description": "List of layers in the model.",
                "items": {
                    "type": "object",
                    "properties": {
                    "type": {
                        "type": "string",
                        "description": "Type of the layer.",
                        "enum": [
                        "conv2d",
                        "linear",
                        "maxpool2d",
                        "batchnorm2d",
                        "batchnorm1d",
                        "dropout",
                        "flatten",
                        "activation"
                        ]
                    },
                    "in_features": {
                        "type": "integer",
                        "description": "Number of input features (for linear layers).",
                        "minimum": 1
                    },
                    "out_features": {
                        "type": "integer",
                        "description": "Number of output features (for linear layers).",
                        "minimum": 1
                    },
                    "out_channels": {
                        "type": "integer",
                        "description": "Number of output channels (for Conv2D layers).",
                        "minimum": 1
                    },
                    "kernel_size": {
                        "type": "integer",
                        "description": "Kernel size for Conv2D or MaxPool2D layers.",
                        "minimum": 1
                    },
                    "stride": {
                        "type": "integer",
                        "description": "Stride for Conv2D or MaxPool2D layers.",
                        "minimum": 1,
                        "default": 1
                    },
                    "padding": {
                        "type": "integer",
                        "description": "Padding for Conv2D layers.",
                        "minimum": 0,
                        "default": 0
                    },
                    "num_features": {
                        "type": "integer",
                        "description": "Number of features for BatchNorm layers.",
                        "minimum": 1
                    },
                    "affine": {
                        "type": "boolean",
                        "description": "Whether BatchNorm has learnable parameters (gamma, beta).",
                        "default": True
                    },
                    "track_running_stats": {
                        "type": "boolean",
                        "description": "Whether BatchNorm tracks running statistics.",
                        "default": True
                    },
                    "p": {
                        "type": "number",
                        "description": "Dropout probability.",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 0.5
                    },
                    "activation": {
                        "type": "string",
                        "description": "Type of activation function.",
                        "enum": ["relu", "leakyrelu", "sigmoid", "tanh", "softmax", "log_softmax"]
                    },
                    "dim": {
                        "type": "integer",
                        "description": "Dimension for Softmax activation.",
                        "default": 1
                    }
                    },
                    "required": ["type"],
                    "oneOf": [
                    {
                        "properties": { "type": { "const": "conv2d" } },
                        "required": ["out_channels", "kernel_size"]
                    },
                    {
                        "properties": { "type": { "const": "linear" } },
                        "required": ["in_features", "out_features"]
                    },
                    {
                        "properties": { "type": { "const": "maxpool2d" } },
                        "required": ["kernel_size"]
                    },
                    {
                        "properties": { "type": { "const": "batchnorm2d" } },
                        "required": ["num_features"]
                    },
                    {
                        "properties": { "type": { "const": "batchnorm1d" } },
                        "required": ["num_features"]
                    },
                    {
                        "properties": { "type": { "const": "dropout" } },
                        "required": ["p"]
                    },
                    {
                        "properties": { "type": { "const": "flatten" } }
                    },
                    {
                        "properties": { "type": { "const": "activation" } },
                        "required": ["activation"]
                    }
                    ]
                }
                }
            },
            "required": ["input_channels", "layers"]
        }
