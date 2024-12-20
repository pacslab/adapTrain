import torch
import torch.nn as nn

from typing import TextIO, Union, Dict

from src.configurations import ModelConfiguration

class Model(nn.Module):


    def __init__(self,
                 m_config: Union[TextIO, Dict]) -> None:
        """
        Initialize the model with the given configuration.

        Args:
            m_config (dict): A dictionary defining the model's architecture. 
                           It must include an 'input_channels' key and a 'layers' key, 
                           where 'layers' is a list of layer configurations.
        """
        super(Model, self).__init__()
        
        self._m_config = ModelConfiguration(m_config).__dict__

        self._layer_builders = self._initialize_layer_builders()

        self.model = self._build_model(self._m_config)
        
        del self._layer_builders
        
    
    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (Tensor): Input tensor.
        Returns:
            Tensor: Output tensor after passing through the model.
        """
        return self.model(x)

    def _initialize_layer_builders(self):
        """
        Define a mapping (registry) between layer types and their respective 
        builder functions. This registry allows for easy addition of new layer types.

        Returns:
            dict: A dictionary where keys are layer types (as strings) and values 
                  are the corresponding builder functions.
        """
        return {
            "conv2d": self._build_conv2d,
            "linear": self._build_linear,
            "maxpool2d": self._build_maxpool2d,
            "batchnorm2d": self._build_batchnorm2d,
            "batchnorm1d": self._build_batchnorm1d,
            "dropout": self._build_dropout,
            "flatten": self._build_flatten,
            "activation": self._build_activation,
        }

    def _build_model(self, _m_config):
        """
        Build the model as a sequential stack of layers based on the provided configuration.

        Args:
            _m_config (dict): The model configuration dictionary containing 'input_channels' 
                           and 'layers'.

        Returns:
            nn.Sequential: The constructed PyTorch sequential model.
        """
        layers = []
        input_channels = _m_config.get("input_channels", 1)  # Default to 1 channel if not specified
        for layer_config in _m_config["layers"]:
            layer_type = layer_config["type"].lower()  # Standardize layer type to lowercase
            # Check if the layer type is supported by the registry
            if layer_type in self._layer_builders:
                # Build the layer(s) and append them to the layers list
                layer = self._layer_builders[layer_type](layer_config, input_channels)
                layers.extend(layer if isinstance(layer, list) else [layer])
                # Update input_channels for subsequent layers if needed
                input_channels = self._update_input_channels(layer_type, layer_config, input_channels)
            else:
                raise ValueError(f"Unsupported layer type: {layer_type}")
        
        return nn.Sequential(*layers)

    def _build_conv2d(self, _m_config, input_channels):
        """
        Build a Conv2D layer.

        Args:
            _m_config (dict): Configuration for the Conv2D layer, including 'out_channels', 
                           'kernel_size', 'stride', and 'padding'.
            input_channels (int): Number of input channels to the Conv2D layer.

        Returns:
            nn.Conv2d: The constructed Conv2D layer.
        """
        return nn.Conv2d(
            in_channels=input_channels,
            out_channels=_m_config["out_channels"],
            kernel_size=_m_config["kernel_size"],
            stride=_m_config.get("stride", 1),
            padding=_m_config.get("padding", 0)
        )

    def _build_linear(self, _m_config, _):
        """
        Build a Linear (fully connected) layer.

        Args:
            _m_config (dict): Configuration for the Linear layer, including 'in_features' 
                           and 'out_features'.

        Returns:
            nn.Linear: The constructed Linear layer.
        """
        return nn.Linear(
            in_features=_m_config["in_features"],
            out_features=_m_config["out_features"],
            bias=False
        )

    def _build_maxpool2d(self, _m_config, _):
        """
        Build a MaxPool2D layer.

        Args:
            _m_config (dict): Configuration for the MaxPool2D layer, including 'kernel_size' 
                           and 'stride'.

        Returns:
            nn.MaxPool2d: The constructed MaxPool2D layer.
        """
        return nn.MaxPool2d(
            kernel_size=_m_config["kernel_size"],
            stride=_m_config.get("stride", 2)
        )
        
    def _build_batchnorm1d(self, _m_config, _):
        """
        Build a BatchNorm1D layer.

        Args:
            _m_config (dict): Configuration for the BatchNorm1D layer, including 'num_features'.

        Returns:
            nn.BatchNorm1d: The constructed BatchNorm1D layer.
        """
        return nn.BatchNorm1d(
            num_features=_m_config["num_features"],
            momentum=_m_config.get("momentum", 1.0),
            affine=_m_config.get("affine", True),
            track_running_stats=_m_config.get("track_running_stats", False),
        )

    def _build_batchnorm2d(self, _m_config, _):
        """
        Build a BatchNorm2D layer.

        Args:
            _m_config (dict): Configuration for the BatchNorm2D layer, including 'num_features'.

        Returns:
            nn.BatchNorm2d: The constructed BatchNorm2D layer.
        """
        return nn.BatchNorm2d(
            num_features=_m_config["num_features"]
        )

    def _build_dropout(self, _m_config, _):
        """
        Build a Dropout layer.

        Args:
            _m_config (dict): Configuration for the Dropout layer, including 'p' (dropout probability).

        Returns:
            nn.Dropout: The constructed Dropout layer.
        """
        return nn.Dropout(
            p=_m_config.get("p", 0.5)
        )

    def _build_flatten(self, _m_config, _):
        """
        Build a Flatten layer.

        Args:
            _m_config (dict): Configuration for the Flatten layer (no parameters needed).

        Returns:
            nn.Flatten: The constructed Flatten layer.
        """
        return nn.Flatten()

    def _build_activation(self, _m_config, _):
        """
        Build an activation layer.

        Args:
            _m_config (dict): Configuration for the activation layer, including 'activation' 
                           which specifies the activation type (e.g., 'relu', 'sigmoid').

        Returns:
            nn.Module: The constructed activation layer.
        """
        activations = {
            "relu": nn.ReLU,
            "leakyrelu": nn.LeakyReLU,
            "sigmoid": nn.Sigmoid,
            "tanh": nn.Tanh,
            "softmax": lambda: nn.Softmax(dim=_m_config.get("dim", 1)),
            "log_softmax": lambda: nn.LogSoftmax(dim=_m_config.get("dim", 1)),
        }
        activation_type = _m_config.get("activation", "relu").lower()
        if activation_type in activations:
            return activations[activation_type]()
        else:
            raise ValueError(f"Unsupported activation type: {activation_type}")

    def _update_input_channels(self, layer_type, _m_config, current_channels):
        """
        Update the number of input channels for the next layer, if applicable.

        Args:
            layer_type (str): The type of the current layer (e.g., 'conv2d').
            _m_config (dict): Configuration of the current layer.
            current_channels (int): The current number of input channels.

        Returns:
            int: The updated number of input channels.
        """
        if layer_type == "conv2d":
            return _m_config["out_channels"]
        elif layer_type == "batchnorm1d":
            return _m_config["num_features"]
        return current_channels
    
    
    def get_optimizer(self, learning_rate=0.01):
        """
        Get the optimizer based on the configuration.

        Returns:
            torch.optim.Optimizer: The optimizer object.
        """
        optimizer_type = self._m_config.get("optimizer", {}).lower()
        if optimizer_type == "adam":
            return torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        elif optimizer_type == "sgd":
            return torch.optim.SGD(self.model.parameters(), lr=learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
        
    
    def get_loss_function(self):
        """
        Get the loss function based on the configuration.

        Returns:
            torch.nn.Module: The loss function object.
        """
        loss_function_type = self._m_config.get("loss_function", {}).lower()
        if loss_function_type == "cross_entropy":
            return nn.CrossEntropyLoss()
        elif loss_function_type == "nll":
            return nn.NLLLoss()
        else:
            raise ValueError(f"Unsupported loss function type: {loss_function_type}")
