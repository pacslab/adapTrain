from src.models import Model
import json

from src import main

vgg16_config = """
{
    "input_channels": 3,
    "layers": [
        {"type": "conv2d", "out_channels": 64, "kernel_size": 3, "padding": 1},
        {"type": "activation", "activation": "relu"},
        {"type": "conv2d", "out_channels": 64, "kernel_size": 3, "padding": 1},
        {"type": "activation", "activation": "relu"},
        {"type": "maxpool2d", "kernel_size": 2, "stride": 2},

        {"type": "conv2d", "out_channels": 128, "kernel_size": 3, "padding": 1},
        {"type": "activation", "activation": "relu"},
        {"type": "conv2d", "out_channels": 128, "kernel_size": 3, "padding": 1},
        {"type": "activation", "activation": "relu"},
        {"type": "maxpool2d", "kernel_size": 2, "stride": 2},

        {"type": "conv2d", "out_channels": 256, "kernel_size": 3, "padding": 1},
        {"type": "activation", "activation": "relu"},
        {"type": "conv2d", "out_channels": 256, "kernel_size": 3, "padding": 1},
        {"type": "activation", "activation": "relu"},
        {"type": "conv2d", "out_channels": 256, "kernel_size": 3, "padding": 1},
        {"type": "activation", "activation": "relu"},
        {"type": "maxpool2d", "kernel_size": 2, "stride": 2},

        {"type": "conv2d", "out_channels": 512, "kernel_size": 3, "padding": 1},
        {"type": "activation", "activation": "relu"},
        {"type": "conv2d", "out_channels": 512, "kernel_size": 3, "padding": 1},
        {"type": "activation", "activation": "relu"},
        {"type": "conv2d", "out_channels": 512, "kernel_size": 3, "padding": 1},
        {"type": "activation", "activation": "relu"},
        {"type": "maxpool2d", "kernel_size": 2, "stride": 2},

        {"type": "conv2d", "out_channels": 512, "kernel_size": 3, "padding": 1},
        {"type": "activation", "activation": "relu"},
        {"type": "conv2d", "out_channels": 512, "kernel_size": 3, "padding": 1},
        {"type": "activation", "activation": "relu"},
        {"type": "conv2d", "out_channels": 512, "kernel_size": 3, "padding": 1},
        {"type": "activation", "activation": "relu"},
        {"type": "maxpool2d", "kernel_size": 2, "stride": 2},

        {"type": "flatten"},
        {"type": "linear", "in_features": 25088, "out_features": 4096},
        {"type": "activation", "activation": "relu"},
        {"type": "dropout", "p": 0.5},

        {"type": "linear", "in_features": 4096, "out_features": 4096},
        {"type": "activation", "activation": "relu"},
        {"type": "dropout", "p": 0.5},

        {"type": "linear", "in_features": 4096, "out_features": 1000}
    ]
}
"""

# Load the JSON string as a Python dictionary
# config = json.loads(vgg16_config)
# model = Model(config)
# print(model)

if __name__ == "__main__":
    main.main()

