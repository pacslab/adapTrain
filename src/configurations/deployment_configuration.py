from .configuration import Configuration

from typing import TextIO, Union, Dict


class DeploymentConfiguration(Configuration):
    def __init__(self, config_file: Union[TextIO, Dict]):
        super(DeploymentConfiguration, self).__init__(config_file)
    
    
    def _load_config_json_schema(self):
        """
        Load the JSON schema for the deployment configuration file.
        """
        self._config_json_schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "Deployment Configuration Schema",
            "description": "Schema for the deployment configuration file.",
            "type": "object",
            "properties": {
                "device": {
                    "type": "string",
                    "description": "Device to use for training.",
                    "enum": ["cpu", "cuda"],
                    "default": "cpu"
                },
                "dist_backend": {
                    "type": "string",
                    "default": "gloo",
                    "enum": [
                        "gloo",
                        "nccl",
                    ],
                    "description": "Distributed backend to use for training."
                },
                "dist_url": {
                    "type": "string",
                    "default": "tcp://127.0.0.1:9000",
                    "description": "URL to connect to for distributed training."
                },
                "num_workers": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Number of worker nodes to use for distributed training."
                },
                "workers_url": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "URLs for the worker nodes."
                },
                "workers_image": {
                    "type": "string",
                    "default": "genericdockerhub/adaptrain-worker:latest",
                    "description": "URL of the workers' docker image"
                },
                "namespace": {
                    "type": "string",
                    "default": "adaptrain",
                    "description": "The namespace used for deployment"
                },
                "node_names": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "names of the nodes in the cluster to be used for deployment"
                }
            },
            "required": [],
            "additionalProperties": False
        }

