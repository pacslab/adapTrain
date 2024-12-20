from .configuration import Configuration

from typing import TextIO, Union, Dict


class PartitioningConfiguration(Configuration):
    def __init__(self, config_file: Union[TextIO, Dict]):
        super(PartitioningConfiguration, self).__init__(config_file)
    
    
    def _load_config_json_schema(self):
        """
        Load the JSON schema for the partition configuration file.
        """
        self._config_json_schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "Partitioning Configuration Schema",
            "description": "Schema for the partitioning configuration file.",
            "type": "object",
            "properties": {
                "repartition_iter": {
                "type": "integer",
                "minimum": 1,
                "deafult": 150,
                "description": "Number of iterations after which to repartition the model."
                },
                "log_interval": {
                "type": "integer",
                "minimum": 1,
                "default": 25,
                "description": "Number of iterations after which to log the training loss."
                }
            },
            "required": [],
            "additionalProperties": False
        }
