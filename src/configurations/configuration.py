from abc import ABC, abstractmethod

from typing import Union, TextIO, Dict

import json
import jsonschema


class Configuration(ABC):
    def __init__(self, config_file: Union[TextIO, Dict]):
        self._load_config_json_schema()
        
        self._deserialize(config_file=config_file)
        

    @abstractmethod
    def _load_config_json_schema(self):
        """
        Load the JSON schema for the configuration file.
        """
        pass
    
    
    def _apply_defaults(self, config_dict: Dict):
        """
        Apply default values from the schema to the configuration if not provided by the user.
        """
        for key, value in self._config_json_schema["properties"].items():
            if "default" in value and key not in config_dict:
                config_dict[key] = value["default"]
        return config_dict
    

    def _deserialize(self, config_file: Union[TextIO, Dict]):
        """
        Deserialize the configuration file.
        
        Args:
            config_file (Union[TextIO, Dict]): The configuration file to deserialize.
        """
        try:
            if isinstance(config_file, dict):
                config_dict = config_file
                
            else:
                config_dict = json.load(config_file)
                
            config_dict = self._apply_defaults(config_dict)
                
            jsonschema.validate(instance=config_dict, schema=self._config_json_schema)  
        
        except json.decoder.JSONDecodeError as e:
            raise ValueError(
                f'Please provide a valid JSON file. Error: {e}'
            )
            
        except jsonschema.exceptions.ValidationError as e:
            raise ValueError(
                f'Invalid configuration file. Error: {e}'
            )
            
        else:
            self.__dict__.update(**config_dict)
            del self.__dict__["_config_json_schema"]
    
    