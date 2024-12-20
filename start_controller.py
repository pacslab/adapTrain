import os
import json
import time

from src.nodes import Controller
from src.logger import logger


def main():
    if not os.path.exists('./configs'):
        raise FileNotFoundError("Please put the configuration files in the 'app/configs' folder.")


    m_config, m_config_path = None, os.getenv('M_CONFIG_PATH', './configs/m_config.json')
    p_config, p_config_path = None, os.getenv('P_CONFIG_PATH', './configs/p_config.json')
    d_config, d_config_path = None, os.getenv('D_CONFIG_PATH', './configs/d_config.json')


    if not os.path.exists(m_config_path):
        raise FileNotFoundError(f"Model configuration file not found at {m_config_path}.")

    if not os.path.exists(p_config_path):
        raise FileNotFoundError(f"Partitioner configuration file not found at {p_config_path}.")

    if not os.path.exists(d_config_path):
        raise FileNotFoundError(f"Dataset configuration file not found at {d_config_path}.")


    with open(m_config_path, 'r') as f:
        m_config = json.load(f)

    with open(p_config_path, 'r') as f:
        p_config = json.load(f)
        
    with open(d_config_path, 'r') as f:
        d_config = json.load(f)
        
    
    dataset_path = os.getenv('DATASET_PATH', './dataset')
        

    if not os.path.exists(dataset_path):
        raise FileNotFoundError("Please put the dataset in the 'app/dataset' folder.")
    
    
    data = [
        'train_x.npy',
        'train_y.npy',
        'test_x.npy',
        'test_y.npy'
    ]

    for d_name in data:
        if not os.path.exists(f'./dataset/{d_name}'):
            raise FileNotFoundError(f"Please put the {d_name} file in the 'app/dataset' folder.")

        
    controller = Controller(m_config, p_config, d_config, dataset_path)



if __name__ == '__main__':
    main()
