import random
import torch
import torch.distributed as dist
from torch.utils.data import Subset, DataLoader
from torchvision import datasets, transforms

import numpy as np

from tqdm import tqdm

from multiprocessing import Process

from kubernetes import client, config

from typing import TextIO, Union, Dict, List

from src.configurations import ModelConfiguration
from src.configurations import DeploymentConfiguration
from src.configurations import PartitioningConfiguration
from src.configurations.defaults import LAYERS_TO_PARTITION

from src.utils import partition_layer
from src.utils import update_layer
from src.utils import adaptive_partitioning

from src.dataset import AdapTrainDataset

from src.models import Model

from src.logger import logger


class Controller:
    
    def __init__(self,
                 m_config: Union[TextIO, Dict],
                 p_config: Union[TextIO, Dict],
                 d_config: Union[TextIO, Dict],
                 dataset_path: str = None) -> None:
        
        if dataset_path is None:
            raise ValueError("Please provide the dataset path.")
        
        torch.manual_seed(1)
        
        self._d_config = DeploymentConfiguration(d_config)
        self._m_config = ModelConfiguration(m_config).__dict__
        self._p_config = PartitioningConfiguration(p_config).__dict__
        
        self.dataset_path = dataset_path
        
        self.raw_model = Model(m_config)
        
        logger.info(f"Model for training: {self.raw_model}")

        # partitioned_indices includes the indices of the layers to partition for each worker
        self.layers_to_partition_indices, \
            self.partitioned_indices_per_workers_per_layer, \
                self.partitioned_parameters_to_workers_per_layer = self._extract_layers_to_partition()

        self.num_of_workers = self._d_config.num_workers
        self.workers_image = self._d_config.workers_image
        self.node_names = self._d_config.node_names
        self.namespace = self._d_config.namespace
        self.service_url = self._get_service_info()
        
        # Initial partition coefficients
        self.partition_coefficients = torch.tensor([1 / self._d_config.num_workers] * self._d_config.num_workers)
        
        self.training_round_times = torch.tensor([0.0] * self._d_config.num_workers)

        
        self._init_nodes()
        self._init_controller()
        
        # self._scheduler()
        
        # self._destroy()

        
        
    def _init_controller(self):
        dist.init_process_group(backend=self._d_config.dist_backend,
                                init_method=self._d_config.dist_url,
                                rank=0,
                                world_size=self._d_config.num_workers + 1)

        self._scheduler()



    def _get_service_info(self):
        config.load_incluster_config()
        api = client.CoreV1Api()
        service = api.read_namespaced_service("adaptrain-service", self.namespace)

        if service.spec.ports and service.spec.cluster_ip:
            service_ip = service.spec.cluster_ip
            service_port = service.spec.ports[0].port
            return f"tcp://{service_ip}:{service_port}"
        else:
            raise ValueError(f"No ports found for service adaptrain-service in namespace {self.namespace}")


    def _deploy_worker(self, worker_rank):
        config.load_incluster_config()
        
        container = client.V1Container(
            name=f"worker-{worker_rank}",
            image=self.workers_image,
            image_pull_policy="Always",
            stdin=True,
            tty=True,
            args=[
                "--rank", f"{worker_rank}",
                "--dist-url", f"{self.service_url}"
                ],
        )

        pod_spec = client.V1PodSpec(
            containers=[container],
            restart_policy="Never",
            node_name=self.node_names[worker_rank - 1]
        )

        pod_metadata = client.V1ObjectMeta(
            name=f"adaptrain-worker-{worker_rank}",
            namespace=self.namespace,
            labels={"app.kubernetes.io/name": f"adaptrain-worker-{worker_rank}"}
        )

        pod = client.V1Pod(
            api_version="v1",
            kind="Pod",
            metadata=pod_metadata,
            spec=pod_spec,
        )

        api = client.CoreV1Api()
        api.create_namespaced_pod(namespace=self.namespace, body=pod)
        logger.info(f"Worker {worker_rank} deployed successfully.")
        
    
    # def _destroy_nodes(self):
    #     for worker in tqdm(self.workers):
    #         worker.destroy()
            
    #     # logger.info("All nodes have been destroyed.")
    
    
    def _init_nodes(self):
        for rank in tqdm(range(1, self.num_of_workers + 1)):
            self._deploy_worker(rank)
            
        logger.info("All nodes have been deployed.")
        
    
    def _generate_partitions(self):
        self.layers_to_partition_indices, \
            self.partitioned_indices_per_workers_per_layer, \
                self.partitioned_parameters_to_workers_per_layer = self._extract_layers_to_partition()
                
        linear_layers_to_partition = [layer_i for layer_i in self.layers_to_partition_indices if self.raw_model.model[layer_i].__class__.__name__ == "Linear"]
        for i, layer_i in enumerate(self.layers_to_partition_indices):
            if self.raw_model.model[layer_i].__class__.__name__ == "Linear":
                if layer_i != linear_layers_to_partition[-1]:
                    layer_size = self.raw_model.model[layer_i].out_features
                else:
                    layer_size = self.raw_model.model[layer_i].in_features
                partition_dim_0 = True if layer_i != linear_layers_to_partition[-1] else False
                partition_dim_1 = True if layer_i != linear_layers_to_partition[0] else False
            elif self.raw_model.model[layer_i].__class__.__name__ == "BatchNorm1d":
                layer_size = self.raw_model.model[layer_i].num_features
                partition_dim_0 = True
                partition_dim_1 = False
            
            if self.raw_model.model[layer_i].__class__.__name__ == "BatchNorm1d" or layer_i == linear_layers_to_partition[-1]:
                previous_linear_layer_index = self.layers_to_partition_indices.index(max((idx for idx in linear_layers_to_partition if idx < layer_i), default=None))
                self.partitioned_indices_per_workers_per_layer[i] = self.partitioned_indices_per_workers_per_layer[previous_linear_layer_index]
            else:
                
                layer_neurons = np.arange(layer_size)
                np.random.shuffle(layer_neurons)
                
                cursor = 0
                for worker_index in range(self.num_of_workers):
                    if worker_index < self.num_of_workers - 1:
                        end_cursor = cursor + int(self.partition_coefficients[worker_index] * layer_size)
                    else:
                        end_cursor = layer_size

                    # Generate current indexes based on whether coefficients are equal
                    current_indexes = torch.tensor(layer_neurons[cursor:end_cursor])

                    # Update cursor and log the indexes
                    cursor = end_cursor
                    self.partitioned_indices_per_workers_per_layer[i][worker_index] = current_indexes.detach().clone()

            previous_linear_layer_index = self.layers_to_partition_indices.index(max((idx for idx in linear_layers_to_partition if idx < layer_i), default=0))

            dim_0_indices = self.partitioned_indices_per_workers_per_layer[i] if partition_dim_0 else None
            dim_1_indices = self.partitioned_indices_per_workers_per_layer[previous_linear_layer_index] if partition_dim_1 else None
   
            partitioned_weights, partitioned_biases = partition_layer(
                self.raw_model.model[layer_i],
                True,
                False if self.raw_model.model[layer_i].__class__.__name__ == "Linear" else True,
                dim_0_indices,
                dim_1_indices
            )
            
            self.partitioned_parameters_to_workers_per_layer[i] = (partitioned_weights, partitioned_biases)
    
    
    def _receive_training_round_times(self):
        for i in range(self.num_of_workers):
            dist.recv(tensor=self.training_round_times[i], src=i + 1)
        
        self.partition_coefficients = adaptive_partitioning(self.training_round_times, self.partition_coefficients)
        
        logger.info(f"New Partitioning Distribution: {self.partition_coefficients}")
    
    
    def _send_coefficients_to_workers(self):
        for i in range(self.num_of_workers):
            dist.send(tensor=torch.tensor(self.partition_coefficients), dst=i + 1)
    
    
    def _send_partitions_to_workers(self):
        for layer_i, layer in enumerate(self.raw_model.model):
            if layer.__class__ in LAYERS_TO_PARTITION and layer_i in self.layers_to_partition_indices:
                i = self.layers_to_partition_indices.index(layer_i)
                partitioned_weights, partitioned_biases = self.partitioned_parameters_to_workers_per_layer[i]
                
                for worker_i in range(self.num_of_workers):
                    if layer.weight is not None:
                        dist.send(tensor=partitioned_weights[worker_i], dst=worker_i + 1)
                    if layer.bias is not None:
                        dist.send(tensor=partitioned_biases[worker_i], dst=worker_i + 1)
            else:
                for worker_i in range(self.num_of_workers):
                    if hasattr(layer, 'weight') and layer.weight is not None:
                        dist.send(tensor=layer.weight.data, dst=worker_i + 1)
                    if hasattr(layer, 'bias') and layer.bias is not None:
                        dist.send(tensor=layer.weight.data, dst=worker_i + 1)

    
    def _receive_learned_partitions_from_workers(self):
        linear_layers_to_partition = [layer_i for layer_i in self.layers_to_partition_indices if self.raw_model.model[layer_i].__class__.__name__ == "Linear"]

        for layer_i, layer in enumerate(self.raw_model.model):
            if layer.__class__ in LAYERS_TO_PARTITION and layer_i in self.layers_to_partition_indices:
                i = self.layers_to_partition_indices.index(layer_i)
                partitioned_weights, partitioned_biases = self.partitioned_parameters_to_workers_per_layer[i]
                
                for worker_i in range(self.num_of_workers):
                    if layer.weight is not None:
                        dist.recv(tensor=partitioned_weights[worker_i], src=worker_i + 1)
                    if layer.bias is not None:
                        dist.recv(tensor=partitioned_biases[worker_i], src=worker_i + 1)
                        
                if layer.__class__.__name__ == "Linear":
                    partition_dim_0 = True if layer_i != linear_layers_to_partition[-1] else False
                    partition_dim_1 = True if layer_i != linear_layers_to_partition[0] else False
                elif layer.__class__.__name__ == "BatchNorm1d":
                    partition_dim_0 = True
                    partition_dim_1 = False

                previous_linear_layer_index = self.layers_to_partition_indices.index(max((idx for idx in linear_layers_to_partition if idx < layer_i), default=0))                        
                dim_0_indices = self.partitioned_indices_per_workers_per_layer[i] if partition_dim_0 else None
                dim_1_indices = self.partitioned_indices_per_workers_per_layer[previous_linear_layer_index] if partition_dim_1 else None
                
                new_layer = update_layer(layer, partitioned_weights, partitioned_biases, dim_0_indices, dim_1_indices)

                self.raw_model.model[layer_i] = new_layer
        
    
    
    def _extract_layers_to_partition(self):
        """
        Extract the layers to partition from the model.
        
        Returns:
            list: A list containing the extracted layers to partition.
        """
        extracted_layers_indices = []
        
        for i, layer in enumerate(self.raw_model.model):
            if layer.__class__ in LAYERS_TO_PARTITION:
                extracted_layers_indices.append(i)
                
        if self.raw_model.model[extracted_layers_indices[-1]].__class__.__name__ != "Linear":
            extracted_layers_indices.pop()
                
        return (extracted_layers_indices, 
                [[[] for _ in range(self._d_config.num_workers)] for _ in range(len(extracted_layers_indices))],
                [([], []) for _ in range(len(extracted_layers_indices))])
        
        
        
    def _scheduler(self):
        """__summary__
        Main function to schedule the training process with the workers.
        """

        self._generate_partitions()
        
        num_epochs = self._m_config["num_epochs"]
        batch_size = self._m_config["batch_size"]

        train_set = AdapTrainDataset(X_path=f"{self.dataset_path}/train_x.npy", y_path=f"{self.dataset_path}/train_y.npy")
        
        test_set = AdapTrainDataset(X_path=f"{self.dataset_path}/test_x.npy", y_path=f"{self.dataset_path}/test_y.npy")

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,
                                drop_last=True)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,
                                drop_last=False)
        
        subset_size = len(train_loader) // self.num_of_workers
        
        for epoch_i in range(num_epochs):
            self.raw_model.train()
            for batch_i in range(subset_size):
                if batch_i % self._p_config["repartition_iter"] == 0:
                    if epoch_i > 0 or batch_i > 0:
                        logger.info(f"Controller: Repartitioning at epoch {epoch_i}, batch {batch_i}")
                        self._send_coefficients_to_workers()
                        self._generate_partitions()
                        self._send_partitions_to_workers()
                        logger.info(f"Controller: Partitions sent to workers")
                
                if (batch_i + 1) % self._p_config["repartition_iter"] == 0 or batch_i == subset_size - 1:
                    logger.info(f"Controller: Receiving learned partitions from workers")
                    self._receive_learned_partitions_from_workers()
                    self._receive_training_round_times()
                    logger.info(f"Controller: Learned partitions received from workers and updated the raw model")
            
            logger.info(f"Epoch {epoch_i} finished.")
            
            self._test_and_save_model(test_loader)
      
            
    def _test_and_save_model(self, test_loader):
        self.raw_model.eval()

        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for _, (X, y) in enumerate(test_loader):
                output = self.raw_model(X)
                
                test_pred = output.max(1, keepdim=True)[1]
                test_correct += test_pred.eq(y.view_as(test_pred)).sum().item()
                test_total += y.shape[0]

            test_acc = float(test_correct) / float(test_total)

        logger.info(f"Accuracy: %{test_acc * 100:.2f} Correct: {test_correct}, Total: {test_total}")
        
        model_name = f"model_acc_{test_acc:.4f}.pth"
        
        torch.save(self.raw_model.state_dict(), model_name)
        
        logger.info(f"Model saved as {model_name} with accuracy %{test_acc * 100:.2f}")