import time
from random import shuffle
import torch.nn as nn
import argparse
import numpy as np
import google_speech_data_loader as speech_dataset
from torch.utils.data import DataLoader
import torch.nn.parallel
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from ist_utils import *


class DNNGoogleSpeechBatchNorm2Layer(nn.Module):
    def __init__(self, partition_num=1, sample_size=4096, model_size=4096, label_num=35):
        super(DNNGoogleSpeechBatchNorm2Layer, self).__init__()
        self.partition_num = partition_num
        self.partition_dim = model_size // partition_num
        self.temp_hidden_layer_index = [i for i in range(model_size)]
        self.fc1 = nn.Linear(sample_size, model_size, False)
        self.bn1 = nn.BatchNorm1d(model_size, momentum=1.0, track_running_stats=False)
        self.fc2 = nn.Linear(model_size, label_num, False)
        self.bn2 = nn.BatchNorm1d(label_num, momentum=1.0, affine=False, track_running_stats=False)
        # The following is used for distributed training.
        if partition_num != 1:
            self.hidden_layer_index_log = []
            self.fc1_weight_partition = []
            self.bn1_weight_partition = []
            self.bn1_bias_partition = []
            self.fc2_weight_partition = []

    def forward(self, x):
        x = self.fc1(x)
        # print(x[0])
        x = self.bn1(x)
        # print(x[0])
        x = nn.functional.relu(x, inplace=True)
        # print(x[0])
        x = self.fc2(x)
        # print(x[0])
        x = self.bn2(x)
        x = nn.functional.log_softmax(x, dim=1)
        return x

    def partition_to_list(self, coefs):
        # coefs = args.init_partitions
        print("Repartition parameters!")
        shuffle(self.temp_hidden_layer_index)
        self.hidden_layer_index_log.clear()
        cursor = 0 #added
        for i in range(self.partition_num):
            end_cursor = cursor + int(coefs[i]*self.partition_dim*self.partition_num) if i<self.partition_num - 1 else len(self.temp_hidden_layer_index) #added
            # current_indexes = torch.tensor(self.temp_hidden_layer_index[i * self.partition_dim:(i + 1) * self.partition_dim]) #commented
            current_indexes = torch.tensor(self.temp_hidden_layer_index[cursor:end_cursor]) if not np.all(coefs == coefs[0]) else torch.tensor(self.temp_hidden_layer_index[i * self.partition_dim:(i + 1) * self.partition_dim])
            cursor = end_cursor #added
            self.hidden_layer_index_log.append(current_indexes)

        self.fc1_weight_partition.clear()
        self.bn1_weight_partition.clear()
        self.bn1_bias_partition.clear()
        self.fc2_weight_partition.clear()
        self.fc1_weight_partition = partition_FC_layer_by_output_dim_0(
            self.fc1.weight, self.hidden_layer_index_log)
        self.bn1_weight_partition, self.bn1_bias_partition = partition_BN_layer(
            self.bn1.weight, self.bn1.bias, self.hidden_layer_index_log)
        self.fc2_weight_partition = partition_FC_layer_by_input_dim_1(
            self.fc2.weight, self.hidden_layer_index_log)

    def flush(self):
        # update the model based on the collected parameters.
        # Here we have to get around pytorch variable by use variable.data,
        # since leaf variable disallowing in-place operation
        update_tensor_by_update_lists_dim_0(self.fc1.weight.data, self.fc1_weight_partition,
                                            self.hidden_layer_index_log)
        update_tensor_by_update_lists_dim_0(self.bn1.weight.data, self.bn1_weight_partition,
                                            self.hidden_layer_index_log)
        update_tensor_by_update_lists_dim_0(self.bn1.bias.data, self.bn1_bias_partition,
                                            self.hidden_layer_index_log)
        update_tensor_by_update_lists_dim_1(self.fc2.weight.data, self.fc2_weight_partition,
                                            self.hidden_layer_index_log)



class DNNGoogleSpeechBatchNorm3Layer(nn.Module):
    def __init__(self, partition_num=1, sample_size=4096, model_size=4096, label_num=35):
        super(DNNGoogleSpeechBatchNorm3Layer, self).__init__()
        self.partition_num = partition_num
        self.partition_dim = model_size // partition_num
        self.temp_hidden_layer_index1 = [i for i in range(model_size)]
        self.temp_hidden_layer_index2 = [i for i in range(model_size)]
        self.fc1 = nn.Linear(sample_size, model_size, False)
        self.bn1 = nn.BatchNorm1d(model_size, momentum=1.0, affine=True, track_running_stats=False)
        self.fc2 = nn.Linear(model_size, model_size, False)
        self.bn2 = nn.BatchNorm1d(model_size, momentum=1.0, affine=True, track_running_stats=False)
        self.fc3 = nn.Linear(model_size, label_num, False)
        self.bn3 = nn.BatchNorm1d(label_num, momentum=1.0, affine=False, track_running_stats=False)
        # The following is used for distributed training.
        if partition_num != 1:
            self.hidden_layer_index_log1 = []
            self.fc1_weight_partition = []
            self.bn1_weight_partition = []
            self.bn1_bias_partition = []
            self.hidden_layer_index_log2 = []
            self.fc2_weight_partition = []
            self.bn2_weight_partition = []
            self.bn2_bias_partition = []
            self.fc3_weight_partition = []

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = nn.functional.relu(x, inplace=True)
        x = self.fc2(x)
        x = self.bn2(x)
        x = nn.functional.relu(x, inplace=True)
        x = self.fc3(x)
        x = self.bn3(x)
        x = nn.functional.log_softmax(x, dim=1)
        return x

    def partition_to_list(self, coefs):
        print("Repartition parameters!")

        shuffle(self.temp_hidden_layer_index1)
        self.hidden_layer_index_log1.clear()
        cursor = 0 #added
        for i in range(self.partition_num):
            end_cursor = cursor + int(coefs[i]*self.partition_dim*self.partition_num) if i<self.partition_num - 1 else len(self.temp_hidden_layer_index1) #added
            current_indexes = torch.tensor(self.temp_hidden_layer_index1[cursor:end_cursor]) if not np.all(coefs == coefs[0]) else torch.tensor(self.temp_hidden_layer_index1[i * self.partition_dim:(i + 1) * self.partition_dim])
            cursor = end_cursor #added
            self.hidden_layer_index_log1.append(current_indexes)

        shuffle(self.temp_hidden_layer_index2)
        self.hidden_layer_index_log2.clear()
        cursor = 0
        for i in range(self.partition_num):
            end_cursor = cursor + int(coefs[i]*self.partition_dim*self.partition_num) if i<self.partition_num - 1 else len(self.temp_hidden_layer_index2) #added
            current_indexes = torch.tensor(self.temp_hidden_layer_index2[cursor:end_cursor]) if not np.all(coefs == coefs[0]) else torch.tensor(self.temp_hidden_layer_index2[i * self.partition_dim:(i + 1) * self.partition_dim])
            cursor = end_cursor #added
            self.hidden_layer_index_log2.append(current_indexes)

        self.fc1_weight_partition.clear()
        self.bn1_weight_partition.clear()
        self.bn1_bias_partition.clear()
        self.fc2_weight_partition.clear()
        self.bn2_weight_partition.clear()
        self.bn2_bias_partition.clear()
        self.fc3_weight_partition.clear()

        self.fc1_weight_partition = partition_FC_layer_by_output_dim_0(
            self.fc1.weight, self.hidden_layer_index_log1)
        self.bn1_weight_partition, self.bn1_bias_partition = partition_BN_layer(
            self.bn1.weight, self.bn1.bias, self.hidden_layer_index_log1)
        self.fc2_weight_partition = partition_FC_layer_by_dim_01(
            self.fc2.weight, self.hidden_layer_index_log2, self.hidden_layer_index_log1)
        self.bn2_weight_partition, self.bn2_bias_partition = partition_BN_layer(
            self.bn2.weight, self.bn2.bias, self.hidden_layer_index_log2)
        self.fc3_weight_partition = partition_FC_layer_by_input_dim_1(
            self.fc3.weight, self.hidden_layer_index_log2)

    def flush(self):
        print('update the model based on collected parameters!')
        # update the model based on the collected parameters.
        # Here we have to get around pytorch variable by use variable.data,
        # since leaf variable disallowing in-place operation
        update_tensor_by_update_lists_dim_0(self.fc1.weight.data, self.fc1_weight_partition,
                                            self.hidden_layer_index_log1)
        update_tensor_by_update_lists_dim_0(self.bn1.weight.data, self.bn1_weight_partition,
                                            self.hidden_layer_index_log1)
        update_tensor_by_update_lists_dim_0(self.bn1.bias.data, self.bn1_bias_partition,
                                            self.hidden_layer_index_log1)
        update_tensor_by_update_lists_dim_01(self.fc2.weight.data, self.fc2_weight_partition,
                                             self.hidden_layer_index_log2, self.hidden_layer_index_log1)
        update_tensor_by_update_lists_dim_0(self.bn2.weight.data, self.bn2_weight_partition,
                                            self.hidden_layer_index_log2)
        update_tensor_by_update_lists_dim_0(self.bn2.bias.data, self.bn2_bias_partition,
                                            self.hidden_layer_index_log2)
        update_tensor_by_update_lists_dim_1(self.fc3.weight.data, self.fc3_weight_partition,
                                            self.hidden_layer_index_log2)





def dispatch_model_to_workers(args, partitioned_model, raw_model=None, sync_time=0, coefs=None):
    print('dispatch_model_to_workers called!')
    
    print(f"Coefs: {coefs}")
    if args.dynamic and sync_time > 0:
        new_coefs = np.array(coefs)
        print(f"Old Coefs: {coefs}")
        new_coefs, new_partition_size = decide_new_partition_size(args, sync_time, coefs)
        coefs[:] = new_coefs
        print(f"New Coefs: {new_coefs}")
        
        if args.layers == 2:
            partitioned_model.partition_num = 1
            partitioned_model.partition_dim = new_partition_size
            partitioned_model.temp_hidden_layer_index = [i for i in range(new_partition_size)]
            partitioned_model.fc1 = nn.Linear(4096, new_partition_size, False)
            partitioned_model.bn1 = nn.BatchNorm1d(new_partition_size, momentum=1.0, track_running_stats=False)
            partitioned_model.fc2 = nn.Linear(new_partition_size, 35, False)
            partitioned_model.bn2 = nn.BatchNorm1d(35, momentum=1.0, affine=False, track_running_stats=False)
        elif args.layers == 3:
            partitioned_model.partition_num = 1
            partitioned_model.partition_dim = new_partition_size
            partitioned_model.temp_hidden_layer_index1 = [i for i in range(new_partition_size)]
            partitioned_model.temp_hidden_layer_index2 = [i for i in range(new_partition_size)]
            partitioned_model.fc1 = nn.Linear(4096, new_partition_size, False)
            partitioned_model.bn1 = nn.BatchNorm1d(new_partition_size, momentum=1.0, affine=True, track_running_stats=False)
            partitioned_model.fc2 = nn.Linear(new_partition_size, new_partition_size, False)
            partitioned_model.bn2 = nn.BatchNorm1d(new_partition_size, momentum=1.0, affine=True, track_running_stats=False)
            partitioned_model.fc3 = nn.Linear(new_partition_size, 35, False)
            partitioned_model.bn3 = nn.BatchNorm1d(35, momentum=1.0, affine=False, track_running_stats=False)

    
    if args.rank == 0:
        assert(raw_model is not None)
        raw_model.partition_to_list(coefs)
        # print(raw_model.fc1_weight_partition[0].shape)
        # print(raw_model.fc1_weight_partition[1].shape)
        # print(raw_model.fc1_weight_partition[2].shape)
        # print(raw_model.fc1_weight_partition[3].shape)
        
        partitioned_model.fc1.weight.data = raw_model.fc1_weight_partition[0]
        partitioned_model.fc2.weight.data = raw_model.fc2_weight_partition[0]
        partitioned_model.bn1.weight.data = raw_model.bn1_weight_partition[0]
        partitioned_model.bn1.bias.data = raw_model.bn1_bias_partition[0]
        if args.layers == 3:
            partitioned_model.fc3.weight.data = raw_model.fc3_weight_partition[0]
            partitioned_model.bn2.weight.data = raw_model.bn2_weight_partition[0]
            partitioned_model.bn2.bias.data = raw_model.bn2_bias_partition[0]

        if not np.all(coefs == coefs[0]):
            for i in range(1,args.world_size):
                dist.send(tensor=raw_model.fc1_weight_partition[i], dst=i)
                dist.send(tensor=raw_model.fc2_weight_partition[i], dst=i)
                dist.send(tensor=raw_model.bn1_weight_partition[i], dst=i)
                dist.send(tensor=raw_model.bn1_bias_partition[i], dst=i)
                if args.layers == 3:
                    dist.send(tensor=raw_model.fc3_weight_partition[i], dst=i)
                    dist.send(tensor=raw_model.bn2_weight_partition[i], dst=i)
                    dist.send(tensor=raw_model.bn2_bias_partition[i], dst=i)
        else:
            dist.scatter(tensor=partitioned_model.fc1.weight.data, scatter_list=raw_model.fc1_weight_partition, src=0) #commented
            dist.scatter(tensor=partitioned_model.fc2.weight.data, scatter_list=raw_model.fc2_weight_partition, src=0) #commented
            dist.scatter(tensor=partitioned_model.bn1.weight.data, scatter_list=raw_model.bn1_weight_partition, src=0) #commented
            dist.scatter(tensor=partitioned_model.bn1.bias.data, scatter_list=raw_model.bn1_bias_partition, src=0) #commented
            if args.layers == 3:
                dist.scatter(tensor=partitioned_model.fc3.weight.data, scatter_list=raw_model.fc3_weight_partition, src=0)
                dist.scatter(tensor=partitioned_model.bn2.weight.data, scatter_list=raw_model.bn2_weight_partition, src=0)
                dist.scatter(tensor=partitioned_model.bn2.bias.data, scatter_list=raw_model.bn2_bias_partition, src=0)

    else:

        if not np.all(coefs == coefs[0]):
            dist.recv(tensor=partitioned_model.fc1.weight.data, src=0)
            dist.recv(tensor=partitioned_model.fc2.weight.data, src=0)
            dist.recv(tensor=partitioned_model.bn1.weight.data, src=0)
            dist.recv(tensor=partitioned_model.bn1.bias.data, src=0)
            if args.layers == 3:
                dist.recv(tensor=partitioned_model.fc3.weight.data, src=0)
                dist.recv(tensor=partitioned_model.bn2.weight.data, src=0)
                dist.recv(tensor=partitioned_model.bn2.bias.data, src=0)
        else:
            dist.scatter(tensor=partitioned_model.fc1.weight.data, scatter_list=[], src=0) #commented
            dist.scatter(tensor=partitioned_model.fc2.weight.data, scatter_list=[], src=0) #commented
            dist.scatter(tensor=partitioned_model.bn1.weight.data, scatter_list=[], src=0) #commented
            dist.scatter(tensor=partitioned_model.bn1.bias.data, scatter_list=[], src=0) #commented
            if args.layers == 3:
                dist.scatter(tensor=partitioned_model.fc3.weight.data, scatter_list=[], src=0)
                dist.scatter(tensor=partitioned_model.bn2.weight.data, scatter_list=[], src=0)
                dist.scatter(tensor=partitioned_model.bn2.bias.data, scatter_list=[], src=0)

        #added
        

def decide_new_partition_size(args, sync_time, coefs):
    
    if args.dynamic and not args.randomized_coefs:
        sync_time_tensor = torch.tensor([sync_time], dtype=torch.float32)
        gathered_times = [torch.zeros(1) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_times, sync_time_tensor)
        gathered_times = np.array([time.item() for time in gathered_times])
        new_coefs = np.sqrt((1/(np.sum(np.sqrt(1/(gathered_times/coefs**2)))**2))/(gathered_times/coefs**2))
    elif args.dynamic and args.randomized_coefs:
        coefs_tensor = torch.zeros(args.world_size)
        if args.rank == 0:
            alpha = [1, 1, 1, 1]
            random_coefs = np.random.dirichlet(alpha)
            coefs_tensor = torch.tensor(random_coefs, dtype=torch.float32)

        dist.broadcast(coefs_tensor, src=0)
        new_coefs = coefs_tensor.numpy()
    
    # new_coefs = np.sqrt((1/(np.sum(np.sqrt(1/(gathered_times/coefs**2)))**2))/(gathered_times/coefs**2))
    model_name = 'DNN_speech_' + str(args.layers) + '_layer_BN_' + str(args.epochs) + '_' + str(args.model_size) \
                 + '_cascaded_' + str(args.world_size) + '_' + str(args.repartition_iter)
    with open('./log/' + model_name + '_coefs.log', "a") as myfile:
        myfile.write(f"{new_coefs[args.rank]} ")
    partition_size = (int(args.model_size*new_coefs[args.rank]) if args.rank < args.world_size -1 else args.model_size - sum([int(new_coefs[i]*args.model_size) for i in range(args.world_size - 1)])) if args.dynamic else args.model_size//args.world_size
    # print(f"New Partition Size: {partition_size}")
    return new_coefs, partition_size


def push_model_to_parameter_server(args, partitioned_model, raw_model=None, coefs=None):
    print('push_model_to_parameter_server called!')
    if args.rank == 0:
        assert(raw_model is not None)

        if not np.all(coefs == coefs[0]):
            # print(raw_model.fc1_weight_partition[0].shape)
            # print(raw_model.fc1_weight_partition[1].shape)
            # print(raw_model.fc1_weight_partition[2].shape)
            # print(raw_model.fc1_weight_partition[3].shape)
            for i in range(1,args.world_size):
                dist.recv(tensor=raw_model.fc1_weight_partition[i], src=i)
                dist.recv(tensor=raw_model.fc2_weight_partition[i], src=i)
                dist.recv(tensor=raw_model.bn1_weight_partition[i], src=i)
                dist.recv(tensor=raw_model.bn1_bias_partition[i], src=i)
                if args.layers == 3:
                    dist.recv(tensor=raw_model.fc3_weight_partition[i], src=i)
                    dist.recv(tensor=raw_model.bn2_weight_partition[i], src=i)
                    dist.recv(tensor=raw_model.bn2_bias_partition[i], src=i)
        else:
            dist.gather(tensor=partitioned_model.fc1.weight.data, gather_list=raw_model.fc1_weight_partition, dst=0) #commented
            dist.gather(tensor=partitioned_model.fc2.weight.data, gather_list=raw_model.fc2_weight_partition, dst=0) #commented
            dist.gather(tensor=partitioned_model.bn1.weight.data, gather_list=raw_model.bn1_weight_partition, dst=0) #commented
            dist.gather(tensor=partitioned_model.bn1.bias.data, gather_list=raw_model.bn1_bias_partition, dst=0) #commented
            if args.layers == 3:
                dist.gather(tensor=partitioned_model.fc2.weight.data, gather_list=raw_model.fc2_weight_partition, dst=0)
                dist.gather(tensor=partitioned_model.bn2.weight.data, gather_list=raw_model.bn2_weight_partition, dst=0)
                dist.gather(tensor=partitioned_model.bn2.bias.data, gather_list=raw_model.bn2_bias_partition, dst=0)
    else:

        if not np.all(coefs == coefs[0]):
            # print(partitioned_model.fc1.weight.data.shape)
            dist.send(tensor=partitioned_model.fc1.weight.data, dst=0)
            dist.send(tensor=partitioned_model.fc2.weight.data, dst=0)
            dist.send(tensor=partitioned_model.bn1.weight.data, dst=0)
            dist.send(tensor=partitioned_model.bn1.bias.data, dst=0)
            if args.layers == 3:
                dist.send(tensor=partitioned_model.fc3.weight.data, dst=0)
                dist.send(tensor=partitioned_model.bn2.weight.data, dst=0)
                dist.send(tensor=partitioned_model.bn2.bias.data, dst=0)
        else:
            dist.gather(tensor=partitioned_model.fc1.weight.data, gather_list=[], dst=0) #comment
            dist.gather(tensor=partitioned_model.fc2.weight.data, gather_list=[], dst=0) #comment
            dist.gather(tensor=partitioned_model.bn1.weight.data, gather_list=[], dst=0) #comment
            dist.gather(tensor=partitioned_model.bn1.bias.data, gather_list=[], dst=0) #comment
            if args.layers == 3:
                dist.gather(tensor=partitioned_model.fc3.weight.data, gather_list=[], dst=0) #comment
                dist.gather(tensor=partitioned_model.bn2.weight.data, gather_list=[], dst=0) #comment
                dist.gather(tensor=partitioned_model.bn2.bias.data, gather_list=[], dst=0) #comment

        #added
        

def train(args, partitioned_model, raw_model, optimizer, train_loader, epoch, train_time_log, sync_time_log, coefs):
    start_time = time.time()
    partitioned_model.train()
    if args.rank == 0:
        raw_model.train()
    for i, batch in enumerate(train_loader):
        if i < len(train_loader) // args.world_size:
            if i % args.repartition_iter == 0:
                if args.rank == 0:
                    dispatch_model_to_workers(args, partitioned_model, raw_model, sync_time_log[-1] if len(sync_time_log) > 0 else 0, coefs=coefs)
                else:
                    dispatch_model_to_workers(args, partitioned_model, sync_time=sync_time_log[-1] if len(sync_time_log) > 0 else 0, coefs=coefs)
                if args.dynamic:
                    optimizer = torch.optim.SGD(partitioned_model.parameters(), lr=args.lr)
                    partitioned_model.train()
                sync_start_time = time.time()
            data, target = batch['wav'].float(), batch['label']
            optimizer.zero_grad()
            output = partitioned_model(data)
            loss = nn.functional.nll_loss(output, target)
            loss.backward()
            optimizer.step()  # This will just update the local data which reduces communication overhead.
            i += 1
            train_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            train_correct = train_pred.eq(target.view_as(train_pred)).sum().item()
            if i % args.log_interval == 0:
                print('Train Epoch {} iter {} <Loss: {:.6f}, Accuracy: {:.2f}%>'.format(
                    epoch, i, loss.item(), 100. * train_correct / target.shape[0]))
            if (i + 1) % args.repartition_iter == 0 or i == len(train_loader) // args.world_size:
                sync_end_time = time.time()
                sync_elapsed_time = sync_end_time - sync_start_time
                sync_time_log.append(sync_elapsed_time)
                if args.rank == 0:
                    push_model_to_parameter_server(args, partitioned_model, raw_model, coefs)
                    raw_model.flush()
                else:
                    push_model_to_parameter_server(args, partitioned_model, coefs=coefs)                
        else:
            break
    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Node {}: Train Epoch {} total time {:3.2f}s'.format(args.rank, epoch, elapsed_time))
    train_time_log[epoch-1] = elapsed_time


def test(args, raw_model, test_loader, epoch, test_loss_log, test_acc_log):
    # currently only do test on rank0 node.
    assert(args.rank == 0)
    raw_model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for _, batch in enumerate(test_loader):
            data, target = batch['wav'].float(), batch['label']
            output = raw_model(data)
            test_loss += nn.functional.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            test_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            test_correct += test_pred.eq(target.view_as(test_pred)).sum().item()
            test_total += target.shape[0]
        test_acc = float(test_correct) / float(test_total)
        test_loss /= float(test_total)
    print("Epoch {} Test Loss: {:.6f}; Test Accuracy: {:.2f}.\n".format(epoch, test_loss, test_acc))
    test_loss_log[epoch - 1] = test_loss
    test_acc_log[epoch - 1] = test_acc


def worker_process(args):
    coefs = np.array(args.init_partitions)
    assert(args.rank != 0)
    print(args.dist_backend, args.dist_url, args.rank, args.world_size)
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            rank=args.rank, world_size=args.world_size)
    device = torch.device('cpu')
    train_set = speech_dataset.train_dataset()
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True,
                              drop_last=True)
    model_name = 'DNN_speech_' + str(args.layers) +'_layer_BN_' + str(args.epochs) + '_' + str(args.model_size) \
                 + '_cascaded_' + str(args.world_size) + '_' + str(args.repartition_iter)
    
    # partition_size = args.model_size//args.world_size
    partition_size = int(args.model_size*coefs[args.rank]) if args.rank < args.world_size -1 else args.model_size - sum([int(coefs[i]*args.model_size) for i in range(args.world_size - 1)])
    print(f"Initial Partition Size: {partition_size}")
    if args.layers == 2:
        partitioned_model = DNNGoogleSpeechBatchNorm2Layer(partition_num=1,
                                                       model_size=partition_size).to(device)
    elif args.layers == 3:
        partitioned_model = DNNGoogleSpeechBatchNorm3Layer(partition_num=1,
                                                       model_size=partition_size).to(device)

    print(f"{args.layers}-Layer Model Initialized")    
    optimizer = torch.optim.SGD(partitioned_model.parameters(), lr=args.lr)
    epochs = args.epochs
    train_time_log = np.zeros(epochs)
    sync_time_log = []
    with open('./log/' + model_name + '_coefs.log', "a") as myfile:
        myfile.write(f"{coefs[args.rank]} ")
    for epoch in range(1, epochs + 1):
        print(epoch)
        train(args, partitioned_model, None, optimizer, train_loader, epoch, train_time_log, sync_time_log, coefs)
        np.savetxt('./log/' + model_name + '_train_time.log', train_time_log, fmt='%1.4f', newline=' ')
        np.savetxt('./log/' + model_name + '_sync_time.log', np.array(sync_time_log), fmt='%1.4f', newline=' ')


def parameter_server_process(args):
    coefs = np.array(args.init_partitions)
    print("Waiting for workers...")
    assert (args.rank == 0)
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            rank=args.rank, world_size=args.world_size)
    print("Initialized")
    device = torch.device('cpu')
    train_set = speech_dataset.train_dataset()
    test_set = speech_dataset.test_dataset()
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True,
                              drop_last=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True,
                             drop_last=False)
    model_name = 'DNN_speech_' + str(args.layers) + '_layer_BN_' + str(args.epochs) + '_' + str(args.model_size) \
                 + '_cascaded_' + str(args.world_size) + '_' + str(args.repartition_iter)
    print("we are going to train from scratch.")
    
    partition_size = int(args.model_size*coefs[args.rank])
    
    # print(f"Partition Size: {partition_size}")
    print(f"Initial Partition Size: {partition_size}")
    
    if args.layers == 2:
        raw_model = DNNGoogleSpeechBatchNorm2Layer(partition_num=args.world_size,
                                               model_size=args.model_size).to(device)

        partitioned_model = DNNGoogleSpeechBatchNorm2Layer(partition_num=1,
                                                       model_size=partition_size).to(device)
    elif args.layers == 3:
        raw_model = DNNGoogleSpeechBatchNorm3Layer(partition_num=args.world_size,
                                               model_size=args.model_size).to(device)
        
        partitioned_model = DNNGoogleSpeechBatchNorm3Layer(partition_num=1,
                                                       model_size=partition_size).to(device)
    
    print(f"{args.layers}-Layer Model Initialized")
    optimizer = torch.optim.SGD(partitioned_model.parameters(), lr=args.lr)
    epochs = args.epochs
    train_time_log = np.zeros(epochs)
    test_loss_log = np.zeros(epochs)
    test_acc_log = np.zeros(epochs)
    sync_time_log = []
    with open('./log/' + model_name + '_coefs.log', "a") as myfile:
        myfile.write(f"{coefs[args.rank]} ")
    for epoch in range(1, epochs + 1):
        print(epoch)
        train(args, partitioned_model, raw_model, optimizer, train_loader, epoch, train_time_log, sync_time_log, coefs)
        test(args, raw_model, test_loader, epoch, test_loss_log, test_acc_log)
        np.savetxt('./log/' + model_name + '_train_time.log', train_time_log, fmt='%1.4f', newline=' ')
        np.savetxt('./log/' + model_name + '_sync_time.log', np.array(sync_time_log), fmt='%1.4f', newline=' ')
        np.savetxt('./log/' + model_name + '_test_loss.log', test_loss_log, fmt='%1.4f', newline=' ')
        np.savetxt('./log/' + model_name + '_test_acc.log', test_acc_log, fmt='%1.4f', newline=' ')
        torch.save(raw_model, './trained_models/' + model_name + '.pth')


def list_of_floats(arg):
    return list(map(float, arg.split(',')))


def main(args):
    # parser = argparse.ArgumentParser(description='PyTorch 2-layer DNN on google speech dataset (subnet single PS)')
    # parser.add_argument('--dist-backend', type=str, default='gloo', metavar='S',
    #                     help='backend type for distributed PyTorch')
    # parser.add_argument('--dist-url', type=str, default='tcp://127.0.0.1:9000', metavar='S',
    #                     help='master ip for distributed PyTorch')
    # parser.add_argument('--rank', type=int, default=1, metavar='R',
    #                     help='rank for distributed PyTorch')
    # parser.add_argument('--world-size', type=int, default=2, metavar='D',
    #                     help='partition group (default: 2)')
    # parser.add_argument('--model-size', type=int, default=4096, metavar='N',
    #                     help='model size for intermediate layers (default: 4096)')
    # parser.add_argument('--batch-size', type=int, default=128, metavar='N',
    #                     help='input batch size for training (default: 100)')
    # parser.add_argument('--epochs', type=int, default=100, metavar='N',
    #                     help='number of epochs to train (default: 100)')
    # parser.add_argument('--repartition-iter', type=int, default=20, metavar='N',
    #                     help='keep model in local update mode for how many iteration (default: 5)')
    # parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
    #                     help='learning rate (default: 0.001 for BN)')
    # parser.add_argument('--seed', type=int, default=1, metavar='S',
    #                     help='random seed (default: 1)')
    # parser.add_argument('--log-interval', type=int, default=1, metavar='N',
    #                     help='how many batches to wait before logging training status')
    # parser.add_argument('--dynamic', type=bool, default=False, metavar='N',
    #                     help='if uses dynamic approach')
    # parser.add_argument('--randomized-coefs', type=bool, default=False, metavar='N',
    #                     help='if uses randomzied coefficients')
    # parser.add_argument('--layers', type=int, default=2, metavar='N',
    #                     help='number of layers')
    # parser.add_argument('--init-partitions', type=list_of_floats, default='0.25,0.25,0.25,0.25', metavar='N',
    #                     help='initial partition probabilities among workers')
    # args = parser.parse_args()
    print(f"Dynamic: {args.dynamic}")
    # assert(torch.cuda.is_available())
    torch.manual_seed(args.seed)

    if args.rank == 0:
        parameter_server_process(args)
    else:
        worker_process(args)


# if __name__ == '__main__':
#     main()