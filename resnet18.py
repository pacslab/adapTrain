import os

import torch.nn.functional as functional
import torch.distributed as dist
import numpy as np
import argparse
import torch
import time
import torchvision.transforms as transforms
from torchvision.datasets.cifar import CIFAR10
from random import shuffle, choice, seed, sample

from data import (
    get_cifar10_loaders,
    get_cifar100_loaders,
    get_tin_data,
    get_svhn_loaders,
)
from utils import get_demon_momentum, aggregate_resnet_optimizer_statistics


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return torch.nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                           padding=dilation, groups=groups, bias=False, dilation=dilation)

class PreActBlock(torch.nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.active_flag = True
        self.scale_constant = 1.0
        self.bn1 = torch.nn.BatchNorm2d(in_planes, affine=True, track_running_stats=False)
        self.conv1 = torch.nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(planes, affine=True, track_running_stats=False)
        self.conv2 = torch.nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.downsample = None
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = torch.nn.Sequential(
                torch.nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        if self.active_flag:  # Instead of zero out weights, this can also avoid computation.
            out = functional.relu(self.bn1(x))
            shortcut = self.downsample(out) if self.downsample is not None else x
            out = self.conv1(out)
            out = self.conv2(functional.relu(self.bn2(out))) * self.scale_constant
            out += shortcut
        else:
            out = x
        return out


class PreActResNet(torch.nn.Module):
    # taken from https://github.com/kuangliu/pytorch-cifar

    def __init__(self, block, num_blocks, out_size=512, num_classes=10):
        super(PreActResNet, self).__init__()
        self.in_planes = 64
        self.num_blocks = num_blocks
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.fc = torch.nn.Linear(out_size * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = functional.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


# This is our focus RESNET MODEL
def PreActResNet18(blocks=[2, 2, 2, 2], out_size=512, num_classes=10):
    return PreActResNet(PreActBlock, blocks, out_size=out_size, num_classes=num_classes)

'''
def PreActResNet101(blocks=[3, 4, 23, 3], out_size=512, num_classes=10):
    return PreActResNet(PreActBlock, blocks, out_size=out_size, num_classes=num_classes)


def PreActResNet152(blocks=[3, 4, 36, 3], out_size=512, num_classes=10):
    return PreActResNet(PreActBlock, blocks, out_size=out_size, num_classes=num_classes)


def PreActResNet200(blocks=[3, 4, 50, 3], out_size=512, num_classes=10):
    return PreActResNet(PreActBlock, blocks, out_size=out_size, num_classes=num_classes)

'''

def sample_block_indices_with_overlap(num_sites, num_blocks, min_blocks_per_site, ideal_blocks_per_site_list):
    ideal_blocks_per_site_list = [(i if i > min_blocks_per_site else min_blocks_per_site) for i in ideal_blocks_per_site_list]
    blocks_per_site_list = [i for i in ideal_blocks_per_site_list]
    # blocks = list(range(1, num_blocks + 1))
    while(sum(blocks_per_site_list) < num_blocks):
        #increase blocks per site for one site with minimum deviation from the initial state
        effect = [ (blocks_per_site_list[i] + 1 - ideal_blocks_per_site_list[i])/ideal_blocks_per_site_list[i] for i in range(num_sites)]
        idx = effect.index(min(effect))
        blocks_per_site_list[idx] += 1
        
    blocks = list(range(1, num_blocks + 1))
    shuffle(blocks)
    initial_site_indices = [ blocks[ (0 if i==0 else sum(blocks_per_site_list[:i])) : len(blocks) if i+1 == num_sites else sum(blocks_per_site_list[:i+1]) ] for i in range(num_sites)]

    if sum(blocks_per_site_list) == num_blocks:
        return initial_site_indices, blocks_per_site_list

    for i in range(num_sites):
        if blocks_per_site_list[i] == len(initial_site_indices[i]):
            continue

        additional_blocks_num = blocks_per_site_list[i] - len(initial_site_indices[i])
        sampling_pool = set(range(1, num_blocks + 1)) - set(initial_site_indices[i])
        blocks_to_add = sample(list(sampling_pool), k=additional_blocks_num)
        initial_site_indices[i] = list(set(initial_site_indices[i]).union(set(blocks_to_add)))
    
    return initial_site_indices, blocks_per_site_list

test_rank = 0
test_total_time = 0


def broadcast_module(module: torch.nn.Module, rank_list=None, source=0):
    if rank_list is None:
        group = dist.group.WORLD
    else:
        group = dist.new_group(rank_list)

    for para in module.parameters():
        dist.broadcast(para.data, src=source, group=group, async_op=False)

    if rank_list is not None:
        dist.destroy_process_group(group)


def reduce_module(specs, args, module: torch.nn.Module, rank_list=None):
    if rank_list is None:
        raise 'error'
    else:
        group = dist.new_group(rank_list)
    for para in module.parameters():
        dist.reduce(para.data, dst=min(rank_list), op=dist.ReduceOp.SUM, group=group)
        if args.rank == min(rank_list):  # compute average
            if rank_list is None:
                para.data = para.data.div_(specs['world_size'])
            else:
                para.data = para.data.div_(len(rank_list))
    if rank_list is not None:
        dist.destroy_process_group(group)


def all_reduce_module(specs, args, module: torch.nn.Module, rank_list=None):
    group = dist.group.WORLD
    for para in module.parameters():
        dist.all_reduce(para.data, op=dist.ReduceOp.SUM, group=group)
        if rank_list is None:
            para.data = para.data.div_(specs['world_size'])
        else:
            para.data = para.data.div_(len(rank_list))
    if rank_list is not None:
        dist.destroy_process_group(group)


def decide_new_blocks_per_worker(args, specs, sync_time):
    if args.dynamic and not args.random:
        sync_time_tensor = torch.tensor([sync_time], dtype=torch.float32).cuda()
        gathered_times = [torch.zeros(1, dtype=torch.float32).cuda() for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_times, sync_time_tensor)
        gathered_times = np.array([time.item() for time in gathered_times])
        old_blocks_list = np.array(specs['init_blocks_per_site'])
        print(f"Gathered Times: {gathered_times}")
        print(f"Old Blocks: {old_blocks_list}")
        blocks_per_worker_list = np.ceil(specs['layer_sizes'][2]*old_blocks_list / (gathered_times * np.sum(old_blocks_list / gathered_times))).astype(int)
        print(f"Blocks per worker: {blocks_per_worker_list}")
        new_blocks_per_worker,new_blocks_sizes = sample_block_indices_with_overlap(num_sites=specs['world_size'],
                                                              num_blocks=specs['layer_sizes'][2],
                                                              min_blocks_per_site=specs['min_blocks_per_site'],
                                                              ideal_blocks_per_site_list=blocks_per_worker_list)

    
    elif args.dynamic and args.random:
        blocks_tensor = torch.zeros(args.world_size).cuda()
        if args.rank == 0:
            blocks_per_worker_list = np.array(random_integers_sum_to_n(specs['layer_sizes'][2], specs['world_size']))
            blocks_tensor = torch.tensor(blocks_per_worker_list, dtype=torch.int).cuda()
            
        dist.broadcast(blocks_tensor, src=0)
        new_blocks = blocks_tensor.numpy()
        new_blocks_per_worker,new_blocks_sizes = sample_block_indices_with_overlap(num_sites=specs['world_size'],
                                                              num_blocks=specs['layer_sizes'][2],
                                                              min_blocks_per_site=specs['min_blocks_per_site'],
                                                              ideal_blocks_per_site_list=new_blocks)

    return new_blocks_per_worker, new_blocks_sizes



class ISTResNetModel():
    def __init__(self, model: PreActResNet, num_sites=4, min_blocks_per_site=0):
        self.base_model = model
        self.min_blocks_per_site = min_blocks_per_site
        self.num_sites = num_sites
        self.site_indices = None
        if min_blocks_per_site == 0:
            self.scale_constant = 1.0 / num_sites
        else:
            # dropout prob becomes total blocks per site / total blocks in layer3
            self.scale_constant = max(1.0 / num_sites, min_blocks_per_site / 22)
        self.layer_server_list = []

    def prepare_eval(self):
        for i in range(1, self.base_model.num_blocks[2]):
            self.base_model.layer3[i].active_flag = True
            self.base_model.layer3[i].scale_constant = self.scale_constant

    def prepare_train(self, args):
        for i in range(1, self.base_model.num_blocks[2]):
            self.base_model.layer3[i].active_flag = i in self.site_indices[args.rank]
            self.base_model.layer3[i].scale_constant = 1.0

    def dispatch_model(self, specs, args, sync_time):
        if args.dynamic:
            self.site_indices, specs['init_blocks_per_site'] = decide_new_blocks_per_worker(args, specs, sync_time)
            print(f"New Blocks per Site: {specs['init_blocks_per_site']}")
        else:    
            self.site_indices,_ = sample_block_indices_with_overlap(num_sites=self.num_sites,
                                                              num_blocks=self.base_model.num_blocks[2],
                                                              min_blocks_per_site=self.min_blocks_per_site,
                                                              ideal_blocks_per_site_list=specs['init_blocks_per_site'])
            
        
        with open('./log/' + args.model_name + '_blocks.log', "a") as myfile:
            myfile.write(f"{specs['init_blocks_per_site'][args.rank]} ")

        print(self.site_indices)
        for i in range(1, self.base_model.num_blocks[2]):
            current_group = []
            for site_i in range(self.num_sites):
                if i in self.site_indices[site_i]:
                    current_group.append(site_i)
            if not (self.layer_server_list[i] in current_group):
                current_group.append(self.layer_server_list[i])
            broadcast_module(self.base_model.layer3[i], rank_list=current_group, source=self.layer_server_list[i])
            self.base_model.layer3[i].active_flag = i in self.site_indices[args.rank]

    def sync_model(self, specs, args):
        # aggregate conv1
        all_reduce_module(specs, args, self.base_model.conv1)
        # aggregate layer 1 & 2 & 4
        all_reduce_module(specs, args, self.base_model.layer1)
        all_reduce_module(specs, args, self.base_model.layer2)
        all_reduce_module(specs, args, self.base_model.layer4)
        # aggregate FC layer
        all_reduce_module(specs, args, self.base_model.fc)
        # apply IST aggregation here
        all_reduce_module(specs, args, self.base_model.layer3[0])
        self.layer_server_list = [-1]
        for i in range(1, self.base_model.num_blocks[2]):

            current_group = []
            for site_i in range(self.num_sites):
                if i in self.site_indices[site_i]:
                    current_group.append(site_i)
            self.layer_server_list.append(min(current_group))
            reduce_module(specs, args, self.base_model.layer3[i], rank_list=current_group)

    def ini_sync_dispatch_model(self, specs, args):
        # broadcast conv1
        broadcast_module(self.base_model.conv1, source=0)

        # # broadcast layer 1 & 2 & 4
        broadcast_module(self.base_model.layer1, source=0)

        broadcast_module(self.base_model.layer2, source=0)

        broadcast_module(self.base_model.layer4, source=0)

        # # broadcast FC layer
        broadcast_module(self.base_model.fc, source=0)

        broadcast_module(self.base_model.layer3[0], source=0)

        self.site_indices, blocks_sizes = sample_block_indices_with_overlap(num_sites=self.num_sites,
                                                              num_blocks=self.base_model.num_blocks[2],
                                                              min_blocks_per_site=self.min_blocks_per_site,
                                                              ideal_blocks_per_site_list=specs['init_blocks_per_site'])
        
        with open('./log/' + args.model_name + '_blocks.log', "a") as myfile:
            myfile.write(f"{blocks_sizes[args.rank]} ")

        print(self.site_indices)
        # # apply IST here
        for i in range(1, self.base_model.num_blocks[2]):
            broadcast_module(self.base_model.layer3[i], source=0)
            self.base_model.layer3[i].active_flag = i in self.site_indices[args.rank]

    def prepare_whole_model(self, specs, args):
        for i in range(1, self.base_model.num_blocks[2]):

            current_group = []
            for site_i in range(self.num_sites):
                if i in self.site_indices[site_i]:
                    current_group.append(site_i)

            if not (current_group[0] == 0):
                broadcast_module(self.base_model.layer3[i], rank_list=[0, min(current_group)],
                                 source=min(current_group))
    

def random_integers_sum_to_n(total, count):
    # Generate random points and sort them
    points = sorted(sample(range(1, total), count - 1))
    
    # Create intervals between points
    numbers = [points[0]] + [points[i] - points[i - 1] for i in range(1, count - 1)] + [total - points[-1]]
    
    return numbers


def train(specs, args, start_time, model_name, ist_model: ISTResNetModel, optimizer, device, train_loader, test_loader,
          epoch, num_sync, num_iter, train_time_log, test_loss_log, test_acc_log, sync_time_log):
    # employ a step schedule for the sub nets
    lr = specs.get('lr', 1e-2)
    if epoch > int(specs['epochs']*0.5):
        lr /= 10
    if epoch > int(specs['epochs']*0.75):
        lr /= 10
    if optimizer is not None:
        for pg in optimizer.param_groups:
            pg['lr'] = lr
    print(f'Learning Rate: {lr}')

    # training loop
    for i, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        if num_iter % specs['repartition_iter'] == 0 or i == 0:
            if num_iter>0:
                # specs['init_blocks_per_site'] = random_integers_sum_to_n(specs['layer_sizes'][2], specs['world_size'])
                print('running model dispatch')
                print(f"Sync Time Log Length: {len(sync_time_log)}")
                if len(sync_time_log) > 0:
                    print(f"Sync Time: {sync_time_log[num_sync - 1]}")

                ist_model.dispatch_model(specs, args, sync_time_log[num_sync - 1])
                print('model dispatch finished')
                sync_start_time = time.time()
            optimizer = torch.optim.SGD(
                    ist_model.base_model.parameters(), lr=lr,
                    momentum=specs.get('momentum', 0.9), weight_decay=specs.get('wd', 5e-4))

        optimizer.zero_grad()
        output = ist_model.base_model(data)
        loss = functional.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        train_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
        train_correct = train_pred.eq(target.view_as(train_pred)).sum().item()
        print('Train Epoch {} iter {} <Loss: {:.6f}, Accuracy: {:.2f}%>'.format(
                    epoch, num_iter, loss.item(), 100. * train_correct / target.shape[0]))
        if (
                ((num_iter + 1) % specs['repartition_iter'] == 0) or
                (i == len(train_loader) - 1 and epoch == specs['epochs'])):
            if num_sync == 0:
                sync_start_time = start_time

            # if num_sync > 0:
            delay = specs['time_delay']*(time.time() - sync_start_time)
            time.sleep(delay)
            sync_end_time = time.time()
            sync_elapsed_time = sync_end_time - sync_start_time
            sync_time_log[num_sync] = sync_elapsed_time
            np.savetxt('./log/' + model_name + '_sync_time.log', sync_time_log, fmt='%1.4f', newline=' ')
            
            print('running model sync')
            ist_model.sync_model(specs, args)
            print('model sync finished')
            num_sync = num_sync + 1
            end_time = time.time()
            elapsed_time = end_time - start_time
            print('Node {}: Train Num sync {} total time {:3.2f}s'.format(args.rank, num_sync, elapsed_time))
            if args.rank == 0:
                if num_sync == 1:
                    train_time_log[num_sync - 1] = elapsed_time
                else:
                    train_time_log[num_sync - 1] = train_time_log[num_sync - 2] + elapsed_time
                print('total time {:3.2f}s'.format(train_time_log[num_sync - 1]))
                print('total broadcast time',test_total_time)
            
            print(f'preparing the whole model for testing')
            ist_model.prepare_whole_model(specs,args)
            print(f'model prepared, now testing')
            test(specs,args, ist_model, device, test_loader, epoch, num_sync, test_loss_log, test_acc_log)
            print('done testing')
            start_time = time.time()
        num_iter = num_iter + 1

    # save model checkpoint at the end of each epoch
    if args.rank == 0:
        np.savetxt('./log/' + model_name + '_train_time.log', train_time_log, fmt='%1.4f', newline=' ')
        np.savetxt('./log/' + model_name + '_test_loss.log', test_loss_log, fmt='%1.4f', newline=' ')
        np.savetxt('./log/' + model_name + '_test_acc.log', test_acc_log, fmt='%1.4f', newline=' ')
        checkpoint = {
                'model': ist_model.base_model.state_dict(),
                'epoch': epoch,
                'num_sync': num_sync,
                'num_iter': num_iter,
        }
        torch.save(checkpoint, './log/' + model_name + '_model.pth')
    return num_sync, num_iter, start_time, optimizer



# delay = specs['time_delay']*(time.time() - sync_start_time)
# time.sleep(delay)

def test(specs, args, ist_model: ISTResNetModel, device, test_loader, epoch, num_sync, test_loss_log, test_acc_log):
    # Do validation only on prime node in cluster.
    if args.rank == 0:
        ist_model.prepare_eval()
        ist_model.base_model.eval()
        agg_val_loss = 0.
        num_correct = 0.
        total_ex = 0.
        criterion = torch.nn.CrossEntropyLoss()
        for model_in, labels in test_loader:
            model_in = model_in.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                model_output = ist_model.base_model(model_in)
                val_loss = criterion(model_output, labels)
            agg_val_loss += val_loss.item()
            _, preds = model_output.max(1)
            total_ex += labels.size(0)
            num_correct += preds.eq(labels).sum().item()
        agg_val_loss /= len(test_loader)
        val_acc = num_correct / total_ex
        print("Epoch {} Number of Sync {} Local Test Loss: {:.6f}; Test Accuracy: {:.4f}.\n".format(epoch, num_sync,
                                                                                                    agg_val_loss,
                                                                                                    val_acc))
        test_loss_log[num_sync - 1] = agg_val_loss
        test_acc_log[num_sync - 1] = val_acc
        ist_model.prepare_train(args)  # reset all scale constants
        ist_model.base_model.train()
        return val_acc


def list_of_ints(arg):
    return list(map(int, arg.split(',')))


def list_of_floats(arg):
    return list(map(float, arg.split(',')))


def main():
    specs = {
        'test_type': 'ist_resnet',  # should be either ist or baseline
        'model_type': 'preact_resnet',  # use to specify type of resnet to use in baseline
        'use_valid_set': False,
        'model_version': 'v1',  # only used for the mobilenet tests
        # 'dataset': 'cifar10', #DATASET SPECIFICATION
        # 'repartition_iter': 50,  # number of iterations to perform before re-sampling subnets
        'epochs': 40,
        # 'world_size': 4,  # ***** number of subnets to use during training
        # 'layer_sizes': [3, 4, 23, 3],  # used for resnet baseline, number of blocks in each section
        'expansion': 1.,
        # 'lr': .01,
        'momentum': 0.9,
        'wd': 5e-4,
        'log_interval': 5,
        'min_blocks_per_site': 0,  # used for the resnet ist, allow overlapping block partitions to occur
    }

    parser = argparse.ArgumentParser(description='PyTorch ResNet (IST distributed)')
    parser.add_argument('--dataset', type=str, default='cifar10')
    # The following dist-backed default should be changed to gloo for CPU usage
    parser.add_argument('--dist-backend', type=str, default='nccl', metavar='S',
                        help='backend type for distributed PyTorch')
    parser.add_argument('--dist-url', type=str, default='tcp://127.0.0.1:9000', metavar='S',
                        help='master ip for distributed PyTorch')
    parser.add_argument('--rank', type=int, default=0, metavar='R',
                        help='rank for distributed PyTorch')
    parser.add_argument('--world-size', type=int, default=2, metavar='D',
                        help='partition group (default: 2)')
    parser.add_argument('--epochs', type=int, default=40, metavar='D',
                        help='epochs')
    parser.add_argument('--layer-sizes', type=list_of_ints, metavar='S',
                        help='layer sizes')
    parser.add_argument('--repartition_iter', type=int, default=50, metavar='N',
                        help='keep model in local update mode for how many iteration (default: 5)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 1.0 for BN)')
    parser.add_argument('--pytorch-seed', type=int, default=-1, metavar='S',
                        help='random seed (default: -1)')
    # The following default should be changed to False for CPU
    parser.add_argument('--use-cuda', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='if this is set to True, will use cuda to train')
    parser.add_argument('--cuda-id', type=int, default=0, metavar='N',
                        help='cuda index, if the instance has multiple GPUs.')
    parser.add_argument('--model_name', type=str, default='cifar10_local_iter')
    parser.add_argument('--gpu-limit', type=float, help="GPU Usage Limit for each worker")
    parser.add_argument('--time-delay', type=float, default=0.0, help="Time Delay (ms) for each worker")
    parser.add_argument('--dynamic', type=bool, default=False, metavar='N', help='if uses dynamic approach')
    parser.add_argument('--random', type=bool, default=False, metavar='N', help='if uses random approach')
    parser.add_argument('--init-blocks-per-site', type=list_of_ints, default='6,6,6,5', metavar='N', help='initial partitioning among workers')


    args = parser.parse_args()

    specs['repartition_iter'] = args.repartition_iter
    specs['lr'] = args.lr
    specs['world_size'] = args.world_size
    specs['dataset'] = args.dataset
    specs['layer_sizes'] = args.layer_sizes
    specs['gpu_limit'] = args.gpu_limit
    specs['time_delay'] = args.time_delay
    specs['epochs'] = args.epochs
    specs['init_blocks_per_site'] = args.init_blocks_per_site

    if args.pytorch_seed == -1:
        torch.manual_seed(args.rank)
        seed(0)
    else:
        torch.manual_seed(args.rank * args.pytorch_seed)
        seed(args.pytorch_seed)
    # seed(0)  # This makes sure, node use the same random key so that they does not need to sync partition info.
    if args.use_cuda:
        print(args.cuda_id, torch.cuda.device_count())
        assert args.cuda_id < torch.cuda.device_count()
        device = torch.device('cuda', args.cuda_id)
        print(f"Worker {args.rank}: Limit = {specs['gpu_limit']}")
        torch.cuda.set_per_process_memory_fraction(specs['gpu_limit'], device=torch.device('cuda:0'))
        # torch.cuda.set_per_process_memory_fraction(0.5, device=torch.device('cuda:0'))
    else:
        device = torch.device('cpu')
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            rank=args.rank, world_size=specs['world_size'])
    global test_rank
    test_rank = args.rank
    if specs['dataset'] == 'cifar10':
        out_size = 512
        for_cifar = True
        num_classes = 10
        input_size = 32
        trn_dl, test_dl = get_cifar10_loaders(specs.get('use_valid_set', False))
        criterion = torch.nn.CrossEntropyLoss()
    elif specs['dataset'] == 'cifar100':
        out_size = 512
        for_cifar = True
        num_classes = 100
        input_size = 32
        trn_dl, test_dl = get_cifar100_loaders(specs.get('use_valid_set', False))
        criterion = torch.nn.CrossEntropyLoss()
    elif specs['dataset'] == 'svhn':
        for_cifar = True  # size of data is the same
        num_classes = 10
        input_size = 32
        out_size = 512
        trn_dl, test_dl = get_svhn_loaders()
        criterion = torch.nn.CrossEntropyLoss()
    else:
        raise NotImplementedError(f'{specs["dataset"]} dataset not supported')

    # check if model has been checkpointed already
    model_name = args.model_name
    if os.path.exists('./log/' + model_name + '_model.pth'):
        print('Loading model from a checkpoint!')
        ist_model = ISTResNetModel(
            model=PreActResNet18(out_size=out_size, num_classes=num_classes).to(device),
            num_sites=specs['world_size'], min_blocks_per_site=specs['min_blocks_per_site'])
        checkpoint = torch.load('./log/' + model_name + '_model.pth')
        ist_model.base_model.load_state_dict(checkpoint['model'])
        start_epoch = checkpoint['epoch']
        num_sync = checkpoint['num_sync']
        num_iter = checkpoint['num_iter']
        train_time_log = np.loadtxt('./log/' + model_name + '_train_time.log') if args.rank == 0 else None
        test_loss_log = np.loadtxt('./log/' + model_name + '_test_loss.log') if args.rank == 0 else None
        test_acc_log = np.loadtxt('./log/' + model_name + '_test_acc.log') if args.rank == 0 else None
        print(f'Starting at epoch {start_epoch}')
    else:
        print('Training model from scratch!')
        ist_model = ISTResNetModel(
            model=PreActResNet18(blocks=specs['layer_sizes'], out_size=out_size, num_classes=num_classes).to(device),
            num_sites=specs['world_size'], min_blocks_per_site=specs['min_blocks_per_site'])
        train_time_log = np.zeros(1000) if args.rank == 0 else None
        test_loss_log = np.zeros(1000) if args.rank == 0 else None
        test_acc_log = np.zeros(1000) if args.rank == 0 else None
        sync_time_log = np.zeros(1000)
        start_epoch = 0
        num_sync = 0
        num_iter = 0

    print('running initial sync')
    ist_model.ini_sync_dispatch_model(specs, args)
    print('initial sync finished')
    epochs = specs['epochs']
    optimizer = None
    start_time = time.time()
    for epoch in range(start_epoch + 1, epochs + 1):
        num_sync, num_iter, start_time, optimizer = train(
            specs, args, start_time, model_name, ist_model, optimizer, device,
            trn_dl, test_dl, epoch, num_sync, num_iter, train_time_log, test_loss_log,
            test_acc_log, sync_time_log)


if __name__ == '__main__':
    main()