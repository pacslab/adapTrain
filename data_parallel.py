import argparse
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.distributed import DistributedSampler
import google_speech_data_loader as speech_dataset
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import time



class GSR2LayerModel(nn.Module):
    def __init__(self, sample_size=4096, model_size=4096, label_num=35):
        super(GSR2LayerModel, self).__init__()
        self.fc1 = nn.Linear(sample_size, model_size, False)
        self.bn1 = nn.BatchNorm1d(model_size, momentum=1.0, track_running_stats=False)
        self.fc2 = nn.Linear(model_size, label_num, False)
        self.bn2 = nn.BatchNorm1d(label_num, momentum=1.0, affine=False, track_running_stats=False)

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
    

class GSR3LayerModel(nn.Module):
    def __init__(self, sample_size=4096, model_size=4096, label_num=35):
        super(GSR3LayerModel, self).__init__()
        self.fc1 = nn.Linear(sample_size, model_size, False)
        self.bn1 = nn.BatchNorm1d(model_size, momentum=1.0, affine=True, track_running_stats=False)
        self.fc2 = nn.Linear(model_size, model_size, False)
        self.bn2 = nn.BatchNorm1d(model_size, momentum=1.0, affine=True, track_running_stats=False)
        self.fc3 = nn.Linear(model_size, label_num, False)
        self.bn3 = nn.BatchNorm1d(label_num, momentum=1.0, affine=False, track_running_stats=False)

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



def run(rank, args):
    torch.manual_seed(args.seed)

    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            rank=args.rank, world_size=args.world_size)
    
    device = torch.device('cpu')
    
    train_set = speech_dataset.train_dataset()
    test_set = speech_dataset.test_dataset() 
    
    sampler = DistributedSampler(train_set, num_replicas=args.world_size, rank=args.rank, shuffle=True, drop_last=False)
    
    dataloader = DataLoader(train_set, batch_size=args.batch_size, pin_memory=True, num_workers=0,
                             drop_last=False, shuffle=False, sampler=sampler)
    
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True,
                             drop_last=False)
    

    model_name = 'DNN_speech_' + str(args.layers) +'_layer_BN_' + str(args.epochs) + '_' + str(args.model_size) \
                 + '_cascaded_' + str(args.world_size)
    
    if args.layers == 2:
        model = GSR2LayerModel().to(device= device)

    elif args.layers == 3:
        model = GSR3LayerModel().to(device=device)
    
    ddp_model = DDP(model)

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), args.lr)
    epochs = args.epochs
    train_time_log = np.zeros(epochs)
    test_loss_log = np.zeros(epochs)
    test_acc_log = np.zeros(epochs)

    for epoch in range(1, epochs + 1):
        dataloader.sampler.set_epoch(epoch)
        start_time = time.time()
        model.train()
        for i, batch in enumerate(dataloader):
            data, target = batch['wav'].float(), batch['label']
            optimizer.zero_grad()
            output = model(data)
            loss = nn.functional.nll_loss(output, target)
            loss.backward()
            optimizer.step()  # This will just update the local data which reduces communication overhead.
            train_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            train_correct = train_pred.eq(target.view_as(train_pred)).sum().item()
            print('Train Epoch {} iter {} <Loss: {:.6f}, Accuracy: {:.2f}%>'.format(
                    epoch, i, loss.item(), 100. * train_correct / target.shape[0]))
        
        dist.barrier()
        end_time = time.time()
        elapsed_time = end_time - start_time
        print('Node {}: Train Epoch {} total time {:3.2f}s'.format(args.rank, epoch, elapsed_time))
        train_time_log[epoch-1] = elapsed_time
        np.savetxt('./log/' + model_name + '_train_time.log', train_time_log, fmt='%1.4f', newline=' ')
        # Test On Rank 0
        if args.rank == 0:
            model.eval()
            test_loss = 0.0
            test_correct = 0
            test_total = 0
            with torch.no_grad():
                for _, batch in enumerate(test_loader):
                    data, target = batch['wav'].float(), batch['label']
                    output = model(data)
                    test_loss += nn.functional.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                    test_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
                    test_correct += test_pred.eq(target.view_as(test_pred)).sum().item()
                    test_total += target.shape[0]
                test_acc = float(test_correct) / float(test_total)
                test_loss /= float(test_total)
            
            print("Epoch {} Test Loss: {:.6f}; Test Accuracy: {:.2f}.\n".format(epoch, test_loss, test_acc))
            test_loss_log[epoch - 1] = test_loss
            test_acc_log[epoch - 1] = test_acc
            np.savetxt('./log/' + model_name + '_test_loss.log', test_loss_log, fmt='%1.4f', newline=' ')
            np.savetxt('./log/' + model_name + '_test_acc.log', test_acc_log, fmt='%1.4f', newline=' ')
    
    # dist.barrier()
    dist.destroy_process_group()



def main(args):
    print("DDP Started")
    run(args.rank, args)
    # torch.multiprocessing.spawn(run, args=(args,), nprocs=args.world_size, join=True)


    
    


# if __name__ == '__main__':
#     main()