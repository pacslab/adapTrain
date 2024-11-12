import argparse
from ist import main as ist_main
from data_parallel import main as ddp_main

def list_of_floats(arg):
    return list(map(float, arg.split(',')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch 2-layer DNN on google speech dataset (subnet single PS)')
    parser.add_argument('--method', type=str, default='ist', metavar='S',
                        help='which method to use')
    parser.add_argument('--dist-backend', type=str, default='gloo', metavar='S',
                        help='backend type for distributed PyTorch')
    parser.add_argument('--dist-url', type=str, default='tcp://127.0.0.1:9000', metavar='S',
                        help='master ip for distributed PyTorch')
    parser.add_argument('--rank', type=int, default=1, metavar='R',
                        help='rank for distributed PyTorch')
    parser.add_argument('--world-size', type=int, default=2, metavar='D',
                        help='partition group (default: 2)')
    parser.add_argument('--model-size', type=int, default=4096, metavar='N',
                        help='model size for intermediate layers (default: 4096)')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--repartition-iter', type=int, default=20, metavar='N',
                        help='keep model in local update mode for how many iteration (default: 5)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.001 for BN)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--dynamic', type=bool, default=False, metavar='N',
                        help='if uses dynamic approach')
    parser.add_argument('--randomized-coefs', type=bool, default=False, metavar='N',
                        help='if uses randomzied coefficients')
    parser.add_argument('--layers', type=int, default=2, metavar='N',
                        help='number of layers')
    parser.add_argument('--init-partitions', type=list_of_floats, default='0.25,0.25,0.25,0.25', metavar='N',
                        help='initial partition probabilities among workers')
    args = parser.parse_args()

    if args.method == 'ist':
        print("Method: IST")
        ist_main(args)
    elif args.method == 'ddp':
        print("Method: DDP")
        ddp_main(args)
