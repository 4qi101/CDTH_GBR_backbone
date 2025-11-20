import argparse

parser = argparse.ArgumentParser(description='GBR Backbone - Cross-modal Retrieval with Granular Ball')
# 基本超参数
parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="GPU device id")
parser.add_argument('--batch_size', type=int, default=256, help="batch size")
parser.add_argument('--num_workers', type=int, default=0, help="number of workers for DataLoader")
parser.add_argument('--code_len', type=int, default=64, help="hash code length")
parser.add_argument('--seed', type=int, default=1234, help="random seed")

# 数据集相关
parser.add_argument('--data_path', type=str, default='data/Dataset/', help="dataset root path")
parser.add_argument('--dset', type=str, default='coco', choices=['coco', 'nuswide', 'flickr25k'], help="dataset name")

# 训练相关参数
parser.add_argument('--epochs', type=int, default=20, help="training epochs")
parser.add_argument('--lr', type=float, default=0.0001, help="learning rate")







