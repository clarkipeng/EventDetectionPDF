import sys, os
sys.path.append('../')

from train import train
from eval import get_optimal_cutoff
from src.utils import get_loss
from src.load_dataset import SleepDataset

import torch
import argparse
import warnings
from sklearn.model_selection import KFold


def get_args_parser():
    parser = argparse.ArgumentParser('get training parameters', add_help=False)
    
    parser.add_argument('--datadir', default='./data', type=str, )
    #type
    parser.add_argument('--downsample', default=10, type=int)
    parser.add_argument('--agg_feats', default=True, type=bool)
    parser.add_argument('--use_time_cat', default=True, type=bool)
    parser.add_argument('--sequence_length', default=7*(24*60*12), type=int,
                        help='length of training timeseries where each step = 5s'
                        )
    #training
    parser.add_argument('--bs', default=10, type=int)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--folds', default=4, type=int)
    parser.add_argument('--normalize', default=True, type=bool)
    #helper
    parser.add_argument('--device', default=('cuda' if torch.cuda.is_available() else 'cpu'), type=str)
    parser.add_argument('--workers', default=4, type=int)


    return parser
if __name__ == "__main__":
    warnings.simplefilter("ignore", category=RuntimeWarning)
    parser = argparse.ArgumentParser('training and evaluation script for all task subtypes', parents=[get_args_parser()])
    args = parser.parse_args()
    for model in ['rnn','unet','unet_t','prectime']:
        for objective in ['seg1','seg2','hard','gau','custom']:
            
            train(
                data_dir = args.datadir,
                model_name = model,
                objective = objective,
                sequence_length = args.sequence_length,
                downsample = args.downsample,
                agg_feats = args.agg_feats,
                folds = args.folds,
                epochs = args.epochs,
                bs = args.bs,
                normalize = args.normalize,
                use_time_cat = args.use_time_cat,
                device  = args.device,
                workers = args.workers,
            )

