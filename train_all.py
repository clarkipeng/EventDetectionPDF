import sys, os

sys.path.append("../")

from train import train

from src.sleep import get_sleep_dataclass
from src.bowshock import get_bowshock_dataclass
from src.fraud import get_fraud_dataclass
from src.seizure import get_seizure_dataclass

from src.utils import get_loss, DataClass

import torch
import argparse
import warnings
from sklearn.model_selection import KFold


def get_args_parser():
    parser = argparse.ArgumentParser("get training parameters", add_help=False)

    parser.add_argument(
        "--datadir",
        default="./data",
        type=str,
    )
    # type
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["bowshock", "sleep", "fraud", "seizure"],
    )
    # data
    parser.add_argument("--downsample", default=10, type=int)
    parser.add_argument(
        "--agg_feats",
        default="stat",
        type=str,
        choices=["stat", "none", "all"],
        help="stat - aggregates mean, max, min, and std across downsampled series. none - pure downsampling. all - no signal is lost, all features are retained",
    )
    parser.add_argument("--use_cat", default=True, type=bool)
    parser.add_argument(
        "--sequence_length",
        default=None,
        type=int,
        help="length of training timeseries in timesteps",
    )
    # training
    parser.add_argument("--bs", default=10, type=int)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--folds", default=4, type=int)
    parser.add_argument("--normalize", default=True, type=bool)
    # helper
    parser.add_argument(
        "--device", default=("cuda" if torch.cuda.is_available() else "cpu"), type=str
    )
    parser.add_argument("--workers", default=4, type=int)

    return parser


if __name__ == "__main__":
    warnings.simplefilter("ignore", category=RuntimeWarning)
    parser = argparse.ArgumentParser(
        "training and evaluation script for all task subtypes",
        parents=[get_args_parser()],
    )
    args = parser.parse_args()

    if args.dataset == "sleep":
        dataclass = get_sleep_dataclass()
    elif args.dataset == "bowshock":
        dataclass = get_bowshock_dataclass()
    elif args.dataset == "fraud":
        dataclass = get_fraud_dataclass()
    elif args.dataset == "seizure":
        dataclass = get_seizure_dataclass()
    else:
        raise ValueError(f"{args.dataset} dataset not supported")

    sequence_length = args.sequence_length
    if not sequence_length:
        sequence_length = dataclass.default_sequence_length

    for model in ["rnn", "unet", "unet_t", "prectime"]:
        for objective in ["seg1", "seg2", "hard", "gau", "custom"]:

            train(
                dataclass=dataclass,
                data_dir=args.datadir,
                model_name=model,
                objective=objective,
                sequence_length=sequence_length,
                downsample=args.downsample,
                agg_feats=args.agg_feats,
                folds=args.folds,
                epochs=args.epochs,
                bs=args.bs,
                normalize=args.normalize,
                use_cat=args.use_cat,
                device=args.device,
                workers=args.workers,
            )
