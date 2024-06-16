import sys, os

sys.path.append("../")

from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import gc
import warnings
import random
import json
import argparse

import numpy as np
import pandas as pd

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from sklearn.model_selection import KFold

from src.sleep import get_sleep_dataclass
from src.bowshock import get_bowshock_dataclass
from src.fraud import get_fraud_dataclass
from src.seizure import get_seizure_dataclass

from src.utils import get_loss, DataClass

from models.load_model import get_model

from eval import evaluate, get_optimal_cutoff

from timm.scheduler import CosineLRScheduler


def plot_history(history, model_path=".", show=True):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure()
    plt.plot(epochs, history["train_loss"], label="Training Loss")
    plt.plot(epochs, history["valid_loss"], label="Validation Loss")
    plt.title("Loss evolution")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(model_path, "loss_evo.png"))
    if show:
        plt.show()
    plt.close()

    plt.figure()
    plt.plot(epochs, history["valid_mAP"])
    plt.title("Validation mAP evolution")
    plt.xlabel("Epochs")
    plt.ylabel("mAP")
    plt.savefig(os.path.join(model_path, "mAP_evo.png"))
    if show:
        plt.show()
    plt.close()

    plt.figure()
    plt.plot(epochs, history["lr"])
    plt.title("Learning Rate evolution")
    plt.xlabel("Epochs")
    plt.ylabel("LR")
    plt.savefig(os.path.join(model_path, "lr_evo.png"))
    if show:
        plt.show()
    plt.close()


def train_epoch(loader, loss_fn, model, epoch, scheduler, optimizer, normalize, device):
    train_loss = 0.0
    n_tot_chunks = 0
    model.train()
    pbar = tqdm(loader, desc="Training", unit="batch")
    for step, (X, y, mask_n, id) in enumerate(pbar):
        X = X.to(device).float()
        y = y.to(device).float()

        scheduler.step(step + len(loader) * (epoch - 1))
        pred = model(X)

        loss = loss_fn(pred.double(), y.double()).double().mean().float()

        #         #convert mask
        #         mask = torch.zeros(y.shape)
        #         for i, m in enumerate(mask_n):
        #             mask[i,m[0]:m[1],:] = 1
        #         loss = (loss_fn(pred, y) * mask).sum() / mask.sum()

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e-1)
        optimizer.step()
        optimizer.zero_grad()

        train_loss += loss.item()
        n_tot_chunks += 1
        gc.collect()
    train_loss /= len(loader)
    return model, train_loss


def train(
    dataclass: DataClass,
    data_dir: str,
    model_name: str,
    objective: str,
    sequence_length: int = None,
    downsample: int = 5,
    agg_feats: bool = True,
    folds: int = 5,
    epochs: int = 5,
    bs: int = 16,
    normalize: bool = False,
    use_cat: bool = True,
    device: str = ("cuda" if torch.cuda.is_available() else "cpu"),
    workers: int = 1,
):
    dataset_name = dataclass.name
    dataset_construct = dataclass.dataset_construct

    save_model_path = Path(f"./experiments/{dataset_name}/{model_name}/{objective}")
    save_pred_dir = save_model_path / "predictions"
    save_pred_dir.mkdir(parents=True, exist_ok=True)

    loss_fn = get_loss(objective)
    kfold = KFold(n_splits=folds, shuffle=True, random_state=0)

    for fold in range(folds):
        train_dataset = dataset_construct(
            dataclass,
            data_dir,
            fold,
            kfold=kfold,
            training=True,
            downsample=downsample,
            agg_feats=agg_feats,
            sequence_length=sequence_length,
            target_type=objective,
            normalize=normalize,
            use_cat=use_cat,
        )
        gc.collect()
        val_dataset = dataset_construct(
            dataclass,
            data_dir,
            fold,
            kfold=kfold,
            training=False,
            downsample=downsample,
            agg_feats=agg_feats,
            sequence_length=sequence_length,
            target_type=objective,
            normalize=normalize,
            use_cat=use_cat,
        )
        gc.collect()
        model = get_model(
            dataclass,
            model_name,
            objective,
            sequence_length // downsample,
            downsample,
            agg_feats=agg_feats,
            use_cat=use_cat,
        ).to(device)

        train_loader = DataLoader(
            train_dataset,
            batch_size=bs,
            num_workers=workers,
            shuffle=True,
        )

        steps = len(train_loader) * epochs
        warmup_steps = int(steps * 0.1)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)
        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=steps,
            warmup_t=warmup_steps,
            warmup_lr_init=1e-6,
            lr_min=1e-6,
            cycle_mul=1,
        )

        history = {
            "train_loss": [],
            "valid_loss": [],
            "valid_mAP": [],
            "lr": [],
        }
        best_valid_loss = np.inf

        for epoch in range(1, epochs + 1):
            model, train_loss = train_epoch(
                train_loader,
                loss_fn,
                model,
                epoch,
                scheduler,
                optimizer,
                normalize,
                device,
            )
            valid_loss, valid_mAP = evaluate(
                dataclass,
                objective,
                model_name,
                model,
                val_dataset,
                device,
                workers,
                (
                    save_pred_dir if epoch == epochs else None
                ),  # only save predictons on last epoch
            )
            print(
                f"fold {fold}, epoch {epoch}/{epochs}: train loss: {train_loss:.3f}, valid loss: {valid_loss:.3f}, valid mAP: {valid_mAP:.3f}"
            )
            history["train_loss"].append(train_loss)
            history["valid_mAP"].append(valid_mAP)
            history["valid_loss"].append(valid_loss)
            history["lr"].append(optimizer.param_groups[0]["lr"])

        # save and plot the evolution of the metrics over time
        plot_history(history, model_path=save_model_path)

        model_path = os.path.join(save_model_path, f"model_{fold}.pth")
        torch.save(model.state_dict(), model_path)

        history_path = os.path.join(save_model_path, f"history_{fold}.json")
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=4)

        del (
            train_dataset,
            train_loader,
            val_dataset,
            model,
            optimizer,
            scheduler,
            history,
        )
        gc.collect()

    full_dataset = dataclass.dataset_construct(
        dataclass,
        data_dir,
        -1,
        training=False,
        downsample=downsample,
        agg_feats=agg_feats,
        sequence_length=sequence_length,
        target_type=objective,
    )
    get_optimal_cutoff(
        dataclass, model_name, objective, full_dataset, save_pred_dir, workers=workers
    )


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
    parser.add_argument(
        "--model",
        type=str,
        required=True,
    )  # choices=['rnn', 'unet', 'unet_t', 'prectime']
    parser.add_argument(
        "--objective", type=str, required=True, choices=["seg", "hard", "gau", "custom"]
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
        "training and evaluation script", parents=[get_args_parser()]
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

    train(
        dataclass=dataclass,
        data_dir=args.datadir,
        model_name=args.model,
        objective=args.objective,
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
