import sys, os

sys.path.append("../")

import gc
import random
import argparse
import warnings
import itertools

import numpy as np
import pandas as pd

from tqdm.auto import tqdm

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from sklearn.model_selection import KFold
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

import matplotlib.pyplot as plt
from numba import njit, jit

from src.load_dataset import SleepDataset, normalize_error
from src.EDAP import score
from src.utils import get_loss


tolerances = {
    "onset": [12, 36, 60, 90, 120, 150, 180, 240, 300, 360],
    "wakeup": [12, 36, 60, 90, 120, 150, 180, 240, 300, 360],
}

column_names = {
    "series_id_column_name": "series_id",
    "time_column_name": "step",
    "event_column_name": "event",
    "score_column_name": "score",
}


@njit
def transform_segmentation(predictions: np.ndarray, interval: int):
    scores = np.zeros(len(predictions), dtype=np.float32)

    score_delta_0 = np.sum(predictions[:interval])
    score_delta_1 = np.sum(predictions[interval : 2 * interval])
    for i in range(interval, len(predictions) - interval):
        scores[i] = score_delta_1 - score_delta_0

        score_delta_0 = score_delta_0 - predictions[i - interval] + predictions[i]
        score_delta_1 = score_delta_1 - predictions[i] + predictions[i + interval]
    return scores / interval


def get_candidates(
    predictions: np.ndarray,
    objective: str,
    max_distance: int,
    day_length: int,
    threshold: float = None,  # between 0 and 1
    smooth: int = None,
):
    days = len(predictions) // day_length + 1

    if threshold == None:
        # default values
        threshold = 0.5 if (objective[:3] == "seg") else 0
    if smooth:
        for i in range(predictions.shape[1]):
            predictions[:, i] = gaussian_filter1d(predictions[:, i], smooth)

    if objective[:3] == "seg":
        if len(objective) == 3 or objective[3] == "1":
            candidates, c_scores = [], []

            events = (predictions[1:] > threshold).astype(int) - (
                predictions[:-1] > threshold
            ).astype(int)

            locations = np.where(events == 1)[0]
            scores = np.array(
                [
                    abs(
                        np.mean(predictions[max(0, loc - max_distance) : loc + 1])
                        - np.mean(
                            predictions[
                                loc - 1 : min(len(predictions), loc + max_distance)
                            ]
                        )
                    )
                    for loc in locations
                ]
            )
            candidates.append(locations)
            c_scores.append(scores)

            locations = np.where(events == -1)[0]
            scores = np.array(
                [
                    abs(
                        np.mean(predictions[max(0, loc - max_distance) : loc + 1])
                        - np.mean(
                            predictions[
                                loc - 1 : min(len(predictions), loc + max_distance)
                            ]
                        )
                    )
                    for loc in locations
                ]
            )
            candidates.append(locations)
            c_scores.append(scores)
            return candidates, c_scores
        elif objective[3] == "2":
            scores = transform_segmentation(predictions[:, 0], max_distance)

            predictions = np.zeros((predictions.shape[0], 2))
            predictions[:, 0] = scores
            predictions[:, 1] = -scores
        else:
            raise ValueError(f"{objective} not valid")

    candidates = []
    scores = []

    for i in range(predictions.shape[1]):

        cand = find_peaks(predictions[:, i], height=threshold, distance=max_distance)[0]

        #         cand = cand[np.argsort(predictions[cand,i])[-2*days:]]

        candidates.append(cand)
        scores.append(predictions[cand, i])
    return candidates, scores


def evaluate(
    objective: str,
    model_name: str,
    model: torch.nn.Module,
    dataset: Dataset,
    device: str,
    workers: int = 1,
    save_pred_dir: str = None,
):
    model.eval()
    valid_loss = 0.0
    y_preds = []

    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=workers,
        shuffle=False,
    )
    loss_fn = get_loss(objective)
    downsample = dataset.downsample
    max_distance = (30 * 12) // downsample
    day_length = (24 * 60 * 12) // downsample

    truth = dataset.events

    submission = pd.DataFrame()
    for step, (X, y, mask, id) in enumerate(tqdm(loader, desc="Eval", unit="batch")):
        id = id[0]
        if dataset.sequence_length is None:
            X, y = X.to(device), y.to(device)
            prediction = model(X).detach().cpu()
            target = y.cpu()
        else:
            X = torch.concat(X, 0).to(device).float()
            y = torch.concat(y, 0).to(device).float()
            mask = torch.concat(mask, 0)

            pred = model(X).detach().cpu()
            y = y.cpu()

            prediction, target = [], []
            for pred_, y_, mask_ in zip(pred, y, mask):
                prediction.append(pred_[mask_[0] : mask_[1], :])
                target.append(y_[mask_[0] : mask_[1], :])
            prediction = torch.concat(prediction)
            target = torch.concat(target)

        loss = loss_fn(prediction, target).mean()
        valid_loss += loss.item()

        if objective[:3] == "seg":
            prediction = prediction.sigmoid()

        prediction = prediction.numpy()

        if save_pred_dir:
            np.save(f"{save_pred_dir}/{id}.npy", prediction)
        gc.collect()

        (onsets, wakeups), (onsets_score, wakeups_score) = get_candidates(
            prediction, objective, max_distance, day_length
        )

        data_len = len(dataset.data[id])
        onset = (
            dataset.data[id][["step"]]
            .iloc[np.clip(onsets * downsample + downsample // 2, 0, data_len - 1)]
            .astype(np.int32)
        )
        onset["event"] = "onset"
        onset["series_id"] = id
        onset["score"] = onsets_score
        wakeup = (
            dataset.data[id][["step"]]
            .iloc[np.clip(wakeups * downsample + downsample // 2, 0, data_len - 1)]
            .astype(np.int32)
        )
        wakeup["event"] = "wakeup"
        wakeup["series_id"] = id
        wakeup["score"] = wakeups_score
        submission = pd.concat([submission, onset, wakeup], axis=0)

        # if step==1:
        #     cands = np.zeros((len(prediction),2))
        #     cands[onsets,0] = onsets_score
        #     cands[wakeups,1] = wakeups_score

        #     plt.plot(cands[:day_length])
        #     plt.show()
        #     plt.plot(prediction[:day_length])
        #     plt.show()
        #     plt.plot(target.numpy()[:day_length])
        #     plt.show()

        del onsets, wakeups, onset, wakeup
        gc.collect()
    submission = submission.sort_values(["series_id", "step"]).reset_index(drop=True)
    submission["row_id"] = submission.index.astype(int)
    submission.set_index("row_id")
    submission["score"] = submission["score"].fillna(submission["score"].mean())
    submission = submission[["row_id", "series_id", "step", "event", "score"]]

    valid_loss /= len(loader)
    gc.collect()

    if len(submission) == 0:
        mAP_score = 0
    else:
        mAP_score = score(truth, submission, tolerances, **column_names)
    gc.collect()
    return valid_loss, mAP_score


def get_optimal_cutoff(
    model_name: str,
    objective: str,
    dataset: Dataset,
    save_pred_dir: str,
    workers: int = 1,
):

    if objective == "seg":
        objectives = ["seg1", "seg2"]
    else:
        objectives = [objective]

    def get_score(objective, cutoff, smooth_param, tolerances):

        truth = dataset.events
        submission = pd.DataFrame()
        downsample = dataset.downsample
        max_distance = (30 * 12) // downsample
        day_length = (24 * 60 * 12) // downsample

        for id in dataset.ids:
            pred = np.load(f"{save_pred_dir}/{id}.npy")

            (onsets, wakeups), (onsets_score, wakeups_score) = get_candidates(
                pred,
                objective,
                max_distance,
                day_length,
                threshold=cutoff,
                smooth=smooth_param,
            )

            data_len = len(dataset.data[id])
            onset = (
                dataset.data[id][["step"]]
                .iloc[np.clip(onsets * downsample + downsample // 2, 0, data_len - 1)]
                .astype(np.int32)
            )
            onset["event"] = "onset"
            onset["series_id"] = id
            onset["score"] = onsets_score
            wakeup = (
                dataset.data[id][["step"]]
                .iloc[np.clip(wakeups * downsample + downsample // 2, 0, data_len - 1)]
                .astype(np.int32)
            )
            wakeup["event"] = "wakeup"
            wakeup["series_id"] = id
            wakeup["score"] = wakeups_score
            submission = pd.concat([submission, onset, wakeup], axis=0)

            del onsets, wakeups, onset, wakeup
            gc.collect()

        if len(submission) == 0:
            return 0
        submission = submission.sort_values(["series_id", "step"]).reset_index(
            drop=True
        )
        submission["row_id"] = submission.index.astype(int)
        submission.set_index("row_id")
        submission["score"] = submission["score"].fillna(submission["score"].mean())
        submission = submission[["row_id", "series_id", "step", "event", "score"]]

        mAP_score = score(truth, submission, tolerances, **column_names)
        #         mAP_score = fast_score(truth, submission)
        return mAP_score

    for obj in objectives:
        print(f"{model_name} {obj} results: ")

        default_score = get_score(obj, 0.5, None, tolerances)
        print(f" default score = {default_score:.3f}")

        if objective[:3] == "seg":
            max_pred = 1
            cutoff_space = np.linspace(0, max_pred, 15)
            smooth_space = [None] + [i for i in range(1, 21, 10)]
        else:
            max_pred = 1 / normalize_error(objective)
            cutoff_space = np.linspace(0, max_pred, 15)
            smooth_space = [None] + [i for i in range(1, 21, 10)]

        print(" optimize hyperparams:")
        best_score, best_param = 0, (None, None)
        for cutoff, smooth_param in tqdm(
            itertools.product(cutoff_space, smooth_space),
            desc="Optimizing",
            total=len(cutoff_space) * len(smooth_space),
        ):

            mAP_score = get_score(obj, cutoff, smooth_param, tolerances)
            best_score = max(mAP_score, best_score)

            if best_score == mAP_score:
                best_param = (cutoff, smooth_param)

        print(f"  best params: cutoff = {best_param[0]}, smoothing = {best_param[1]}")
        print(f"  best score = {best_score:.3f}")
        tol_scores = []
        for tol in tolerances["wakeup"]:
            tolerances_ = {"onset": [tol], "wakeup": [tol]}
            mAP_score = get_score(obj, best_param[0], best_param[1], tolerances_)
            tol_scores.append(tol_scores)
            print(f"   tolerance {tol} = {mAP_score:.3f}")


def get_best_scores(
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
    use_time_cat: bool = True,
    device: str = ("cuda" if torch.cuda.is_available() else "cpu"),
    workers: int = 1,
):
    model_path = f"./experiments/{model_name}"

    obj = "seg" if "seg" == objective[:3] else objective
    save_pred_dir = f"./experiments/{model_name}/{obj}/predictions"
    loss_fn = get_loss(objective)
    kfold = KFold(n_splits=folds, shuffle=True, random_state=0)
    full_dataset = SleepDataset(
        data_dir,
        -1,
        training=False,
        downsample=downsample,
        agg_feats=agg_feats,
        sequence_length=sequence_length,
        target_type=objective,
    )
    get_optimal_cutoff(
        model_name, objective, full_dataset, save_pred_dir, workers=workers
    )


def get_args_parser():
    parser = argparse.ArgumentParser("get evaluation parameters", add_help=False)

    parser.add_argument(
        "--datadir",
        default="./data",
        type=str,
    )
    # type
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--objective", type=str, required=True)
    # data
    parser.add_argument("--downsample", default=10, type=int)
    parser.add_argument("--agg_feats", default=True, type=bool)
    parser.add_argument("--use_time_cat", default=True, type=bool)
    parser.add_argument(
        "--sequence_length",
        default=7 * (24 * 60 * 12),
        type=int,
        help="length of training timeseries where each step = 5s",
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

    get_best_scores(
        data_dir=args.datadir,
        model_name=args.model,
        objective=args.objective,
        sequence_length=args.sequence_length,
        downsample=args.downsample,
        agg_feats=args.agg_feats,
        folds=args.folds,
        epochs=args.epochs,
        bs=args.bs,
        normalize=args.normalize,
        use_time_cat=args.use_time_cat,
        device=args.device,
        workers=args.workers,
    )
