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

from src.metrics import score, score_all
from src.sleep import get_sleep_dataclass
from src.bowshock import get_bowshock_dataclass
from src.utils import *


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
    dataclass: DataClass,
    predictions: np.ndarray,
    objective: str,
    threshold: float = None,  # between 0 and 1
    smooth: int = None,
):
    days = len(predictions) // dataclass.day_length + 1
    max_distance = dataclass.max_distance

    if threshold == None:
        # default values
        threshold = 0.5 if (objective[:3] == "seg") else 0
    if smooth:
        for i in range(predictions.shape[1]):
            predictions[:, i] = gaussian_filter1d(predictions[:, i], smooth)

    if objective[:3] == "seg" and dataclass.event_type == "interval":
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


def generate_df(
    id,
    prediction,
    dataclass,
    dataset,
    objective,
    threshold=None,
    smooth=None,
):

    downsample = dataset.downsample
    prediction = np.repeat(
        prediction, downsample, 0
    )  # get rid of downsampling by expanding

    if dataclass.event_type == "interval":

        (onsets, offsets), (onsets_score, offsets_score) = get_candidates(
            dataclass,
            prediction,
            objective,
            threshold,
            smooth,
        )
        onset = dataset.data[id][["step"]].iloc[onsets].astype(np.int32)
        onset["event"] = "onset"
        onset["series_id"] = id
        onset["score"] = onsets_score
        offset = dataset.data[id][["step"]].iloc[offsets].astype(np.int32)
        offset["event"] = "offset"
        offset["series_id"] = id
        offset["score"] = offsets_score
        return pd.concat([onset, offset], axis=0)
    else:
        loc, loc_score = get_candidates(
            dataclass,
            prediction,
            objective,
            threshold,
            smooth,
        )
        dfpred = dataset.data[id][["step"]].iloc[loc[0]].astype(np.int32)
        dfpred["event"] = "event"
        dfpred["series_id"] = id
        dfpred["score"] = loc_score[0]
        return dfpred


def evaluate(
    dataclass: DataClass,
    objective: str,
    model_name: str,
    model: torch.nn.Module,
    dataset: Dataset,
    device: str,
    workers: int = 1,
    save_pred_dir: Path = None,
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

    truth = dataset.events
    downsample = dataset.downsample

    max_distance = dataclass.max_distance // downsample
    day_length = dataclass.day_length // downsample
    if dataclass.event_type == "interval":
        tolerances = {
            "onset": dataclass.tolerances,
            "offset": dataclass.tolerances,
        }
    elif dataclass.event_type == "point":
        tolerances = {
            "event": dataclass.tolerances,
        }
    column_names = dataclass.column_names
    combine_series_id = dataclass.combine_series_id

    submission = pd.DataFrame()
    for step, (X, y, mask, id) in enumerate(tqdm(loader, desc="Eval", unit="batch")):
        id = id[0]
        if np.sum(truth.series_id == id) == 0 and (not combine_series_id):
            continue

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
            np.save(save_pred_dir / f"{id}.npy", prediction)
        gc.collect()

        pred_df = generate_df(id, prediction, dataclass, dataset, objective)
        submission = pd.concat([submission, pred_df], axis=0)

        gc.collect()
    submission = submission.sort_values(["series_id", "step"]).reset_index(drop=True)
    submission["row_id"] = submission.index.astype(int)
    submission.set_index("row_id")
    submission["score"] = submission["score"].fillna(submission["score"].mean())
    submission = submission[["row_id", "series_id", "step", "event", "score"]]
    if save_pred_dir:
        submission.to_csv("example_sub.csv")
        truth.to_csv("example_tru.csv")
    if combine_series_id:
        submission["series_id"] = 0
        truth["series_id"] = 0

    valid_loss /= len(loader)
    gc.collect()

    if len(submission) == 0:
        mAP = 0
    else:
        mAP = score(truth, submission, tolerances, **column_names)
    gc.collect()
    return valid_loss, mAP


def format_score_output(score_dict):
    string = ""
    for name, score in score_dict.items():
        string = string + f"{name} = {score:.3f}, "
    return string


def get_optimal_cutoff(
    dataclass: DataClass,
    model_name: str,
    objective: str,
    dataset: Dataset,
    save_pred_dir: Path,
    workers: int = 1,
):

    if objective == "seg":
        objectives = ["seg1", "seg2"]
    else:
        objectives = [objective]

    downsample = dataset.downsample
    truth = dataset.events

    max_distance = dataclass.max_distance // downsample
    day_length = dataclass.day_length // downsample
    column_names = dataclass.column_names
    combine_series_id = dataclass.combine_series_id

    if dataclass.event_type == "interval":
        tolerances = {
            "onset": dataclass.tolerances,
            "offset": dataclass.tolerances,
        }
    else:
        tolerances = {"event": dataclass.tolerances}

    def get_score(objective, cutoff, smooth_param, tolerances):
        submission = pd.DataFrame()

        for id in dataset.ids:
            if np.sum(truth.series_id == id) == 0 and (not combine_series_id):
                continue
            pred = np.load(save_pred_dir / f"{id}.npy")

            pred_df = generate_df(
                id, pred, dataclass, dataset, objective, cutoff, smooth_param
            )
            submission = pd.concat([submission, pred_df], axis=0)

        if len(submission) == 0:
            return None
        submission = submission.sort_values(["series_id", "step"]).reset_index(
            drop=True
        )
        submission["row_id"] = submission.index.astype(int)
        submission.set_index("row_id")
        submission["score"] = submission["score"].fillna(submission["score"].mean())
        submission = submission[["row_id", "series_id", "step", "event", "score"]]
        if combine_series_id:
            submission["series_id"] = 0
            truth["series_id"] = 0

        scores = score_all(truth, submission, tolerances, **column_names)
        return scores

    for obj in objectives:
        print(f"{model_name} {obj} results: ")

        default_scores = get_score(obj, None, None, tolerances)
        score_metrics = default_scores.keys()
        print(f" default scores: {format_score_output(default_scores)}")

        if objective[:3] == "seg":
            max_pred = 1
        else:
            max_pred = 1 / normalize_error(dataclass, objective)
        cutoff_space = np.linspace(0, max_pred, 15)
        smooth_space = [None, 1, 4, 10, 20, 40, 100]

        for metric in score_metrics:
            print(f" optimizing hyperparams for {metric}:")
            best_params = {}
            for cutoff, smooth_param in tqdm(
                itertools.product(cutoff_space, smooth_space),
                desc=" Optimizing",
                total=len(cutoff_space) * len(smooth_space),
            ):

                scores = get_score(obj, cutoff, smooth_param, tolerances)
                if not scores:  # no score available
                    continue

                for name, score in scores.items():

                    best_score, best_param = best_params.get(name, (0, None))
                    best_score = max(best_score, score)
                    if best_score == score:
                        best_param = (cutoff, smooth_param)

                    best_params[name] = (best_score, best_param)
            best_scores = {n: s for n, (s, p) in best_params.items()}

            print(
                f"  best params: cutoff = {best_params[metric][1][0]}, smoothing = {best_params[metric][1][1]}"
            )
            print(f"  best scores: {format_score_output(best_scores)}")
            tol_scores = []
            for tol in dataclass.tolerances:
                if dataclass.event_type == "interval":
                    tolerances_ = {"onset": [tol], "offset": [tol]}
                if dataclass.event_type == "point":
                    tolerances_ = {"event": [tol]}
                scores = get_score(obj, best_param[0], best_param[1], tolerances_)
                tol_scores.append(scores)
                print(f"   tolerance {tol}: {format_score_output(scores)}")


def get_best_scores(
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

    save_model_path = Path(f"./experiments/{dataset_name}/{model_name}/{objective}/")
    save_pred_dir = save_model_path / "predictions"

    loss_fn = get_loss(objective)
    kfold = KFold(n_splits=folds, shuffle=True, random_state=0)

    full_dataset = dataset_construct(
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
    parser = argparse.ArgumentParser("get evaluation parameters", add_help=False)

    parser.add_argument(
        "--datadir",
        default="./data",
        type=str,
    )
    # type
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--objective", type=str, required=True)
    # data
    parser.add_argument("--downsample", default=10, type=int)
    parser.add_argument("--agg_feats", default=True, type=bool)
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
    else:
        raise ValueError(f"{args.dataset} dataset not supported")

    sequence_length = args.sequence_length
    if not sequence_length:
        sequence_length = dataclass.default_sequence_length

    get_best_scores(
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
