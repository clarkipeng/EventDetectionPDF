import sys, os

sys.path.append("../")

import gc
import random
import argparse
import warnings
import itertools
import multiprocessing
import functools
import joblib

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

from src.metrics import calculate_score

from src.sleep import get_sleep_dataclass
from src.bowshock import get_bowshock_dataclass
from src.fraud import get_fraud_dataclass
from src.seizure import get_seizure_dataclass

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
    param: Dict = {},
):
    days = len(predictions) // dataclass.day_length + 1
    max_distance = dataclass.max_distance

    threshold = param.get("cutoff", None)
    smooth = param.get("smooth", None)
    prominence = param.get("prominence", None)
    max_distance = param.get("distance", max_distance)

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

        cand = find_peaks(
            predictions[:, i],
            height=threshold,
            distance=max_distance,
            prominence=prominence,
        )[0]
        # cand = find_peaks(predictions[:, i], height=threshold, distance=max_distance)[0]

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
    param={},
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
            param,
        )
        onset = dataset.get_step(id, onsets)
        onset["event"] = "onset"
        onset["series_id"] = id
        onset["score"] = onsets_score
        offset = dataset.get_step(id, offsets)
        offset["event"] = "offset"
        offset["series_id"] = id
        offset["score"] = offsets_score
        return pd.concat([onset, offset], axis=0)
    else:
        loc, loc_score = get_candidates(
            dataclass,
            prediction,
            objective,
            param,
        )
        dfpred = dataset.get_step(id, loc[0])
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

        loss = loss_fn(prediction.double(), target.double()).double().mean().float()
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
        mAP = calculate_score(truth, submission, tolerances, **column_names, metrics = ['mAP'])['mAP']
    gc.collect()
    return valid_loss, mAP


def format_score_output(score_dict):
    string = ""
    for name, score in score_dict.items():
        if isinstance(score, float):
            string = string + f"{name} = {score:.3f}, "
        else:
            string = string + f"{name} = {score}, "
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

    max_distance = dataclass.max_distance
    day_length = dataclass.day_length
    column_names = dataclass.column_names
    combine_series_id = dataclass.combine_series_id
    evaluation_metrics = dataclass.evaluation_metrics
    hyperparams_tune = dataclass.hyperparams_tune

    if dataclass.event_type == "interval":
        tolerances = {
            "onset": dataclass.tolerances,
            "offset": dataclass.tolerances,
        }
    else:
        tolerances = {"event": dataclass.tolerances}

    def get_score(objective, param, tolerances):
        submission = pd.DataFrame()

        for id in dataset.ids:
            if np.sum(truth.series_id == id) == 0 and (not combine_series_id):
                continue
            pred = np.load(save_pred_dir / f"{id}.npy")

            pred_df = generate_df(id, pred, dataclass, dataset, objective, param)
            submission = pd.concat([submission, pred_df], axis=0)

        if len(submission) == 0:
            return {k: 0 for k in evaluation_metrics}
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

        scores = calculate_score(truth, submission, tolerances, metrics = evaluation_metrics, **column_names)
        return scores
    
    def get_scores_param(param):
        param = {k: param[i] for i, k in enumerate(hyperparam_dict.keys())}

        scores = get_score(obj, param, tolerances)
        return param,scores
    
    def get_scores_tolerance(param, tol):
        if dataclass.event_type == "interval":
            tolerances_ = {"onset": [tol], "offset": [tol]}
        if dataclass.event_type == "point":
            tolerances_ = {"event": [tol]}
        scores = get_score(obj, param, tolerances_)
        return tol, scores

    for obj in objectives:
        print(f"{model_name} {obj} results: ")

        default_scores = get_score(obj, {}, tolerances)
        print(f" default scores: {format_score_output({k:default_scores[k] for k in evaluation_metrics})}")

        if objective[:3] == "seg":
            max_pred = 1
        else:
            max_pred = 1 / normalize_error(dataclass, objective)

        hyperparam_dict = {}
        if "cutoff" in hyperparams_tune:
            hyperparam_dict["cutoff"] = np.linspace(0, max_pred, 11)
        if "smooth" in hyperparams_tune:
            hyperparam_dict["smooth"] = [None, 1, 10, 100, 1000]
        if "prominence" in hyperparams_tune:
            hyperparam_dict["prominence"] = np.linspace(0, max_pred * 0.5, 8)
        if "distance" in hyperparams_tune:
            hyperparam_dict["distance"] = np.geomspace(1, 1000 * max_distance, 8).astype(
                int
            )

        
        param_search = list(itertools.product(*[hyperparam_dict[k] for k in hyperparam_dict.keys()]))
        if workers > 1:
            param_scores = joblib.Parallel(n_jobs=workers, require='sharedmem')(joblib.delayed(get_scores_param)(param) for param in tqdm(param_search,desc=" Optimizing"))
        else:
            param_scores = [score(obj, param, tolerances) for param in tqdm(param_search,desc=" Optimizing")]
            
        best_params = {}
        for param, scores in param_scores:
            if not scores:  # no score available
                continue

            for name, score in scores.items():
                if name not in evaluation_metrics:
                    continue

                best_score, best_param = best_params.get(name, (0, {}))
                if best_score < score:
                    best_param = param
                    best_score = score

                best_params[name] = (best_score, best_param)
        
        # best_scores = {n: s for n, (s, p) in best_params.items()}
        for metric in evaluation_metrics:
            print(f" optimizing hyperparams for {metric}:")
            print(f"  best params: {format_score_output(best_params[metric][1])}")
            best_scores = get_score(obj, best_params[metric][1], tolerances)
            print(f"  best scores: {format_score_output({k:best_scores[k] for k in evaluation_metrics})}")
            
            if f"{metric}_tolerances" in best_scores.keys():
                tol_scores = best_scores[f"{metric}_tolerances"]
            else:
                if workers > 1:
                    tol_scores = joblib.Parallel(n_jobs=workers, require='sharedmem')(joblib.delayed(get_scores_tolerance)(best_params[metric][1], tol) for tol in dataclass.tolerances)
                else:
                    tol_scores = [score(best_params[metric][1], tol) for tol in dataclass.tolerances]
            
            for tol, scores in zip(dataclass.tolerances,tol_scores):
                print(f"   tolerance {tol}: {metric} = {scores}")


def get_best_scores(
    dataclass: DataClass,
    data_dir: str,
    model_name: str,
    objective: str,
    sequence_length: int = None,
    downsample: int = 5,
    agg_feats: str = "stat",
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
        "--objective", type=str, required=True, choices=["seg", "seg1", "seg2", "hard", "gau", "custom"]
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
    parser.add_argument("--seed", default=0, type=int)

    return parser


if __name__ == "__main__":
    warnings.simplefilter("ignore", category=RuntimeWarning)
    parser = argparse.ArgumentParser(
        "training and evaluation script", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    set_random_seed(args.seed)

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
