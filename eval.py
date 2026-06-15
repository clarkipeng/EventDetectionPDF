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
import json
import time

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
    alternating = bool(param.get("alternating", False))

    if threshold == None:
        # default values
        threshold = 0.5 if is_segmentation_objective(objective) else 0
    if smooth:
        for i in range(predictions.shape[1]):
            predictions[:, i] = gaussian_filter1d(predictions[:, i], smooth)

    if is_segmentation_objective(objective) and dataclass.event_type == "interval":
        postprocess_method = segmentation_postprocess_method(objective)
        if postprocess_method == 1:
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
            if alternating:
                candidates, c_scores = enforce_alternating_interval_candidates(
                    candidates,
                    c_scores,
                )
            return candidates, c_scores
        elif postprocess_method == 2:
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
    if alternating and dataclass.event_type == "interval":
        candidates, scores = enforce_alternating_interval_candidates(
            candidates,
            scores,
        )
    return candidates, scores


def enforce_alternating_interval_candidates(candidates, scores):
    events = []
    for channel in range(2):
        for loc, score in zip(candidates[channel], scores[channel]):
            events.append((int(loc), channel, float(score)))
    events = sorted(events, key=lambda event: (event[0], event[1]))

    selected = []
    current = None
    for event in events:
        loc, channel, score = event
        if current is None:
            if channel == 0:
                current = event
            continue
        if channel == current[1]:
            if score > current[2]:
                current = event
        else:
            selected.append(current)
            current = event

    if current is not None and current[1] == 1:
        selected.append(current)

    filtered_candidates, filtered_scores = [], []
    for channel in range(2):
        channel_events = [event for event in selected if event[1] == channel]
        filtered_candidates.append(
            np.array([event[0] for event in channel_events], dtype=int)
        )
        filtered_scores.append(
            np.array([event[2] for event in channel_events], dtype=float)
        )
    return filtered_candidates, filtered_scores


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
    density_prior: str = "sparse",
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
    truth = dataset.events
    downsample = dataset.downsample
    loss_fn = get_loss(
        objective,
        dataclass=dataclass,
        downsample=downsample,
        density_prior=density_prior,
    )

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

        if is_segmentation_objective(objective):
            prediction = prediction.sigmoid()
        elif is_density_objective(objective):
            prediction = density_logits_to_rates(
                prediction,
                dataclass,
                downsample,
                density_prior=density_prior,
            )

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

    truth_eval = truth.copy()
    if combine_series_id:
        submission["series_id"] = 0
        truth_eval["series_id"] = 0

    valid_loss /= len(loader)
    gc.collect()

    if len(submission) == 0:
        mAP = 0
    else:
        mAP = calculate_score(
            truth_eval, submission, tolerances, **column_names, metrics=["mAP"]
        )["mAP"]
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


def serializable_score(value):
    if isinstance(value, np.ndarray):
        return [serializable_score(v) for v in value.tolist()]
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, dict):
        return {k: serializable_score(v) for k, v in value.items()}
    if isinstance(value, list):
        return [serializable_score(v) for v in value]
    return value


def score_dict_for_json(scores):
    return {key: serializable_score(value) for key, value in scores.items()}


def get_optimal_cutoff(
    dataclass: DataClass,
    model_name: str,
    objective: str,
    dataset: Dataset,
    save_pred_dir: Path,
    workers: int = 1,
    metadata: Dict = None,
):
    start_time = time.perf_counter()

    objectives = segmentation_postprocess_objectives(objective)

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
        score_mean = submission["score"].mean()
        if np.isnan(score_mean):
            score_mean = 0
        submission["score"] = submission["score"].fillna(score_mean)
        submission = submission[["row_id", "series_id", "step", "event", "score"]]
        truth_eval = truth.copy()
        if combine_series_id:
            submission["series_id"] = 0
            truth_eval["series_id"] = 0

        scores = calculate_score(
            truth_eval,
            submission,
            tolerances,
            metrics=evaluation_metrics,
            **column_names,
        )
        return scores

    def get_prediction_max():
        max_pred = 0.0
        for id in dataset.ids:
            pred_path = save_pred_dir / f"{id}.npy"
            if not pred_path.exists():
                continue
            pred = np.load(pred_path)
            if pred.size == 0:
                continue
            max_pred = max(max_pred, float(np.nanmax(pred)))
        return max_pred if max_pred > 0 else 1.0

    def get_scores_param(param):
        param = {k: param[i] for i, k in enumerate(hyperparam_dict.keys())}

        scores = get_score(obj, param, tolerances)
        return param, scores

    def get_scores_tolerance(param, tol):
        if dataclass.event_type == "interval":
            tolerances_ = {"onset": [tol], "offset": [tol]}
        if dataclass.event_type == "point":
            tolerances_ = {"event": [tol]}
        scores = get_score(obj, param, tolerances_)
        return tol, scores

    metadata = metadata or {}
    result_payload = {
        "metadata": {
            **metadata,
            "dataset": metadata.get("dataset", dataclass.name),
            "model": metadata.get("model", model_name),
            "objective": objective,
            "postprocess_objectives": objectives,
            "prediction_dir": str(save_pred_dir),
        },
        "results": [],
    }
    rows = []

    for obj in objectives:
        print(f"{model_name} {obj} results: ")

        default_scores = get_score(obj, {}, tolerances)
        print(
            f" default scores: {format_score_output({k:default_scores[k] for k in evaluation_metrics})}"
        )
        result_obj = {
            "postprocess_objective": obj,
            "default_scores": score_dict_for_json(default_scores),
            "optimized": {},
        }
        for metric in evaluation_metrics:
            rows.append(
                {
                    **result_payload["metadata"],
                    "row_type": "summary",
                    "stage": "default",
                    "postprocess_objective": obj,
                    "optimized_metric": "",
                    "metric": metric,
                    "tolerance": "",
                    "score": serializable_score(default_scores.get(metric, 0)),
                    "params": "{}",
                }
            )

        if is_segmentation_objective(obj):
            max_pred = 1
        elif is_density_objective(obj):
            max_pred = get_prediction_max()
        else:
            max_pred = 1 / normalize_error(dataclass, obj)

        hyperparam_dict = {}
        if "cutoff" in hyperparams_tune:
            hyperparam_dict["cutoff"] = np.linspace(0, max_pred, 11)
        if "smooth" in hyperparams_tune:
            hyperparam_dict["smooth"] = [None, 1, 10, 100, 1000]
        if "prominence" in hyperparams_tune:
            hyperparam_dict["prominence"] = np.linspace(0, max_pred * 0.5, 8)
        if "distance" in hyperparams_tune:
            hyperparam_dict["distance"] = np.geomspace(
                1, 1000 * max_distance, 8
            ).astype(int)
        if dataclass.event_type == "interval":
            hyperparam_dict["alternating"] = [False, True]

        param_search = list(
            itertools.product(*[hyperparam_dict[k] for k in hyperparam_dict.keys()])
        )
        if workers > 1:
            param_scores = joblib.Parallel(n_jobs=workers, require="sharedmem")(
                joblib.delayed(get_scores_param)(param)
                for param in tqdm(param_search, desc=" Optimizing")
            )
        else:
            param_scores = [
                get_scores_param(param)
                for param in tqdm(param_search, desc=" Optimizing")
            ]

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
            if metric not in best_params:
                best_params[metric] = (default_scores.get(metric, 0), {})
            print(f" optimizing hyperparams for {metric}:")
            print(f"  best params: {format_score_output(best_params[metric][1])}")
            best_scores = get_score(obj, best_params[metric][1], tolerances)
            print(
                f"  best scores: {format_score_output({k:best_scores[k] for k in evaluation_metrics})}"
            )
            result_obj["optimized"][metric] = {
                "best_score": serializable_score(best_params[metric][0]),
                "best_params": score_dict_for_json(best_params[metric][1]),
                "scores": score_dict_for_json(best_scores),
            }
            for score_name in evaluation_metrics:
                rows.append(
                    {
                        **result_payload["metadata"],
                        "row_type": "summary",
                        "stage": "tuned",
                        "postprocess_objective": obj,
                        "optimized_metric": metric,
                        "metric": score_name,
                        "tolerance": "",
                        "score": serializable_score(best_scores.get(score_name, 0)),
                        "params": json.dumps(
                            score_dict_for_json(best_params[metric][1]),
                            sort_keys=True,
                        ),
                    }
                )

            if f"{metric}_tolerances" in best_scores.keys():
                tol_scores = best_scores[f"{metric}_tolerances"]
            else:
                if workers > 1:
                    tol_scores = joblib.Parallel(n_jobs=workers, require="sharedmem")(
                        joblib.delayed(get_scores_tolerance)(
                            best_params[metric][1], tol
                        )
                        for tol in dataclass.tolerances
                    )
                else:
                    tol_scores = [
                        get_scores_tolerance(best_params[metric][1], tol)
                        for tol in dataclass.tolerances
                    ]

            tolerance_records = []
            for tol, scores in zip(dataclass.tolerances, tol_scores):
                print(f"   tolerance {tol}: {metric} = {scores}")
                if isinstance(scores, tuple):
                    tol_value, score_value = scores
                    score_value = score_value.get(metric, 0)
                else:
                    tol_value, score_value = tol, scores
                tolerance_records.append(
                    {
                        "tolerance": serializable_score(tol_value),
                        "score": serializable_score(score_value),
                    }
                )
                rows.append(
                    {
                        **result_payload["metadata"],
                        "row_type": "tolerance",
                        "stage": "tuned",
                        "postprocess_objective": obj,
                        "optimized_metric": metric,
                        "metric": metric,
                        "tolerance": serializable_score(tol_value),
                        "score": serializable_score(score_value),
                        "params": json.dumps(
                            score_dict_for_json(best_params[metric][1]),
                            sort_keys=True,
                        ),
                    }
                )
            result_obj["optimized"][metric]["tolerance_scores"] = tolerance_records
        result_payload["results"].append(result_obj)

    result_payload["metadata"]["score_runtime_sec"] = time.perf_counter() - start_time
    results_dir = save_pred_dir.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "scores.json", "w", encoding="utf-8") as f:
        json.dump(result_payload, f, indent=2)
    pd.DataFrame(rows).to_csv(results_dir / "scores.csv", index=False)
    return result_payload


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
    seed: int = 0,
):
    dataset_name = dataclass.name
    dataset_construct = dataclass.dataset_construct

    save_model_path = experiment_path(dataset_name, model_name, objective, seed)
    save_pred_dir = save_model_path / "predictions"

    loss_fn = get_loss(objective, dataclass=dataclass, downsample=downsample)
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
        dataclass,
        model_name,
        objective,
        full_dataset,
        save_pred_dir,
        workers=workers,
        metadata={
            "dataset": dataset_name,
            "model": model_name,
            "objective": objective,
            "seed": seed,
            "sequence_length": sequence_length,
            "downsample": downsample,
            "agg_feats": agg_feats,
            "folds": folds,
            "epochs": epochs,
            "batch_size": bs,
            "normalize": normalize,
            "use_cat": use_cat,
            "device": device,
        },
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
        "--objective",
        type=str,
        required=True,
        choices=OBJECTIVE_CHOICES,
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
    parser.add_argument("--use_cat", default=True, type=str2bool)
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
    parser.add_argument("--normalize", default=True, type=str2bool)
    parser.add_argument("--gaussian_sigma", default=None, type=float)
    parser.add_argument("--tolerance_scale", default=1.0, type=float)
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
    dataclass = apply_objective_overrides(
        dataclass,
        gaussian_sigma=args.gaussian_sigma,
        tolerance_scale=args.tolerance_scale,
    )

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
        seed=args.seed,
    )
