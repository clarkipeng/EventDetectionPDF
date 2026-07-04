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
import re
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
from src.wandb_utils import (
    init_wandb_run,
    log_score_payload_to_wandb,
    log_wandb_artifact,
)


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
        predictions = predictions.copy()
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
    param=None,
    expanded=False,
):
    if param is None:
        param = {}

    downsample = dataset.downsample
    if not expanded:
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
        pin_memory=str(device).startswith("cuda"),
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


def parse_tune_smooth_values(value):
    if value is None:
        return [None, 1, 10, 100, 1000]
    if isinstance(value, (list, tuple)):
        return list(value)

    smooth_values = []
    for token in str(value).split(","):
        token = token.strip()
        if not token:
            continue
        if token.lower() in {"none", "null", "false"}:
            smooth_values.append(None)
        else:
            smooth = float(token)
            smooth_values.append(int(smooth) if smooth.is_integer() else smooth)
    return smooth_values or [None]


def parse_tune_cutoff_values(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple, np.ndarray)):
        return [float(v) for v in value]

    cutoff_values = []
    for token in str(value).split(","):
        token = token.strip()
        if token:
            cutoff_values.append(float(token))
    return cutoff_values or None


def parse_tolerance_values(value):
    if value in (None, "", "default"):
        return None
    if isinstance(value, (list, tuple, np.ndarray)):
        return [int(v) for v in value]

    tolerances = []
    for token in str(value).split(","):
        token = token.strip()
        if token:
            tolerances.append(int(float(token)))
    return tolerances or None


def score_filenames(score_suffix=None):
    if score_suffix in (None, ""):
        return "scores.csv", "scores.json"
    safe_suffix = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(score_suffix)).strip("_")
    if not safe_suffix:
        return "scores.csv", "scores.json"
    return f"scores_{safe_suffix}.csv", f"scores_{safe_suffix}.json"


def get_optimal_cutoff(
    dataclass: DataClass,
    model_name: str,
    objective: str,
    dataset: Dataset,
    save_pred_dir: Path,
    workers: int = 1,
    metadata: Dict = None,
    tune_cutoff_steps: int = 11,
    tune_cutoff_values=None,
    tune_smooth_values=None,
    tune_alternating: bool = True,
    score_suffix: str = None,
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
    tune_cutoff_steps = max(1, int(tune_cutoff_steps))
    tune_cutoff_values = parse_tune_cutoff_values(tune_cutoff_values)
    tune_smooth_values = parse_tune_smooth_values(tune_smooth_values)

    if dataclass.event_type == "interval":
        tolerances = {
            "onset": dataclass.tolerances,
            "offset": dataclass.tolerances,
        }
    else:
        tolerances = {"event": dataclass.tolerances}

    prediction_cache = {}
    for id in dataset.ids:
        if np.sum(truth.series_id == id) == 0 and (not combine_series_id):
            continue
        pred_path = save_pred_dir / f"{id}.npy"
        if pred_path.exists():
            prediction_cache[id] = np.load(pred_path)
    expanded_prediction_cache = {}
    score_cache = {}

    def normalize_value_for_cache(value):
        if value is None:
            return None
        if isinstance(value, (np.floating, float)):
            return float(value)
        if isinstance(value, (np.integer, int, bool)):
            return int(value)
        return value

    def normalize_param_for_cache(param):
        return tuple(
            sorted(
                (key, normalize_value_for_cache(value))
                for key, value in param.items()
            )
        )

    def normalize_tolerances_for_cache(tolerances_):
        return tuple(
            sorted(
                (
                    event,
                    tuple(int(tolerance) for tolerance in values),
                )
                for event, values in tolerances_.items()
            )
        )

    def get_expanded_prediction(id, smooth):
        smooth_key = None if not smooth else float(smooth)
        cache_key = (id, smooth_key)
        if cache_key not in expanded_prediction_cache:
            prediction = np.repeat(prediction_cache[id], downsample, 0)
            if smooth:
                prediction = prediction.copy()
                for channel in range(prediction.shape[1]):
                    prediction[:, channel] = gaussian_filter1d(
                        prediction[:, channel],
                        smooth,
                    )
            expanded_prediction_cache[cache_key] = prediction
        return expanded_prediction_cache[cache_key]

    def get_score(objective, param, tolerances):
        cache_key = (
            objective,
            normalize_param_for_cache(param),
            normalize_tolerances_for_cache(tolerances),
        )
        if cache_key in score_cache:
            return score_cache[cache_key]

        submission_parts = []
        param = dict(param)
        smooth = param.pop("smooth", None)

        for id in prediction_cache:
            prediction = get_expanded_prediction(id, smooth)
            pred_df = generate_df(
                id,
                prediction,
                dataclass,
                dataset,
                objective,
                param,
                expanded=True,
            )
            if len(pred_df) > 0:
                submission_parts.append(pred_df)

        if not submission_parts:
            scores = {k: 0 for k in evaluation_metrics}
            score_cache[cache_key] = scores
            return scores
        submission = pd.concat(submission_parts, axis=0, ignore_index=True)
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
        score_cache[cache_key] = scores
        return scores

    def get_prediction_max():
        max_pred = 0.0
        for pred in prediction_cache.values():
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
            "tune_cutoff_steps": tune_cutoff_steps,
            "tune_cutoff_values": tune_cutoff_values,
            "tune_smooth_values": [
                "none" if value is None else value for value in tune_smooth_values
            ],
            "tune_alternating": tune_alternating,
            "score_tolerances": list(dataclass.tolerances),
            "score_suffix": score_suffix or "",
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
            if tune_cutoff_values is not None:
                hyperparam_dict["cutoff"] = tune_cutoff_values
            else:
                hyperparam_dict["cutoff"] = np.linspace(0, max_pred, tune_cutoff_steps)
        if "smooth" in hyperparams_tune:
            hyperparam_dict["smooth"] = tune_smooth_values
        if "prominence" in hyperparams_tune:
            hyperparam_dict["prominence"] = np.linspace(0, max_pred * 0.5, 8)
        if "distance" in hyperparams_tune:
            hyperparam_dict["distance"] = np.geomspace(
                1, 1000 * max_distance, 8
            ).astype(int)
        if dataclass.event_type == "interval":
            hyperparam_dict["alternating"] = [False, True] if tune_alternating else [False]

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

                if name not in best_params:
                    best_params[name] = (score, param, scores)
                    continue

                best_score, best_param, best_scores = best_params[name]
                if best_score < score:
                    best_param = param
                    best_score = score
                    best_scores = scores

                best_params[name] = (best_score, best_param, best_scores)

        # best_scores = {n: s for n, (s, p) in best_params.items()}
        for metric in evaluation_metrics:
            if metric not in best_params:
                best_params[metric] = (
                    default_scores.get(metric, 0),
                    {},
                    default_scores,
                )
            print(f" optimizing hyperparams for {metric}:")
            best_score, best_param, best_scores = best_params[metric]
            print(f"  best params: {format_score_output(best_param)}")
            print(
                f"  best scores: {format_score_output({k:best_scores[k] for k in evaluation_metrics})}"
            )
            result_obj["optimized"][metric] = {
                "best_score": serializable_score(best_score),
                "best_params": score_dict_for_json(best_param),
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
                            score_dict_for_json(best_param),
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
                            best_param, tol
                        )
                        for tol in dataclass.tolerances
                    )
                else:
                    tol_scores = [
                        get_scores_tolerance(best_param, tol)
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
                            score_dict_for_json(best_param),
                            sort_keys=True,
                        ),
                    }
                )
            result_obj["optimized"][metric]["tolerance_scores"] = tolerance_records
        result_payload["results"].append(result_obj)

    result_payload["metadata"]["score_runtime_sec"] = time.perf_counter() - start_time
    results_dir = save_pred_dir.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    scores_csv, scores_json = score_filenames(score_suffix)
    with open(results_dir / scores_json, "w", encoding="utf-8") as f:
        json.dump(result_payload, f, indent=2)
    pd.DataFrame(rows).to_csv(results_dir / scores_csv, index=False)
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
    run_tag: str = None,
    tune_cutoff_steps: int = 11,
    tune_cutoff_values=None,
    tune_smooth_values=None,
    tune_alternating: bool = True,
    wandb_enabled: bool = False,
    wandb_project: str = "event-detection-pdf",
    wandb_entity: str = None,
    wandb_group: str = None,
    wandb_name: str = None,
    wandb_mode: str = None,
    wandb_tags: str = None,
    wandb_log_artifacts: bool = True,
    score_suffix: str = None,
):
    dataset_name = dataclass.name
    dataset_construct = dataclass.dataset_construct

    if str(model_name).startswith("transformer") and workers > 1:
        print(
            "Reducing Transformer eval tuning workers to 1 to avoid "
            "parallel scoring memory spikes."
        )
        workers = 1

    save_model_path = experiment_path(dataset_name, model_name, objective, seed, run_tag)
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
        normalize=normalize,
        use_cat=use_cat,
    )
    metadata = {
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
        "workers": workers,
        "run_tag": run_tag,
        "tune_cutoff_steps": tune_cutoff_steps,
        "tune_cutoff_values": tune_cutoff_values,
        "tune_smooth_values": tune_smooth_values,
        "tune_alternating": tune_alternating,
        "wandb_enabled": wandb_enabled,
        "wandb_project": wandb_project,
        "wandb_entity": wandb_entity,
        "wandb_group": wandb_group,
        "wandb_name": wandb_name,
        "wandb_mode": wandb_mode,
        "wandb_tags": wandb_tags,
        "wandb_log_artifacts": wandb_log_artifacts,
        "score_tolerances": list(dataclass.tolerances),
        "score_suffix": score_suffix or "",
    }
    wandb_run = init_wandb_run(
        wandb_enabled,
        metadata,
        project=wandb_project,
        entity=wandb_entity,
        group=wandb_group,
        name=wandb_name,
        tags=wandb_tags,
        mode=wandb_mode,
        job_type="eval",
    )
    score_payload = get_optimal_cutoff(
        dataclass,
        model_name,
        objective,
        full_dataset,
        save_pred_dir,
        workers=workers,
        metadata=metadata,
        tune_cutoff_steps=tune_cutoff_steps,
        tune_cutoff_values=tune_cutoff_values,
        tune_smooth_values=tune_smooth_values,
        tune_alternating=tune_alternating,
        score_suffix=score_suffix,
    )
    log_score_payload_to_wandb(wandb_run, score_payload)
    if wandb_log_artifacts:
        results_dir = save_pred_dir.parent / "results"
        scores_csv, scores_json = score_filenames(score_suffix)
        log_wandb_artifact(
            wandb_run,
            f"{dataset_name}-{model_name}-{objective}-seed{seed}-{run_tag or 'default'}-scores",
            "evaluation-results",
            [results_dir / scores_csv, results_dir / scores_json],
        )
    if wandb_run is not None:
        wandb_run.finish()


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
    parser.add_argument(
        "--run_tag",
        default=None,
        type=str,
        help="Optional suffix for the seed directory to evaluate a tagged run.",
    )
    parser.add_argument("--tune_cutoff_steps", default=11, type=int)
    parser.add_argument("--tune_cutoff_values", default=None, type=str)
    parser.add_argument("--tune_smooth_values", default="none,1,10,100,1000", type=str)
    parser.add_argument("--tune_alternating", default=True, type=str2bool)
    parser.add_argument(
        "--score_tolerances",
        default=None,
        type=str,
        help="Optional comma-separated scoring tolerances in dataset timesteps.",
    )
    parser.add_argument(
        "--score_suffix",
        default=None,
        type=str,
        help="Optional suffix for non-destructive score files, e.g. strict3.",
    )
    parser.add_argument("--wandb", default=False, type=str2bool)
    parser.add_argument("--wandb_project", default="event-detection-pdf", type=str)
    parser.add_argument("--wandb_entity", default=None, type=str)
    parser.add_argument("--wandb_group", default=None, type=str)
    parser.add_argument("--wandb_name", default=None, type=str)
    parser.add_argument(
        "--wandb_mode",
        default=None,
        choices=[None, "online", "offline", "disabled"],
    )
    parser.add_argument("--wandb_tags", default=None, type=str)
    parser.add_argument("--wandb_log_artifacts", default=True, type=str2bool)
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
    score_tolerances = parse_tolerance_values(args.score_tolerances)
    if score_tolerances is not None:
        dataclass.tolerances = score_tolerances

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
        run_tag=args.run_tag,
        tune_cutoff_steps=args.tune_cutoff_steps,
        tune_cutoff_values=args.tune_cutoff_values,
        tune_smooth_values=args.tune_smooth_values,
        tune_alternating=args.tune_alternating,
        wandb_enabled=args.wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_group=args.wandb_group,
        wandb_name=args.wandb_name,
        wandb_mode=args.wandb_mode,
        wandb_tags=args.wandb_tags,
        wandb_log_artifacts=args.wandb_log_artifacts,
        score_suffix=args.score_suffix,
    )
