"""Score state reconstruction from saved event-detection predictions.

This is an offline cross-view ablation. BDL models are trained for
ranked event localization, but interval tasks also induce a binary state
sequence. This script converts saved validation predictions into state masks and
reports ordinary segmentation metrics such as per-bin F1 and IoU.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.seizure import get_seizure_dataclass
from src.sleep import get_sleep_dataclass
from src.utils import (
    apply_objective_overrides,
    is_density_objective,
    is_segmentation_objective,
)


OBJECTIVE_LABELS = {
    "density_hard": "BDL-Hard",
    "density_gau": "BDL-Gaussian",
    "density_custom": "BDL-Tolerance",
    "seg": "Cross-entropy",
    "seg_weighted": "Weighted cross-entropy",
    "seg_focal": "Focal loss",
}

METHOD_LABELS = {
    "direct": "Direct state probability",
    "peak_fill": "Event-score peaks to intervals",
    "integral": "Integrated event-score difference",
    "hazard": "Hazard-recursion state probability",
}

OBJECTIVE_ORDER = [
    "density_hard",
    "density_gau",
    "density_custom",
    "seg",
    "seg_weighted",
    "seg_focal",
]

METHOD_ORDER = [
    "hazard",
    "peak_fill",
    "integral",
    "direct",
]


@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    pred_transitions: int = 0
    true_transitions: int = 0
    series: int = 0
    bins: int = 0

    def update(self, pred: np.ndarray, truth: np.ndarray) -> None:
        n = min(len(pred), len(truth))
        if n <= 0:
            return
        pred = np.asarray(pred[:n], dtype=bool)
        truth = np.asarray(truth[:n], dtype=bool)
        self.tp += int(np.sum(pred & truth))
        self.fp += int(np.sum(pred & ~truth))
        self.fn += int(np.sum(~pred & truth))
        self.tn += int(np.sum(~pred & ~truth))
        self.pred_transitions += int(np.sum(pred[1:] != pred[:-1]))
        self.true_transitions += int(np.sum(truth[1:] != truth[:-1]))
        self.series += 1
        self.bins += n


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def metrics_from_counts(counts: Counts) -> dict:
    precision = safe_div(counts.tp, counts.tp + counts.fp)
    recall = safe_div(counts.tp, counts.tp + counts.fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    iou = safe_div(counts.tp, counts.tp + counts.fp + counts.fn)

    neg_precision = safe_div(counts.tn, counts.tn + counts.fn)
    neg_recall = safe_div(counts.tn, counts.tn + counts.fp)
    neg_f1 = safe_div(2 * neg_precision * neg_recall, neg_precision + neg_recall)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_f1": 0.5 * (f1 + neg_f1),
        "iou": iou,
        "accuracy": safe_div(counts.tp + counts.tn, counts.bins),
        "brier": safe_div(counts.fp + counts.fn, counts.bins),
        "pred_positive_rate": safe_div(counts.tp + counts.fp, counts.bins),
        "true_positive_rate": safe_div(counts.tp + counts.fn, counts.bins),
        "transition_count_error": counts.pred_transitions - counts.true_transitions,
        "abs_transition_count_error": abs(
            counts.pred_transitions - counts.true_transitions
        ),
        "series": counts.series,
        "bins": counts.bins,
    }


def parse_csv(value: str | None) -> set[str] | None:
    if value is None or value == "":
        return None
    return {token.strip() for token in value.split(",") if token.strip()}


def parse_int_csv(value: str | None) -> set[int] | None:
    if value is None or value == "":
        return None
    return {int(token.strip()) for token in value.split(",") if token.strip()}


def parse_smooth_values(value: str) -> list[int | None]:
    out: list[int | None] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if token.lower() in {"none", "null", "false"}:
            out.append(None)
        else:
            out.append(int(float(token)))
    return out or [None]


def parse_float_values(value: str) -> list[float]:
    out = []
    for token in value.split(","):
        token = token.strip()
        if token:
            out.append(float(token))
    return out or [1.0]


def parse_initial_probs(value: str, prior: float) -> list[float]:
    out = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if token.lower() == "prior":
            out.append(float(prior))
        else:
            out.append(float(token))
    return sorted(set(float(np.clip(prob, 0.0, 1.0)) for prob in out))


def load_dataclass(name: str):
    if name == "sleep":
        return get_sleep_dataclass()
    if name == "seizure":
        return get_seizure_dataclass()
    raise ValueError(f"Segmentation reconstruction is only implemented for interval datasets, got {name!r}.")


def data_dir_for_dataset(args, dataset: str) -> Path:
    if dataset == "sleep":
        return Path(args.sleep_data)
    if dataset == "seizure":
        return Path(args.seizure_data)
    raise ValueError(f"No data directory argument is defined for {dataset!r}.")


def iter_run_configs(args):
    datasets = parse_csv(args.datasets)
    models = parse_csv(args.models)
    objectives = parse_csv(args.objectives)
    run_tags = parse_csv(args.run_tags)
    seeds = parse_int_csv(args.seeds)

    for config_path in sorted(Path(args.results_root).glob("**/results/run_config.json")):
        run_dir = config_path.parent.parent
        pred_dir = run_dir / "predictions"
        if not pred_dir.exists():
            continue
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        if datasets and config.get("dataset") not in datasets:
            continue
        if models and config.get("model") not in models:
            continue
        if objectives and config.get("objective") not in objectives:
            continue
        if run_tags and str(config.get("run_tag", "")) not in run_tags:
            continue
        if seeds and int(config.get("seed", -1)) not in seeds:
            continue
        yield config_path, run_dir, pred_dir, config


def load_eval_dataset(args, config: dict, cache: dict):
    key = (
        config["dataset"],
        config.get("downsample", 1),
        config.get("agg_feats", "stat"),
        config.get("sequence_length"),
        bool(config.get("normalize", True)),
        bool(config.get("use_cat", True)),
        config.get("gaussian_sigma"),
    )
    if key in cache:
        return cache[key]

    dataclass = load_dataclass(config["dataset"])
    dataclass = apply_objective_overrides(
        dataclass,
        gaussian_sigma=config.get("gaussian_sigma"),
    )
    dataset = dataclass.dataset_construct(
        dataclass,
        data_dir_for_dataset(args, config["dataset"]),
        -1,
        training=False,
        downsample=int(config.get("downsample", 1)),
        agg_feats=config.get("agg_feats", "stat"),
        sequence_length=config.get("sequence_length") or dataclass.default_sequence_length,
        target_type="seg",
        normalize=bool(config.get("normalize", True)),
        use_cat=bool(config.get("use_cat", True)),
    )
    cache[key] = (dataclass, dataset)
    return cache[key]


def series_length(dataset, series_id: str) -> int:
    if hasattr(dataset, "data"):
        return len(dataset.data[series_id])
    if hasattr(dataset, "ds_dir"):
        filepath = series_id.split(".")[0] + ".ftr"
        return len(pd.read_feather(dataset.ds_dir / filepath, columns=["series_id"]))
    raise ValueError("Dataset does not expose a known way to compute series length.")


def truth_mask(dataset, dataclass, series_id: str, downsample: int) -> np.ndarray:
    if dataclass.event_type != "interval":
        raise ValueError("State reconstruction metrics require interval targets.")

    length = series_length(dataset, series_id)
    n_bins = length // downsample
    mask = np.zeros(n_bins, dtype=bool)
    starts, ends = dataset.targets[series_id]
    for start, end in zip(starts, ends):
        i1 = max(0, int(start) // downsample)
        i2 = min(n_bins, (int(end) + downsample - 1) // downsample)
        if i2 > i1:
            mask[i1:i2] = True
    return mask


def load_predictions(pred_dir: Path) -> dict[str, np.ndarray]:
    predictions = {}
    for path in sorted(pred_dir.glob("*.npy")):
        pred = np.load(path)
        if pred.ndim == 1:
            pred = pred[:, None]
        predictions[path.stem] = pred.astype(np.float32, copy=False)
    return predictions


def maybe_smooth(pred: np.ndarray, smooth: int | None) -> np.ndarray:
    if smooth is None or smooth <= 0:
        return pred
    out = pred.copy()
    for channel in range(out.shape[1]):
        out[:, channel] = gaussian_filter1d(out[:, channel], smooth)
    return out


def alternating_events(events: list[tuple[int, int, float]]) -> list[tuple[int, int, float]]:
    events = sorted(events, key=lambda item: (item[0], item[1]))
    selected: list[tuple[int, int, float]] = []
    current = None
    for event in events:
        if current is None:
            current = event
            continue
        if event[1] == current[1]:
            if event[2] > current[2]:
                current = event
        else:
            selected.append(current)
            current = event
    if current is not None:
        selected.append(current)
    return selected


def events_to_mask(events: list[tuple[int, int, float]], length: int) -> np.ndarray:
    mask = np.zeros(length, dtype=bool)
    active = False
    start = 0
    for loc, channel, _ in sorted(events, key=lambda item: (item[0], item[1])):
        loc = min(max(int(loc), 0), length)
        if channel == 0:
            if not active:
                active = True
                start = loc
        else:
            if active:
                if loc > start:
                    mask[start:loc] = True
                active = False
    if active and start < length:
        mask[start:length] = True
    return mask


def peak_fill_mask(
    pred: np.ndarray,
    *,
    cutoff: float,
    smooth: int | None,
    alternating: bool,
    distance: int,
) -> np.ndarray:
    pred = maybe_smooth(pred, smooth)
    events: list[tuple[int, int, float]] = []
    for channel in range(min(2, pred.shape[1])):
        peaks = find_peaks(
            pred[:, channel],
            height=cutoff,
            distance=max(1, int(distance)),
        )[0]
        for loc in peaks:
            events.append((int(loc), channel, float(pred[loc, channel])))
    if alternating:
        events = alternating_events(events)
    return events_to_mask(events, len(pred))


def integral_scores(pred: np.ndarray, smooth: int | None) -> np.ndarray:
    pred = maybe_smooth(pred, smooth)
    if pred.shape[1] < 2:
        return pred[:, 0]
    return np.cumsum(pred[:, 0] - pred[:, 1])


@njit(cache=True)
def hazard_probabilities(
    onset_rate: np.ndarray,
    offset_rate: np.ndarray,
    scale: float,
    initial_prob: float,
) -> np.ndarray:
    probs = np.empty(onset_rate.shape[0], dtype=np.float32)
    prev = initial_prob
    for i in range(onset_rate.shape[0]):
        on_rate = onset_rate[i] * scale
        off_rate = offset_rate[i] * scale
        if on_rate < 0.0:
            on_rate = 0.0
        if off_rate < 0.0:
            off_rate = 0.0
        alpha = 1.0 - np.exp(-on_rate)
        beta = 1.0 - np.exp(-off_rate)
        prev = prev * (1.0 - beta) + (1.0 - prev) * alpha
        if prev < 0.0:
            prev = 0.0
        elif prev > 1.0:
            prev = 1.0
        probs[i] = prev
    return probs


@njit(cache=True)
def threshold_count(thresholds: np.ndarray, value: float) -> int:
    lo = 0
    hi = thresholds.shape[0]
    while lo < hi:
        mid = (lo + hi) // 2
        if value >= thresholds[mid]:
            lo = mid + 1
        else:
            hi = mid
    return lo


@njit(cache=True)
def hazard_grid_counts_series(
    onset_rate: np.ndarray,
    offset_rate: np.ndarray,
    truth: np.ndarray,
    scales: np.ndarray,
    initial_probs: np.ndarray,
    thresholds: np.ndarray,
):
    n = min(onset_rate.shape[0], truth.shape[0])
    n_scales = scales.shape[0]
    n_initials = initial_probs.shape[0]
    n_thresholds = thresholds.shape[0]

    pos_hist = np.zeros((n_scales, n_initials, n_thresholds + 1), dtype=np.int64)
    neg_hist = np.zeros((n_scales, n_initials, n_thresholds + 1), dtype=np.int64)
    trans_diff = np.zeros((n_scales, n_initials, n_thresholds + 1), dtype=np.int64)
    brier_sum = np.zeros((n_scales, n_initials), dtype=np.float64)
    prev_k = np.zeros((n_scales, n_initials), dtype=np.int64)
    probs = np.empty(n_initials, dtype=np.float64)

    true_transitions = 0
    for t in range(1, n):
        if truth[t] != truth[t - 1]:
            true_transitions += 1

    for s_idx in range(n_scales):
        scale = scales[s_idx]
        for i_idx in range(n_initials):
            probs[i_idx] = initial_probs[i_idx]

        for t in range(n):
            on_rate = onset_rate[t] * scale
            off_rate = offset_rate[t] * scale
            if on_rate < 0.0:
                on_rate = 0.0
            if off_rate < 0.0:
                off_rate = 0.0
            alpha = 1.0 - np.exp(-on_rate)
            beta = 1.0 - np.exp(-off_rate)
            y = truth[t]

            for i_idx in range(n_initials):
                p = probs[i_idx]
                p = p * (1.0 - beta) + (1.0 - p) * alpha
                if p < 0.0:
                    p = 0.0
                elif p > 1.0:
                    p = 1.0
                probs[i_idx] = p

                k = threshold_count(thresholds, p)
                if y:
                    pos_hist[s_idx, i_idx, k] += 1
                    brier_sum[s_idx, i_idx] += (p - 1.0) * (p - 1.0)
                else:
                    neg_hist[s_idx, i_idx, k] += 1
                    brier_sum[s_idx, i_idx] += p * p

                if t > 0:
                    old_k = prev_k[s_idx, i_idx]
                    if old_k < k:
                        trans_diff[s_idx, i_idx, old_k] += 1
                        trans_diff[s_idx, i_idx, k] -= 1
                    elif k < old_k:
                        trans_diff[s_idx, i_idx, k] += 1
                        trans_diff[s_idx, i_idx, old_k] -= 1
                prev_k[s_idx, i_idx] = k

    counts = np.zeros((n_scales, n_initials, n_thresholds, 5), dtype=np.int64)
    for s_idx in range(n_scales):
        for i_idx in range(n_initials):
            pos_total = 0
            neg_total = 0
            for k in range(n_thresholds + 1):
                pos_total += pos_hist[s_idx, i_idx, k]
                neg_total += neg_hist[s_idx, i_idx, k]

            pos_running = 0
            neg_running = 0
            for t_idx in range(n_thresholds - 1, -1, -1):
                pos_running += pos_hist[s_idx, i_idx, t_idx + 1]
                neg_running += neg_hist[s_idx, i_idx, t_idx + 1]
                counts[s_idx, i_idx, t_idx, 0] = pos_running
                counts[s_idx, i_idx, t_idx, 1] = neg_running
                counts[s_idx, i_idx, t_idx, 2] = pos_total - pos_running
                counts[s_idx, i_idx, t_idx, 3] = neg_total - neg_running

            trans_running = 0
            for t_idx in range(n_thresholds):
                trans_running += trans_diff[s_idx, i_idx, t_idx]
                counts[s_idx, i_idx, t_idx, 4] = trans_running

    return counts, brier_sum, true_transitions, n


def evaluate_masks(mask_by_id: dict[str, np.ndarray], truth_by_id: dict[str, np.ndarray]) -> dict:
    counts = Counts()
    for series_id, pred_mask in mask_by_id.items():
        truth = truth_by_id.get(series_id)
        if truth is None:
            continue
        counts.update(pred_mask, truth)
    return metrics_from_counts(counts)


def evaluate_mask_fn(series_ids, truth_by_id, mask_fn) -> dict:
    counts = Counts()
    for series_id in series_ids:
        truth = truth_by_id.get(series_id)
        if truth is None:
            continue
        counts.update(mask_fn(series_id), truth)
    return metrics_from_counts(counts)


def positive_prior(truth_by_id: dict[str, np.ndarray]) -> float:
    positives = 0
    total = 0
    for truth in truth_by_id.values():
        positives += int(np.sum(truth))
        total += int(len(truth))
    return safe_div(positives, total)


def brier_from_probs(
    prob_by_id: dict[str, np.ndarray],
    truth_by_id: dict[str, np.ndarray],
) -> float:
    total = 0.0
    n_total = 0
    for series_id, prob in prob_by_id.items():
        truth = truth_by_id.get(series_id)
        if truth is None:
            continue
        n = min(len(prob), len(truth))
        if n <= 0:
            continue
        p = np.clip(prob[:n].astype(np.float64, copy=False), 0.0, 1.0)
        y = truth[:n].astype(np.float64, copy=False)
        total += float(np.sum((p - y) ** 2))
        n_total += n
    return safe_div(total, n_total)


def metrics_from_raw_counts(
    tp: int,
    fp: int,
    fn: int,
    tn: int,
    pred_transitions: int,
    true_transitions: int,
    series: int,
    bins: int,
    brier_sum: float | None = None,
) -> dict:
    counts = Counts(
        tp=int(tp),
        fp=int(fp),
        fn=int(fn),
        tn=int(tn),
        pred_transitions=int(pred_transitions),
        true_transitions=int(true_transitions),
        series=int(series),
        bins=int(bins),
    )
    scores = metrics_from_counts(counts)
    if brier_sum is not None:
        scores["brier"] = safe_div(float(brier_sum), bins)
    return scores


def score_direct(predictions, truth_by_id, cutoff_steps: int):
    best = None
    prob_by_id = {
        series_id: np.clip(pred[:, 0], 0.0, 1.0)
        for series_id, pred in predictions.items()
        if pred.shape[1] >= 1
    }
    brier = brier_from_probs(prob_by_id, truth_by_id)
    for threshold in np.linspace(0.0, 1.0, cutoff_steps):
        masks = {
            series_id: prob >= threshold
            for series_id, prob in prob_by_id.items()
        }
        scores = evaluate_masks(masks, truth_by_id)
        scores["brier"] = brier
        scores["params"] = json.dumps({"threshold": float(threshold)}, sort_keys=True)
        if best is None or scores["f1"] > best["f1"]:
            best = scores
    return best


def score_peak_fill(predictions, truth_by_id, dataclass, downsample: int, cutoff_steps: int, smooth_values):
    max_pred = max(float(np.nanmax(pred)) for pred in predictions.values() if pred.size)
    cutoffs = np.linspace(0.0, max(max_pred, 1e-8), cutoff_steps)
    distance = max(1, int(dataclass.max_distance // downsample))
    series_ids = list(predictions.keys())
    best = None

    for smooth in smooth_values:
        events_by_id = {}
        for series_id, pred in predictions.items():
            pred_smooth = maybe_smooth(pred, smooth)
            events = []
            for channel in range(min(2, pred_smooth.shape[1])):
                peaks = find_peaks(
                    pred_smooth[:, channel],
                    distance=distance,
                )[0]
                for loc in peaks:
                    events.append(
                        (int(loc), channel, float(pred_smooth[loc, channel]))
                    )
            events_by_id[series_id] = events

        for cutoff in cutoffs:
            for alternating in [False, True]:
                def make_mask(series_id, cutoff=float(cutoff), alternating=alternating):
                    events = [
                        event
                        for event in events_by_id[series_id]
                        if event[2] >= cutoff
                    ]
                    if alternating:
                        events = alternating_events(events)
                    return events_to_mask(events, len(predictions[series_id]))

                scores = evaluate_mask_fn(series_ids, truth_by_id, make_mask)
                scores["params"] = json.dumps(
                    {
                        "cutoff": float(cutoff),
                        "smooth": "none" if smooth is None else int(smooth),
                        "alternating": bool(alternating),
                        "distance": distance,
                    },
                    sort_keys=True,
                )
                if best is None or scores["f1"] > best["f1"]:
                    best = scores
    return best


def score_integral(predictions, truth_by_id, cutoff_steps: int, smooth_values):
    best = None
    for smooth in smooth_values:
        scores_by_id = {
            series_id: integral_scores(pred, smooth)
            for series_id, pred in predictions.items()
        }
        finite_scores = np.concatenate(
            [score[np.isfinite(score)] for score in scores_by_id.values() if score.size]
        )
        if finite_scores.size == 0:
            continue
        lo, hi = float(np.min(finite_scores)), float(np.max(finite_scores))
        thresholds = np.linspace(lo, hi, cutoff_steps) if lo < hi else np.array([lo])
        for threshold in thresholds:
            masks = {
                series_id: score >= threshold
                for series_id, score in scores_by_id.items()
            }
            scores = evaluate_masks(masks, truth_by_id)
            scores["params"] = json.dumps(
                {
                    "threshold": float(threshold),
                    "smooth": "none" if smooth is None else int(smooth),
                },
                sort_keys=True,
            )
            if best is None or scores["f1"] > best["f1"]:
                best = scores
    return best


def score_hazard(
    predictions,
    truth_by_id,
    cutoff_steps: int,
    smooth_values,
    hazard_scales,
    initial_probs,
    collect_grid: bool = False,
):
    best = None
    grid_rows = []
    thresholds = np.linspace(0.0, 1.0, cutoff_steps).astype(np.float64)
    scale_array = np.asarray(hazard_scales, dtype=np.float64)
    initial_array = np.asarray(initial_probs, dtype=np.float64)

    for smooth in smooth_values:
        total_counts = np.zeros(
            (
                len(scale_array),
                len(initial_array),
                len(thresholds),
                5,
            ),
            dtype=np.int64,
        )
        total_brier = np.zeros((len(scale_array), len(initial_array)), dtype=np.float64)
        total_true_transitions = 0
        total_bins = 0
        total_series = 0

        for series_id, pred in predictions.items():
            if pred.shape[1] < 2 or series_id not in truth_by_id:
                continue
            pred = maybe_smooth(pred, smooth)
            counts, brier_sum, true_transitions, n = hazard_grid_counts_series(
                pred[:, 0].astype(np.float64, copy=False),
                pred[:, 1].astype(np.float64, copy=False),
                truth_by_id[series_id].astype(np.bool_, copy=False),
                scale_array,
                initial_array,
                thresholds,
            )
            total_counts += counts
            total_brier += brier_sum
            total_true_transitions += int(true_transitions)
            total_bins += int(n)
            total_series += 1

        for scale_idx, scale in enumerate(scale_array):
            for initial_idx, initial_prob in enumerate(initial_array):
                brier_sum = total_brier[scale_idx, initial_idx]
                for threshold_idx, threshold in enumerate(thresholds):
                    raw = total_counts[scale_idx, initial_idx, threshold_idx]
                    scores = metrics_from_raw_counts(
                        raw[0],
                        raw[1],
                        raw[2],
                        raw[3],
                        raw[4],
                        total_true_transitions,
                        total_series,
                        total_bins,
                        brier_sum=brier_sum,
                    )
                    params = {
                        "threshold": float(threshold),
                        "smooth": "none" if smooth is None else int(smooth),
                        "scale": float(scale),
                        "initial_prob": float(initial_prob),
                    }
                    scores["params"] = json.dumps(params, sort_keys=True)
                    if collect_grid:
                        grid_rows.append({**params, **scores})
                    if best is None or scores["f1"] > best["f1"]:
                        best = scores
    return best, grid_rows


def summarize_hazard_grid(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows
    metric_cols = [
        "precision",
        "recall",
        "f1",
        "macro_f1",
        "iou",
        "accuracy",
        "brier",
        "pred_positive_rate",
        "abs_transition_count_error",
    ]
    grouped = (
        rows.groupby(
            [
                "dataset",
                "model",
                "objective",
                "run_tag",
                "smooth",
                "scale",
                "initial_prob",
                "threshold",
            ],
            dropna=False,
        )[metric_cols]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    grouped.columns = [
        "_".join([str(part) for part in col if part])
        if isinstance(col, tuple)
        else col
        for col in grouped.columns
    ]
    grouped["objective_label"] = grouped["objective"].map(OBJECTIVE_LABELS).fillna(
        grouped["objective"]
    )
    return grouped.sort_values(
        ["dataset", "model", "run_tag", "objective", "f1_mean"],
        ascending=[True, True, True, True, False],
    )


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows
    metric_cols = [
        "precision",
        "recall",
        "f1",
        "macro_f1",
        "iou",
        "accuracy",
        "brier",
        "pred_positive_rate",
        "true_positive_rate",
        "transition_count_error",
        "abs_transition_count_error",
    ]
    grouped = (
        rows.groupby(["dataset", "model", "objective", "run_tag", "method"], dropna=False)[
            metric_cols
        ]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    flat_cols = []
    for col in grouped.columns:
        if isinstance(col, tuple):
            flat_cols.append("_".join([str(part) for part in col if part]))
        else:
            flat_cols.append(col)
    grouped.columns = flat_cols
    grouped["objective_label"] = grouped["objective"].map(OBJECTIVE_LABELS).fillna(
        grouped["objective"]
    )
    grouped["method_label"] = grouped["method"].map(METHOD_LABELS).fillna(
        grouped["method"]
    )
    return grouped.sort_values(["dataset", "model", "run_tag", "objective", "method"])


def format_mean_std(mean, std, count) -> str:
    if pd.isna(mean):
        return "--"
    return f"{mean:.3f}"


def write_summary_table(summary: pd.DataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    if summary.empty:
        (outdir / "segmentation_reconstruction.tex").write_text(
            "% No segmentation reconstruction results available yet.\n",
            encoding="utf-8",
        )
        return

    table = summary[
        (summary["dataset"] == "sleep")
        & (summary["model"] == "gru")
        & (summary["run_tag"] == "objective_e20_bs32_eval5")
    ].copy()
    if table.empty:
        table = summary.copy()
    table["objective_order"] = table["objective"].map(
        {name: index for index, name in enumerate(OBJECTIVE_ORDER)}
    )
    table["method_order"] = table["method"].map(
        {name: index for index, name in enumerate(METHOD_ORDER)}
    )
    table = table.sort_values(
        ["objective_order", "method_order", "objective", "method"],
        kind="stable",
    )

    rows = []
    for row in table.itertuples(index=False):
        seeds = int(getattr(row, "f1_count"))
        rows.append(
            {
                "Method": getattr(row, "objective_label"),
                "Reconstruction": getattr(row, "method_label"),
                "F1": format_mean_std(
                    getattr(row, "f1_mean"), getattr(row, "f1_std"), seeds
                ),
                "IoU": format_mean_std(
                    getattr(row, "iou_mean"), getattr(row, "iou_std"), seeds
                ),
                "Brier": format_mean_std(
                    getattr(row, "brier_mean"), getattr(row, "brier_std"), seeds
                ),
                "Precision": format_mean_std(
                    getattr(row, "precision_mean"),
                    getattr(row, "precision_std"),
                    seeds,
                ),
                "Recall": format_mean_std(
                    getattr(row, "recall_mean"), getattr(row, "recall_std"), seeds
                ),
            }
        )
    pd.DataFrame(rows).to_latex(
        outdir / "segmentation_reconstruction.tex",
        index=False,
        escape=False,
    )


def main() -> None:
    args = parse_args()
    smooth_values = parse_smooth_values(args.smooth_values)
    hazard_scales = parse_float_values(args.hazard_scales)
    dataset_cache = {}
    truth_cache = {}
    rows = []
    hazard_grid_rows = []

    for _, run_dir, pred_dir, config in iter_run_configs(args):
        objective = config["objective"]
        if not (is_density_objective(objective) or is_segmentation_objective(objective)):
            continue

        dataclass, dataset = load_eval_dataset(args, config, dataset_cache)
        downsample = int(config.get("downsample", 1))
        predictions = load_predictions(pred_dir)
        if not predictions:
            continue

        print(
            "Scoring "
            f"{config['dataset']}/{config['model']}/{objective}/"
            f"seed={config.get('seed')} tag={config.get('run_tag')}",
            flush=True,
        )
        truth_by_id = {}
        for series_id in predictions.keys():
            if series_id not in dataset.targets:
                continue
            truth_key = (config["dataset"], downsample, series_id)
            if truth_key not in truth_cache:
                truth_cache[truth_key] = truth_mask(
                    dataset, dataclass, series_id, downsample
                )
            truth_by_id[series_id] = truth_cache[truth_key]
        predictions = {
            series_id: pred[: len(truth_by_id[series_id])]
            for series_id, pred in predictions.items()
            if series_id in truth_by_id
        }
        if not predictions:
            continue
        initial_probs = parse_initial_probs(
            args.hazard_initial_probs,
            positive_prior(truth_by_id),
        )

        methods = []
        if is_segmentation_objective(objective):
            methods.append(("direct", score_direct(predictions, truth_by_id, args.cutoff_steps)))
        else:
            methods.append(
                (
                    "peak_fill",
                    score_peak_fill(
                        predictions,
                        truth_by_id,
                        dataclass,
                        downsample,
                        args.cutoff_steps,
                        smooth_values,
                    ),
                )
            )
            methods.append(
                (
                    "integral",
                    score_integral(
                        predictions,
                        truth_by_id,
                        args.cutoff_steps,
                        smooth_values,
                    ),
                )
            )
            hazard_scores, hazard_rows = score_hazard(
                predictions,
                truth_by_id,
                args.cutoff_steps,
                smooth_values,
                hazard_scales,
                initial_probs,
                collect_grid=args.write_hazard_grid,
            )
            methods.append(("hazard", hazard_scores))
            if args.write_hazard_grid:
                hazard_grid_rows.extend(
                    {
                        "dataset": config["dataset"],
                        "model": config["model"],
                        "objective": objective,
                        "seed": config.get("seed"),
                        "run_tag": str(config.get("run_tag", "")),
                        "run_dir": str(run_dir),
                        **record,
                    }
                    for record in hazard_rows
                )

        for method, scores in methods:
            if not scores:
                continue
            rows.append(
                {
                    "dataset": config["dataset"],
                    "model": config["model"],
                    "objective": objective,
                    "seed": config.get("seed"),
                    "run_tag": str(config.get("run_tag", "")),
                    "method": method,
                    "run_dir": str(run_dir),
                    **scores,
                }
            )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows_df = pd.DataFrame(rows)
    rows_df.to_csv(outdir / "segmentation_reconstruction_runs.csv", index=False)
    summary = summarize(rows_df)
    summary.to_csv(outdir / "segmentation_reconstruction.csv", index=False)
    hazard_grid_df = pd.DataFrame(hazard_grid_rows)
    hazard_grid_df.to_csv(outdir / "segmentation_hazard_grid_runs.csv", index=False)
    summarize_hazard_grid(hazard_grid_df).to_csv(
        outdir / "segmentation_hazard_grid.csv",
        index=False,
    )
    write_summary_table(summary, outdir)
    print(f"Scored {len(rows_df)} reconstruction rows.")
    if len(hazard_grid_df) > 0:
        print(f"Wrote {len(hazard_grid_df)} hazard calibration grid rows.")
    print(f"Wrote segmentation reconstruction outputs to {outdir}.")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default="experiments")
    parser.add_argument("--outdir", default="paper/results/generated")
    parser.add_argument("--sleep-data", default="data/sleep")
    parser.add_argument("--seizure-data", default="data/seizure")
    parser.add_argument("--datasets", default="sleep")
    parser.add_argument("--models", default="gru")
    parser.add_argument(
        "--objectives",
        default="density_hard,density_gau,density_custom,seg,seg_weighted,seg_focal",
    )
    parser.add_argument("--run-tags", default="objective_e20_bs32_eval5")
    parser.add_argument(
        "--seeds",
        default="",
        help="Comma-separated seeds to report. Empty aggregates all available saved runs.",
    )
    parser.add_argument("--cutoff-steps", type=int, default=7)
    parser.add_argument(
        "--smooth-values",
        default="none",
        help="Comma-separated Gaussian smoothing widths in output bins. Wider grids are slower.",
    )
    parser.add_argument(
        "--hazard-scales",
        default="0.25,1,4",
        help="Comma-separated multipliers applied before converting BDL outputs to hazards.",
    )
    parser.add_argument(
        "--hazard-initial-probs",
        default="0,prior,1",
        help="Comma-separated initial state probabilities; `prior` uses the validation positive rate.",
    )
    parser.add_argument(
        "--write-hazard-grid",
        default=True,
        type=lambda value: str(value).lower() in {"1", "true", "yes", "y"},
        help="Write per-setting hazard calibration metrics.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
