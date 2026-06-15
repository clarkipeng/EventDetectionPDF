"""Generate paper plots from structured experiment artifacts.

This script intentionally reads result files from experiments/ instead of the
legacy notebook state. It is safe to run before experiments exist; target-kernel
plots can still be produced from dataset metadata.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from src.seizure import get_seizure_dataclass
from src.sleep import get_sleep_dataclass
from src.utils import (
    DENSITY_OBJECTIVES,
    apply_objective_overrides,
    base_objective,
    get_target_distribution,
)


def get_dataclass(dataset):
    if dataset == "sleep":
        return get_sleep_dataclass()
    if dataset == "seizure":
        return get_seizure_dataclass()
    raise ValueError(f"Plotting metadata is not configured for {dataset}")


def load_scores(results_root):
    paths = sorted(Path(results_root).glob("**/results/scores.csv"))
    if not paths:
        return pd.DataFrame()
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame["score_path"] = str(path)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def plot_target_kernels(dataclass, outdir):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for objective in DENSITY_OBJECTIVES:
        values = get_target_distribution(dataclass, base_objective(objective), unit_mass=True)
        center = len(values) // 2
        x = np.arange(len(values)) - center
        ax.plot(x, values, label=objective)
    ax.set_title("Unit-mass boundary-density targets")
    ax.set_xlabel("Offset from annotated boundary")
    ax.set_ylabel("Target mass")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "target_kernels.png", dpi=200)
    plt.close(fig)


def plot_summary_bars(scores, outdir, dataset):
    if scores.empty:
        return
    frame = scores[
        (scores["row_type"] == "summary")
        & (scores["stage"] == "tuned")
        & (scores["optimized_metric"] == "mAP")
        & (scores["metric"] == "mAP")
    ].copy()
    if dataset:
        frame = frame[frame["dataset"] == dataset]
    if frame.empty:
        return
    grouped = (
        frame.groupby(["objective", "model"], as_index=False)["score"]
        .mean()
        .sort_values("score", ascending=False)
    )
    grouped["label"] = grouped["objective"] + "\n" + grouped["model"]
    fig_width = max(8, 0.45 * len(grouped))
    fig, ax = plt.subplots(figsize=(fig_width, 4))
    ax.bar(np.arange(len(grouped)), grouped["score"], color="#4C78A8")
    ax.set_xticks(np.arange(len(grouped)))
    ax.set_xticklabels(grouped["label"], rotation=60, ha="right")
    ax.set_ylabel("Tuned mAP")
    ax.set_title("Objective/model comparison")
    fig.tight_layout()
    fig.savefig(outdir / "summary_bars.png", dpi=200)
    plt.close(fig)


def plot_tolerance_curves(scores, outdir, dataset):
    if scores.empty:
        return
    frame = scores[
        (scores["row_type"] == "tolerance")
        & (scores["stage"] == "tuned")
        & (scores["optimized_metric"] == "mAP")
        & (scores["metric"] == "mAP")
    ].copy()
    if dataset:
        frame = frame[frame["dataset"] == dataset]
    if frame.empty:
        return
    frame["tolerance"] = pd.to_numeric(frame["tolerance"])
    fig, ax = plt.subplots(figsize=(7, 4))
    for objective, group in frame.groupby("objective"):
        curve = group.groupby("tolerance", as_index=False)["score"].mean()
        ax.plot(curve["tolerance"], curve["score"], marker="o", label=objective)
    ax.set_xlabel("Tolerance")
    ax.set_ylabel("mAP")
    ax.set_title("EDAP by tolerance")
    ax.legend(frameon=False, ncols=2)
    fig.tight_layout()
    fig.savefig(outdir / "tolerance_curves.png", dpi=200)
    plt.close(fig)


def plot_prediction_peaks(prediction_path, outdir, max_steps=5000):
    prediction = np.load(prediction_path)
    if prediction.ndim == 1:
        prediction = prediction[:, None]
    prediction = prediction[:max_steps]
    fig, axes = plt.subplots(prediction.shape[1], 1, figsize=(8, 2.5 * prediction.shape[1]), sharex=True)
    if prediction.shape[1] == 1:
        axes = [axes]
    for channel, ax in enumerate(axes):
        values = prediction[:, channel]
        peaks = find_peaks(values, height=0)[0]
        ax.plot(values, color="#4C78A8", linewidth=1.2)
        ax.scatter(peaks, values[peaks], color="#E45756", s=16, zorder=3)
        ax.set_ylabel(f"Channel {channel}")
    axes[-1].set_xlabel("Timestep")
    fig.suptitle(f"Predicted boundary rates: {prediction_path.name}")
    fig.tight_layout()
    fig.savefig(outdir / "prediction_peaks.png", dpi=200)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="sleep", choices=["sleep", "seizure"])
    parser.add_argument("--results-root", default=REPO_ROOT / "experiments")
    parser.add_argument("--outdir", default=REPO_ROOT / "paper" / "figures" / "generated")
    parser.add_argument("--gaussian_sigma", type=float, default=None)
    parser.add_argument("--tolerance_scale", type=float, default=1.0)
    parser.add_argument("--prediction", default=None, help="Optional .npy prediction file to plot.")
    parser.add_argument("--max-steps", type=int, default=5000)
    return parser.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    dataclass = apply_objective_overrides(
        get_dataclass(args.dataset),
        gaussian_sigma=args.gaussian_sigma,
        tolerance_scale=args.tolerance_scale,
    )
    plot_target_kernels(dataclass, outdir)

    scores = load_scores(args.results_root)
    plot_summary_bars(scores, outdir, args.dataset)
    plot_tolerance_curves(scores, outdir, args.dataset)

    if args.prediction:
        plot_prediction_peaks(Path(args.prediction), outdir, args.max_steps)


if __name__ == "__main__":
    main()
