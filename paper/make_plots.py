"""Generate paper plots from structured experiment artifacts.

This script intentionally reads result files from experiments/ instead of the
archived notebook state. It is safe to run before experiments exist;
target-kernel plots can still be produced from dataset metadata.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from matplotlib.patches import FancyBboxPatch
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


MAIN_TOLERANCE_OBJECTIVES = {
    "density_hard",
    "density_gau",
    "density_custom",
    "seg",
    "seg_weighted",
    "seg_focal",
}

OBJECTIVE_LABELS = {
    "density_hard": "BDL-Hard",
    "density_gau": "BDL-Gaussian",
    "density_custom": "BDL-Tolerance",
    "seg": "Cross-entropy",
    "seg_weighted": "Weighted CE",
    "seg_focal": "Focal loss",
}

MODEL_LABELS = {
    "gru": "GRU",
    "unet": "U-Net",
    "unet_t": "Attention-gated U-Net",
    "fgru": "Forward GRU",
    "flstm": "Forward LSTM",
    "causal_transformer": "Causal Transformer",
    "transformer_4l_64h_4a": "Offline Transformer",
    "transformer_6l_128h_8a_10d": "Offline Transformer",
}

MODEL_SHORT_LABELS = {
    "unet_t": "Attention-gated\nU-Net",
    "transformer_4l_64h_4a": "Offline Trans.",
    "transformer_6l_128h_8a_10d": "Offline\nTransformer",
    "causal_transformer": "Causal Trans.",
}

MAIN_RUN_TAG = "objective_e20_bs32_eval5"
TRANSFORMER_STRONG_CANDIDATE_TAG = "transformer_strong_lr5e4_wd1e2_e20_bs8_eval5"
SEIZURE_RUN_TAG = "seizure_main_e20_bs32_eval5"
ONLINE_RUN_TAG = "online_e20_bs32_eval5"

PALETTE = {
    "bdl": "#009E73",
    "bdl_dark": "#00664B",
    "bdl_mid": "#2A9D8F",
    "bdl_soft": "#66A182",
    "bdl_light": "#CDEFE4",
    "seg": "#D55E00",
    "seg_dark": "#8A3B00",
    "seg_mid": "#E6862A",
    "seg_soft": "#E6A000",
    "seg_light": "#F3C49B",
    "accent": "#0072B2",
    "accent_light": "#CBE5F6",
    "gold": "#E69F00",
    "raw": "#1F2937",
    "grid": "#E5E7EB",
    "text": "#1F2937",
    "muted": "#6B7280",
    "negative": "#B94A48",
    "positive": "#009E73",
    "panel": "#F8FAFC",
    "ink": "#111827",
    "night": "#111827",
    "paper": "#F7FAFC",
}


def apply_plot_style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.4,
            "axes.titlesize": 8.4,
            "axes.labelsize": 8.2,
            "axes.edgecolor": "#3A3A3A",
            "axes.labelcolor": PALETTE["text"],
            "xtick.color": PALETTE["text"],
            "ytick.color": PALETTE["text"],
            "axes.grid": True,
            "grid.color": PALETTE["grid"],
            "grid.linewidth": 0.55,
            "grid.alpha": 0.62,
            "axes.titleweight": "medium",
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def objective_label(objective):
    return OBJECTIVE_LABELS.get(objective, str(objective).replace("_", "-"))


def model_label(model):
    return MODEL_LABELS.get(model, str(model).upper())


def model_short_label(model):
    return MODEL_SHORT_LABELS.get(model, model_label(model))


def objective_color(objective):
    colors = {
        "density_hard": PALETTE["bdl"],
        "density_gau": PALETTE["bdl_mid"],
        "density_custom": PALETTE["bdl_soft"],
        "seg": PALETTE["seg"],
        "seg_weighted": PALETTE["seg_mid"],
        "seg_focal": PALETTE["seg_soft"],
    }
    return colors.get(objective, PALETTE["raw"])


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


def save_placeholder(outdir, filename, title):
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    ax.axis("off")
    ax.text(0.5, 0.50, title, ha="center", va="center", fontsize=12)
    fig.tight_layout()
    fig.savefig(outdir / filename, dpi=200)
    plt.close(fig)


def plot_method_overview(outdir):
    t = np.linspace(0, 100, 720)
    onset = 34.0
    offset = 70.0
    sigma = 3.0
    tolerance = 5.0

    rise = 1.0 / (1.0 + np.exp(-(t - onset) / 2.2))
    fall = 1.0 / (1.0 + np.exp(-(t - offset) / 2.6))
    signal = (
        0.17 * np.sin(t / 4.0)
        + 0.07 * np.cos(t / 1.8)
        + 0.04 * np.sin(t / 1.15)
        + 0.62 * (rise - fall)
    )
    state = ((t >= onset) & (t <= offset)).astype(float)

    def unit_kernel(center):
        values = np.exp(-0.5 * ((t - center) / sigma) ** 2)
        values[np.abs(t - center) > 3 * sigma] = 0.0
        area = np.trapz(values, t)
        return values / max(area, 1e-12)

    onset_target = unit_kernel(onset)
    offset_target = unit_kernel(offset)
    count_target = onset_target + offset_target
    count_display = count_target / count_target.max()
    rate = 0.98 * count_display + 0.04 * np.exp(-0.5 * ((t - 52.0) / 8.0) ** 2)
    rate = rate / rate.max()

    fig = plt.figure(figsize=(10.0, 3.75), facecolor="white")
    gs = fig.add_gridspec(
        2,
        3,
        height_ratios=[1.10, 0.96],
        width_ratios=[1.0, 1.0, 1.0],
        hspace=0.34,
        wspace=0.28,
        left=0.060,
        right=0.985,
        bottom=0.115,
        top=0.935,
    )
    ax_signal = fig.add_subplot(gs[0, :])
    ax_state = fig.add_subplot(gs[1, 0], sharex=ax_signal)
    ax_bdl = fig.add_subplot(gs[1, 1], sharex=ax_signal)
    ax_metric = fig.add_subplot(gs[1, 2], sharex=ax_signal)

    for ax in (ax_signal, ax_metric, ax_state, ax_bdl):
        ax.axvspan(onset, offset, color=PALETTE["seg_light"], alpha=0.13, lw=0)
        ax.axvline(onset, color=PALETTE["bdl_dark"], linewidth=1.10, linestyle="--", alpha=0.9)
        ax.axvline(offset, color=PALETTE["accent"], linewidth=1.10, linestyle="--", alpha=0.9)
        ax.grid(axis="x", color=PALETTE["grid"], linewidth=0.62)
        ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.45, alpha=0.50)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    ax_signal.set_facecolor("white")
    ax_signal.plot(t, signal, color=PALETTE["raw"], linewidth=2.25)
    ax_signal.fill_between(t, signal.min() - 0.18, signal, color="#CBD5E1", alpha=0.24)
    signal_y = [np.interp(onset, t, signal), np.interp(offset, t, signal)]
    ax_signal.scatter([onset, offset], signal_y, s=92, color=[PALETTE["bdl"], PALETTE["accent"]], edgecolor="white", linewidth=1.15, zorder=5)
    ax_signal.set_title("A. Signal", loc="left", pad=4, fontsize=8.0, weight="semibold")
    ax_signal.set_ylabel("")
    ax_signal.set_yticks([])
    ax_signal.tick_params(axis="x", labelbottom=False)
    ax_signal.set_ylim(signal.min() - 0.18, signal.max() + 0.36)

    ax_state.set_facecolor("white")
    ax_state.fill_between(t, 0, state, step="mid", color=PALETTE["seg"], alpha=0.88)
    ax_state.plot(t, state, color=PALETTE["seg_dark"], linewidth=1.15)
    ax_state.set_title("B. State", loc="left", fontsize=7.4, weight="semibold", color=PALETTE["seg_dark"])
    ax_state.set_yticks([0, 1])
    ax_state.set_yticklabels(["0", "1"], fontsize=7.7)
    ax_state.set_ylim(-0.08, 1.12)
    ax_state.set_xlabel("")

    ax_bdl.set_facecolor("white")
    ax_bdl.fill_between(t, 0, count_display, color=PALETTE["bdl_light"], alpha=0.40)
    ax_bdl.plot(t, rate, color=PALETTE["bdl"], linewidth=2.45)
    for x, color in [(onset, PALETTE["bdl"]), (offset, PALETTE["accent"])]:
        y = np.interp(x, t, rate)
        ax_bdl.scatter(x, y, s=82, color=color, edgecolor="white", linewidth=1.0, zorder=5)
    ax_bdl.set_title("C. Expected events", loc="left", fontsize=7.4, weight="semibold", color=PALETTE["bdl_dark"])
    ax_bdl.set_yticks([])
    ax_bdl.set_ylim(-0.05, 1.30)
    ax_bdl.set_xlabel("")

    ax_metric.set_facecolor("white")
    ax_metric.set_title("D. Detections", loc="left", fontsize=7.4, weight="semibold")
    ax_metric.set_ylim(-0.1, 1.12)
    ax_metric.set_yticks([])
    ax_metric.set_xlabel("")
    for x, color in [(onset, PALETTE["bdl"]), (offset, PALETTE["accent"])]:
        ax_metric.axvspan(x - tolerance, x + tolerance, color=color, alpha=0.15, lw=0)
        ax_metric.scatter([x], [0.63], s=170, color=color, edgecolor="white", linewidth=1.25, zorder=5)
    ax_metric.plot([onset, offset], [0.63, 0.63], color="#CBD5E1", linewidth=1.55, zorder=0)

    for ax in (ax_signal, ax_state, ax_bdl):
        ax.set_xlim(0, 100)
        ax.tick_params(axis="x", labelsize=8.0)
    ax_metric.set_xlim(0, 100)
    ax_metric.tick_params(axis="x", labelsize=8.0)

    fig.savefig(outdir / "method_overview.png", dpi=280)
    plt.close(fig)


def tuned_map(scores, dataset, *, models=None, objectives=None, run_tags=None, dedupe_cols=None):
    if scores.empty:
        return pd.DataFrame()
    frame = scores[
        (scores["row_type"] == "summary")
        & (scores["stage"] == "tuned")
        & (scores["optimized_metric"] == "mAP")
        & (scores["metric"] == "mAP")
    ].copy()
    frame = frame[frame["dataset"] == dataset]
    frame = frame[frame["objective"].isin(MAIN_TOLERANCE_OBJECTIVES)]
    if models is not None:
        frame = frame[frame["model"].isin(models)]
    if objectives is not None:
        frame = frame[frame["objective"].isin(objectives)]
    if run_tags is not None:
        frame = frame[frame["run_tag"].isin(run_tags)]
    if frame.empty:
        return frame
    frame["score"] = pd.to_numeric(frame["score"], errors="coerce")
    grouped = (
        frame.groupby(["model", "objective", "run_tag", "postprocess_objective"], as_index=False)["score"]
        .mean()
        .dropna()
        .sort_values("score", ascending=False)
    )
    if dedupe_cols is None:
        dedupe_cols = ["model", "objective"]
    return grouped.drop_duplicates(dedupe_cols, keep="first")


def transformer_replaces_attention_unet(scores):
    model = "transformer_6l_128h_8a_10d"
    frame = tuned_map(
        scores,
        "sleep",
        models=[model],
        objectives=["density_hard", "seg"],
        run_tags=[TRANSFORMER_STRONG_CANDIDATE_TAG],
        dedupe_cols=["model", "objective", "run_tag"],
    )
    if frame.empty:
        return False
    bdl = frame[(frame["model"] == model) & (frame["objective"] == "density_hard")]
    seg = frame[(frame["model"] == model) & (frame["objective"] == "seg")]
    if bdl.empty or seg.empty:
        return False
    return float(bdl.iloc[0]["score"]) >= float(seg.iloc[0]["score"])


def sleep_architecture_frame(scores):
    if transformer_replaces_attention_unet(scores):
        base = tuned_map(
            scores,
            "sleep",
            models=["gru", "unet"],
            objectives=["density_hard", "seg"],
            run_tags=[MAIN_RUN_TAG],
        )
        transformer = tuned_map(
            scores,
            "sleep",
            models=["transformer_6l_128h_8a_10d"],
            objectives=["density_hard", "seg"],
            run_tags=[TRANSFORMER_STRONG_CANDIDATE_TAG],
            dedupe_cols=["model", "objective", "run_tag"],
        )
        return pd.concat([base, transformer], ignore_index=True)
    return tuned_map(
        scores,
        "sleep",
        models=["gru", "unet", "unet_t"],
        objectives=["density_hard", "seg"],
        run_tags=[MAIN_RUN_TAG],
    )


def sleep_architecture_models(frame):
    if frame.empty:
        return []
    order = ["gru", "unet", "transformer_6l_128h_8a_10d", "unet_t"]
    available = set(frame["model"].astype(str))
    return [model for model in order if model in available]


def plot_target_kernels(dataclass, outdir, dataset):
    fig = plt.figure(figsize=(11.35, 4.70))
    gs = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.35, 1.45, 1.18],
        wspace=0.34,
    )
    ax_shape = fig.add_subplot(gs[0, 0])
    ax_bins = fig.add_subplot(gs[0, 1])
    ax_flow = fig.add_subplot(gs[0, 2])
    fig.subplots_adjust(top=0.92)
    ax_shape.set_facecolor("#F8FAFC")
    ax_bins.set_facecolor("#F8FAFC")

    colors = {
        "density_hard": objective_color("density_hard"),
        "density_gau": objective_color("density_gau"),
        "density_custom": objective_color("density_custom"),
    }
    values_by_objective = {}
    x_by_objective = {}
    for objective in DENSITY_OBJECTIVES:
        values = get_target_distribution(dataclass, base_objective(objective), unit_mass=True)
        center = len(values) // 2
        x = np.arange(len(values)) - center
        values_by_objective[objective] = values
        x_by_objective[objective] = x

    for objective in ["density_custom", "density_gau", "density_hard"]:
        x = x_by_objective[objective]
        values = values_by_objective[objective]
        positive = values > 0
        if not np.any(positive):
            continue
        normalized = values / values.max()
        label = {
            "density_hard": "Hard",
            "density_gau": "Gaussian",
            "density_custom": "Tolerance",
        }[objective]
        ax_shape.fill_between(
            x[positive],
            0,
            normalized[positive],
            color=colors.get(objective, PALETTE["raw"]),
            alpha=0.20 if objective != "density_hard" else 0.50,
            step="mid" if objective == "density_custom" else None,
        )
        ax_shape.plot(
            x[positive],
            normalized[positive],
            marker="o" if objective == "density_hard" else None,
            markersize=5.2,
            color=colors.get(objective, PALETTE["raw"]),
            linewidth=2.8 if objective == "density_gau" else 2.5,
            drawstyle="steps-mid" if objective == "density_custom" else "default",
            label=label,
        )
    ax_shape.axvline(0, color="#777777", linewidth=1.0, linestyle=":")
    ax_shape.set_title("A. Kernel", loc="left", weight="bold", fontsize=11.2)
    ax_shape.set_xlabel("Offset from label", fontsize=9.4)
    ax_shape.set_ylabel("Relative target", fontsize=9.4)
    ax_shape.set_ylim(-0.04, 1.25)
    ax_shape.legend(loc="upper right", fontsize=8.2, handlelength=1.8)

    gau = values_by_objective["density_gau"]
    x = x_by_objective["density_gau"]
    positive = gau > 0
    display = gau / max(gau.max(), 1e-12)
    bin_width = max(1, int(round(len(gau) / 14)))
    center = len(gau) // 2
    start = max(0, center - 4 * bin_width)
    stop = min(len(gau), center + 5 * bin_width)
    starts = np.arange(start, stop, bin_width)
    centers = starts + bin_width / 2 - center
    masses = np.array([gau[s : min(s + bin_width, len(gau))].sum() for s in starts])
    ax_bins.fill_between(
        x[positive],
        0,
        display[positive],
        color=PALETTE["accent"],
        alpha=0.20,
    )
    ax_bins.plot(
        x[positive],
        display[positive],
        color=PALETTE["accent"],
        linewidth=2.5,
    )
    ax_bins.bar(
        centers,
        masses / max(masses.max(), 1e-12),
        width=bin_width * 0.86,
        color=PALETTE["bdl"],
        alpha=0.82,
        edgecolor="white",
        linewidth=1.0,
    )
    for edge in starts - center:
        ax_bins.axvline(edge, color="#FFFFFF", linewidth=0.7, alpha=0.9)
    ax_bins.set_xlim(centers.min() - bin_width * 0.8, centers.max() + bin_width * 0.8)
    ax_bins.set_ylim(-0.04, 1.23)
    ax_bins.set_yticks([])
    ax_bins.set_xlabel("Output bins", fontsize=9.4)
    ax_bins.set_title("B. Bins", loc="left", weight="bold", fontsize=11.2)

    ax_flow.axis("off")
    ax_flow.set_title("C. Fit", loc="left", weight="bold", fontsize=11.2)
    steps = [
        ("target", r"$Y_b$", PALETTE["bdl"]),
        ("prediction", r"$\lambda_b$", PALETTE["accent"]),
        ("loss", r"$\lambda_b - Y_b\log\lambda_b$", PALETTE["seg"]),
    ]
    for idx, (label, value, color) in enumerate(steps):
        y0 = 0.78 - idx * 0.29
        ax_flow.add_patch(
            FancyBboxPatch(
                (0.04, y0 - 0.085),
                0.90,
                0.20,
                boxstyle="round,pad=0.008,rounding_size=0.018",
                transform=ax_flow.transAxes,
                facecolor=color,
                alpha=0.15,
                edgecolor=color,
                linewidth=1.25,
            )
        )
        ax_flow.text(
            0.18,
            y0,
            label,
            transform=ax_flow.transAxes,
            ha="left",
            va="center",
            fontsize=8.2,
            color=PALETTE["muted"],
        )
        ax_flow.text(
            0.62,
            y0,
            value,
            transform=ax_flow.transAxes,
            ha="center",
            va="center",
            fontsize=14.0 if idx < 2 else 9.2,
            color=color,
            weight="bold",
        )
        if idx < len(steps) - 1:
            ax_flow.text(
                0.62,
                y0 - 0.145,
                r"$\downarrow$",
                transform=ax_flow.transAxes,
                ha="center",
                va="center",
                fontsize=12.0,
                color=PALETTE["muted"],
            )

    for ax in (ax_shape, ax_bins):
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.savefig(outdir / f"{dataset}_target_kernels.png", dpi=240)
    if dataset == "sleep":
        fig.savefig(outdir / "target_kernels.png", dpi=240)
    plt.close(fig)


def plot_target_pipeline_column(dataclass, outdir):
    values = get_target_distribution(dataclass, "gau", unit_mass=True)
    center = len(values) // 2
    x = np.arange(len(values)) - center
    positive = values > 0
    if not np.any(positive):
        return
    display = values / max(values.max(), 1e-12)
    bin_width = max(1, int(round(len(values) / 12)))
    start = max(0, center - 4 * bin_width)
    stop = min(len(values), center + 5 * bin_width)
    starts = np.arange(start, stop, bin_width)
    centers = starts + bin_width / 2 - center
    masses = np.array([values[s : min(s + bin_width, len(values))].sum() for s in starts])
    masses = masses / max(masses.max(), 1e-12)

    fig = plt.figure(figsize=(3.45, 4.05))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 0.92, 0.40], hspace=0.48)
    ax_shape = fig.add_subplot(gs[0, 0])
    ax_bins = fig.add_subplot(gs[1, 0])
    ax_flow = fig.add_subplot(gs[2, 0])

    ax_shape.set_facecolor("#F8FAFC")
    ax_shape.fill_between(x[positive], 0, display[positive], color=PALETTE["bdl_light"], alpha=0.65)
    ax_shape.plot(x[positive], display[positive], color=PALETTE["bdl"], linewidth=2.4)
    ax_shape.axvline(0, color=PALETTE["muted"], linestyle=":", linewidth=1.1)
    ax_shape.scatter([0], [1.0], s=58, color=PALETTE["bdl"], edgecolor="white", linewidth=1.0, zorder=4)
    ax_shape.set_title("A. Kernel", loc="left", fontsize=6.4, weight="semibold")
    ax_shape.set_ylabel("")
    ax_shape.set_xlabel("")
    ax_shape.set_yticks([0, 1])
    ax_shape.tick_params(labelsize=5.9)

    ax_bins.set_facecolor("#F8FAFC")
    ax_bins.bar(centers, masses, width=bin_width * 0.82, color=PALETTE["bdl"], edgecolor="white", linewidth=0.8, alpha=0.86)
    ax_bins.set_title("B. Bins", loc="left", fontsize=6.4, weight="semibold")
    ax_bins.set_ylabel("")
    ax_bins.set_xlabel("")
    ax_bins.set_yticks([])
    ax_bins.tick_params(labelsize=5.9)

    ax_flow.axis("off")
    ax_flow.set_title("C. Fit", loc="left", fontsize=6.4, weight="semibold")
    ax_flow.text(0.24, 0.48, r"$Y_b$", transform=ax_flow.transAxes, ha="center", va="center", fontsize=9.6, weight="bold", color=PALETTE["bdl"])
    ax_flow.text(0.50, 0.48, r"$\rightarrow$", transform=ax_flow.transAxes, ha="center", va="center", fontsize=8.3, color=PALETTE["muted"])
    ax_flow.text(0.76, 0.48, r"$\lambda_b$", transform=ax_flow.transAxes, ha="center", va="center", fontsize=9.6, weight="bold", color=PALETTE["accent"])

    for ax in (ax_shape, ax_bins):
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.subplots_adjust(left=0.17, right=0.98, top=0.975, bottom=0.075)
    fig.savefig(outdir / "target_pipeline_column.png", dpi=280)
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
    frame = frame[frame["objective"].isin(MAIN_TOLERANCE_OBJECTIVES)]
    if frame.empty:
        return
    grouped = (
        frame.groupby(["objective", "model"], as_index=False)["score"]
        .mean()
        .sort_values("score", ascending=False)
    )
    grouped["label"] = [
        f"{objective_label(row.objective)}\n{model_label(row.model)}"
        for row in grouped.itertuples()
    ]
    fig_width = max(8, 0.45 * len(grouped))
    fig, ax = plt.subplots(figsize=(fig_width, 4))
    colors = [objective_color(obj) for obj in grouped["objective"]]
    bars = ax.bar(np.arange(len(grouped)), grouped["score"], color=colors)
    ax.xaxis.grid(False)
    ax.yaxis.grid(True)
    for bar, score in zip(bars, grouped["score"]):
        ax.text(bar.get_x() + bar.get_width() / 2, score + 0.01, f"{score:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(np.arange(len(grouped)))
    ax.set_xticklabels(grouped["label"], rotation=60, ha="right")
    ax.set_ylabel("Tuned mAP")
    ax.set_title("Objective/model comparison")
    fig.tight_layout()
    fig.savefig(outdir / f"{dataset}_summary_bars.png", dpi=200)
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
    frame = frame[frame["objective"].isin(MAIN_TOLERANCE_OBJECTIVES)]
    if frame.empty:
        return
    frame["tolerance"] = pd.to_numeric(frame["tolerance"])
    if dataset == "sleep":
        frame["display_tolerance"] = frame["tolerance"] / 12.0
        tolerance_label = "Tolerance (minutes)"
    elif dataset == "seizure":
        frame["display_tolerance"] = frame["tolerance"] / 256.0
        tolerance_label = "Tolerance (seconds)"
    else:
        frame["display_tolerance"] = frame["tolerance"]
        tolerance_label = "Tolerance"
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = {objective: objective_color(objective) for objective in MAIN_TOLERANCE_OBJECTIVES}
    for objective, group in frame.groupby("objective"):
        curve = group.groupby("display_tolerance", as_index=False)["score"].mean()
        ax.plot(
            curve["display_tolerance"],
            curve["score"],
            marker="o",
            color=colors.get(objective, PALETTE["raw"]),
            linewidth=1.8,
            label=objective_label(objective),
        )
    ax.set_xlabel(tolerance_label)
    ax.set_ylabel("mAP")
    ax.set_title("EDAP by tolerance")
    ax.legend(frameon=False, ncols=2)
    fig.tight_layout()
    fig.savefig(outdir / f"{dataset}_tolerance_curves.png", dpi=200)
    plt.close(fig)


def plot_sleep_objective_sweep(scores, outdir):
    objectives = [
        "density_hard",
        "density_gau",
        "density_custom",
        "seg",
        "seg_weighted",
        "seg_focal",
    ]
    frame = tuned_map(
        scores,
        "sleep",
        models=["gru"],
        objectives=objectives,
        run_tags=[MAIN_RUN_TAG, "lr3e3_e20_gpu_bs32_eval5"],
    )
    if frame.empty:
        save_placeholder(outdir, "sleep_objective_sweep.png", "Sleep objective sweep")
        return
    frame["rank"] = frame["objective"].map({name: idx for idx, name in enumerate(objectives)})
    frame = frame.sort_values("rank")
    fig, ax = plt.subplots(figsize=(7.5, 3.5))
    colors = [objective_color(obj) for obj in frame["objective"]]
    bars = ax.bar(np.arange(len(frame)), frame["score"], color=colors)
    ax.xaxis.grid(False)
    ax.yaxis.grid(True)
    for bar, score in zip(bars, frame["score"]):
        ax.text(bar.get_x() + bar.get_width() / 2, score + 0.01, f"{score:.3f}", ha="center", va="bottom", fontsize=8.5)
    ax.set_ylim(0, max(frame["score"]) * 1.16)
    ax.set_xticks(np.arange(len(frame)))
    ax.set_xticklabels([objective_label(obj) for obj in frame["objective"]], rotation=35, ha="right")
    ax.set_ylabel("Tuned mAP")
    ax.set_title("Sleep GRU objective sweep")
    fig.tight_layout()
    fig.savefig(outdir / "sleep_objective_sweep.png", dpi=200)
    plt.close(fig)


def plot_sleep_main_results(scores, outdir):
    objectives = [
        "density_hard",
        "density_gau",
        "density_custom",
        "seg",
        "seg_weighted",
        "seg_focal",
    ]
    objective_frame = tuned_map(
        scores,
        "sleep",
        models=["gru"],
        objectives=objectives,
        run_tags=[MAIN_RUN_TAG, "lr3e3_e20_gpu_bs32_eval5"],
    )
    arch_frame = sleep_architecture_frame(scores)
    tolerance_frame = scores[
        (scores["row_type"] == "tolerance")
        & (scores["stage"] == "tuned")
        & (scores["optimized_metric"] == "mAP")
        & (scores["metric"] == "mAP")
        & (scores["dataset"] == "sleep")
        & (scores["model"] == "gru")
        & (scores["run_tag"].isin([MAIN_RUN_TAG, "lr3e3_e20_gpu_bs32_eval5"]))
        & (scores["objective"].isin(["density_hard", "density_gau", "density_custom", "seg"]))
    ].copy()

    if objective_frame.empty or arch_frame.empty or tolerance_frame.empty:
        save_placeholder(outdir, "sleep_main_results.png", "Sleep benchmark main results")
        return

    objective_frame["rank"] = objective_frame["objective"].map({name: idx for idx, name in enumerate(objectives)})
    objective_frame = objective_frame.sort_values("rank")
    selected_postprocess = objective_frame.set_index("objective")["postprocess_objective"].to_dict()
    tolerance_frame = tolerance_frame[
        tolerance_frame.apply(
            lambda row: row["postprocess_objective"] == selected_postprocess.get(row["objective"], row["postprocess_objective"]),
            axis=1,
        )
    ].copy()
    if tolerance_frame.empty:
        save_placeholder(outdir, "sleep_main_results.png", "Sleep benchmark main results")
        return
    tolerance_frame["tolerance"] = pd.to_numeric(tolerance_frame["tolerance"])
    tolerance_frame["score"] = pd.to_numeric(tolerance_frame["score"], errors="coerce")
    tolerance_frame["display_tolerance"] = tolerance_frame["tolerance"] / 12.0

    objective_colors = {objective: objective_color(objective) for objective in objectives}
    fig = plt.figure(figsize=(8.10, 5.15), facecolor="white")
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.05, 1.0],
        width_ratios=[1.16, 1.0],
        wspace=0.34,
        hspace=0.44,
    )
    ax_tol = fig.add_subplot(gs[:, 0])
    ax_obj = fig.add_subplot(gs[0, 1])
    ax_arch = fig.add_subplot(gs[1, 1])
    line_objectives = ["density_hard", "density_gau", "density_custom", "seg"]
    line_colors = {objective: objective_color(objective) for objective in line_objectives}
    for objective in line_objectives:
        group = tolerance_frame[tolerance_frame["objective"] == objective]
        if group.empty:
            continue
        curve = group.groupby("display_tolerance", as_index=False)["score"].mean().sort_values("display_tolerance")
        ax_tol.plot(
            curve["display_tolerance"],
            curve["score"],
            marker="o",
            markersize=2.8 if objective == "density_hard" else 2.35,
            linewidth=1.45 if objective == "density_hard" else 1.12,
            color=line_colors[objective],
            label=objective_label(objective),
            alpha=0.98 if objective == "density_hard" else 0.84,
        )
    ax_tol.set_xlabel("Tolerance (minutes)", fontsize=7.8)
    ax_tol.set_ylabel("AP", fontsize=7.8)
    ax_tol.set_title("A. AP by tolerance", loc="left", weight="semibold", fontsize=7.1)
    ax_tol.legend(
        loc="upper center",
        bbox_to_anchor=(0.55, -0.11),
        fontsize=5.9,
        ncols=2,
        handlelength=1.05,
        columnspacing=0.85,
        labelspacing=0.22,
        borderaxespad=0.0,
    )
    ax_tol.set_ylim(0, 1.0)
    colors = [objective_colors.get(obj, PALETTE["raw"]) for obj in objective_frame["objective"]]
    y = np.arange(len(objective_frame))[::-1]
    ax_obj.barh(y, objective_frame["score"], color=colors, height=0.46, edgecolor="white", linewidth=0.55)
    ax_obj.set_yticks(y)
    ax_obj.set_yticklabels([objective_label(obj) for obj in objective_frame["objective"]], fontsize=6.7)
    ax_obj.set_xlim(0, max(objective_frame["score"]) * 1.12)
    ax_obj.set_xlabel("mAP", fontsize=7.8)
    ax_obj.set_title("B. Objectives", loc="left", weight="semibold", fontsize=7.1)
    ax_obj.xaxis.grid(True)
    ax_obj.yaxis.grid(False)

    models = sleep_architecture_models(arch_frame)
    model_names = [model_short_label(model) for model in models]
    bdl_values = []
    seg_values = []
    for model in models:
        bdl = arch_frame[(arch_frame["model"] == model) & (arch_frame["objective"] == "density_hard")]
        seg = arch_frame[(arch_frame["model"] == model) & (arch_frame["objective"] == "seg")]
        bdl_values.append(float(bdl.iloc[0]["score"]) if not bdl.empty else np.nan)
        seg_values.append(float(seg.iloc[0]["score"]) if not seg.empty else np.nan)
    y_arch = np.arange(len(models))[::-1] * 0.72
    ax_arch.set_title("C. Backbones", loc="left", weight="semibold", fontsize=7.1)
    for yi, model_name, bdl, seg in zip(y_arch, model_names, bdl_values, seg_values):
        if not (np.isfinite(bdl) and np.isfinite(seg)):
            continue
        ax_arch.plot([seg, bdl], [yi, yi], color=PALETTE["bdl_dark"], linewidth=0.95, alpha=0.46, zorder=1)
        ax_arch.scatter(seg, yi, s=24, color=PALETTE["seg"], edgecolor="white", linewidth=0.42, zorder=3)
        ax_arch.scatter(bdl, yi, s=28, color=PALETTE["bdl"], edgecolor="white", linewidth=0.48, zorder=4)
    ax_arch.set_yticks(y_arch)
    ax_arch.set_yticklabels(model_names, fontsize=6.7, linespacing=0.95)
    ax_arch.set_xlabel("mAP", fontsize=7.8)
    ax_arch.set_xlim(0.20, max(np.nanmax(bdl_values), np.nanmax(seg_values)) * 1.18)
    ax_arch.set_ylim(y_arch.min() - 0.34, y_arch.max() + 0.34)
    ax_arch.xaxis.grid(True)
    ax_arch.yaxis.grid(False)
    ax_arch.scatter([], [], s=24, color=PALETTE["seg"], label="CE")
    ax_arch.scatter([], [], s=28, color=PALETTE["bdl"], label="BDL")
    ax_arch.legend(loc="lower right", fontsize=5.9, handlelength=0.55, borderaxespad=0.20, frameon=False)

    for ax in (ax_obj, ax_arch, ax_tol):
        ax.set_facecolor("white")
        ax.grid(color=PALETTE["grid"], linewidth=0.50, alpha=0.52)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    fig.subplots_adjust(left=0.078, right=0.985, top=0.945, bottom=0.125)
    fig.savefig(outdir / "sleep_main_results.png", dpi=300)
    plt.close(fig)


def plot_sleep_prediction_example(outdir):
    """Plot one real validation window from saved sleep predictions."""
    series_id = "0402a003dae9"
    id_map = 2
    onset_step = 233976.0
    wake_step = 238404.0
    downsample = 10
    margin_steps = 5000
    raw_step_seconds = 5.0
    paths = {
        "bdl": REPO_ROOT
        / "experiments/sleep/gru/density_hard/seed_0_objective_e20_bs32_eval5/predictions"
        / f"{series_id}.npy",
        "seg": REPO_ROOT
        / "experiments/sleep/gru/seg/seed_0_objective_e20_bs32_eval5/predictions"
        / f"{series_id}.npy",
        "series": REPO_ROOT / "data/sleep/train_series.parquet",
    }
    if not all(path.exists() for path in paths.values()):
        save_placeholder(outdir, "sleep_prediction_example.png", "Sleep validation example")
        return

    start = int(onset_step - margin_steps)
    stop = int(wake_step + margin_steps)
    try:
        series = pd.read_parquet(
            paths["series"],
            filters=[("id_map", "=", id_map), ("step", ">=", start), ("step", "<=", stop)],
        ).sort_values("step")
    except Exception:
        save_placeholder(outdir, "sleep_prediction_example.png", "Sleep validation example")
        return
    if series.empty:
        save_placeholder(outdir, "sleep_prediction_example.png", "Sleep validation example")
        return

    bdl = np.load(paths["bdl"])
    seg = np.load(paths["seg"])[:, 0]
    pred_steps = np.arange(len(seg)) * downsample
    mask = (pred_steps >= start) & (pred_steps <= stop)
    pred_steps = pred_steps[mask]
    bdl = bdl[mask]
    seg = seg[mask]

    hours_raw = (series["step"].to_numpy() - onset_step) * raw_step_seconds / 3600.0
    hours_pred = (pred_steps - onset_step) * raw_step_seconds / 3600.0
    onset_hour = 0.0
    wake_hour = (wake_step - onset_step) * raw_step_seconds / 3600.0

    angle = series["anglez"].astype(float).rolling(48, center=True, min_periods=1).median().to_numpy()
    enmo = series["enmo"].astype(float).rolling(48, center=True, min_periods=1).median().to_numpy()
    angle_scale = max(np.nanpercentile(np.abs(angle - np.nanmedian(angle)), 92), 1e-6)
    angle_norm = (angle - np.nanmedian(angle)) / angle_scale
    enmo_norm = enmo / max(np.nanpercentile(enmo, 98), 1e-6)

    bdl_on = bdl[:, 0] / max(bdl[:, 0].max(), 1e-12)
    bdl_off = bdl[:, 1] / max(bdl[:, 1].max(), 1e-12)
    transition = np.abs(np.gradient(seg))
    transition = transition / max(transition.max(), 1e-12)

    fig = plt.figure(figsize=(8.55, 6.85), facecolor="white")
    gs = fig.add_gridspec(3, 1, height_ratios=[1.12, 1.0, 1.10], hspace=0.26)
    ax_signal = fig.add_subplot(gs[0, 0])
    ax_seg = fig.add_subplot(gs[1, 0], sharex=ax_signal)
    ax_bdl = fig.add_subplot(gs[2, 0], sharex=ax_signal)
    axes = [ax_signal, ax_seg, ax_bdl]

    for ax in axes:
        ax.axvspan(onset_hour, wake_hour, color=PALETTE["seg_light"], alpha=0.10, lw=0)
        ax.axvline(onset_hour, color=PALETTE["bdl_dark"], linestyle="--", linewidth=0.95)
        ax.axvline(wake_hour, color=PALETTE["accent"], linestyle="--", linewidth=0.95)
        ax.grid(axis="x", color=PALETTE["grid"], linewidth=0.55)
        ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.38, alpha=0.58)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    ax_signal.set_facecolor("white")
    ax_signal.set_title("A. Signal", loc="left", fontsize=7.6, weight="semibold")
    ax_signal.plot(hours_raw, angle_norm, color=PALETTE["raw"], linewidth=0.82, label="anglez")
    ax_signal.fill_between(hours_raw, 0, enmo_norm, color=PALETTE["gold"], alpha=0.22, label="ENMO")
    ax_signal.set_ylim(-1.9, 2.1)
    ax_signal.set_yticks([])
    ax_signal.set_ylabel("")
    ax_signal.legend(
        loc="upper right",
        fontsize=5.9,
        ncols=2,
        handlelength=1.35,
        columnspacing=0.9,
        borderaxespad=0.15,
    )

    ax_seg.set_facecolor("white")
    ax_seg.set_title("B. Segmentation", loc="left", fontsize=7.6, weight="semibold")
    ax_seg.plot(hours_pred, seg, color=PALETTE["seg"], linewidth=1.30, label="mask")
    ax_seg.plot(hours_pred, transition, color=PALETTE["seg_dark"], linewidth=0.76, alpha=0.52, label="edge")
    ax_seg.set_ylim(-0.04, 1.04)
    ax_seg.set_ylabel("")
    ax_seg.set_yticks([0, 1])

    ax_bdl.set_facecolor("white")
    ax_bdl.set_title("C. BDL", loc="left", fontsize=7.6, weight="semibold")
    ax_bdl.plot(hours_pred, bdl_on, color=PALETTE["bdl"], linewidth=1.45, label="onset")
    ax_bdl.plot(hours_pred, bdl_off, color=PALETTE["accent"], linewidth=1.45, label="wake-up")
    ax_bdl.fill_between(hours_pred, 0, bdl_on, color=PALETTE["bdl"], alpha=0.08)
    ax_bdl.fill_between(hours_pred, 0, bdl_off, color=PALETTE["accent"], alpha=0.08)
    onset_peak = float(np.interp(onset_hour, hours_pred, bdl_on))
    wake_peak = float(np.interp(wake_hour, hours_pred, bdl_off))
    ax_bdl.scatter(
        [onset_hour, wake_hour],
        [onset_peak, wake_peak],
        s=[56, 56],
        color=[PALETTE["bdl"], PALETTE["accent"]],
        edgecolor="white",
        linewidth=0.95,
        zorder=5,
    )
    ax_bdl.set_ylim(-0.04, 1.05)
    ax_bdl.set_ylabel("")
    ax_bdl.set_xlabel("Hours relative to onset", fontsize=8.0)
    ax_bdl.set_yticks([0, 1])
    ax_bdl.set_xlim(hours_raw.min(), hours_raw.max())

    for ax in axes:
        ax.tick_params(axis="both", labelsize=7.2)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.955, bottom=0.078)
    fig.savefig(outdir / "sleep_prediction_example.png", dpi=300)
    plt.close(fig)

    fig_col = plt.figure(figsize=(3.45, 5.95))
    gs_col = fig_col.add_gridspec(3, 1, height_ratios=[0.98, 0.84, 1.00], hspace=0.19)
    c_signal = fig_col.add_subplot(gs_col[0, 0])
    c_seg = fig_col.add_subplot(gs_col[1, 0], sharex=c_signal)
    c_bdl = fig_col.add_subplot(gs_col[2, 0], sharex=c_signal)
    c_axes = [c_signal, c_seg, c_bdl]
    for ax in c_axes:
        ax.axvspan(onset_hour, wake_hour, color=PALETTE["seg_light"], alpha=0.16, lw=0)
        ax.axvline(onset_hour, color=PALETTE["bdl_dark"], linestyle="--", linewidth=1.05)
        ax.axvline(wake_hour, color=PALETTE["accent"], linestyle="--", linewidth=1.05)
        ax.grid(axis="x", color=PALETTE["grid"], linewidth=0.6)
        ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.45, alpha=0.70)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    c_signal.plot(hours_raw, angle_norm, color=PALETTE["raw"], linewidth=0.85)
    c_signal.fill_between(hours_raw, 0, enmo_norm, color=PALETTE["gold"], alpha=0.24)
    c_signal.set_ylim(-1.8, 2.05)
    c_signal.set_yticks([])
    c_signal.set_ylabel("")

    c_seg.plot(hours_pred, seg, color=PALETTE["seg"], linewidth=1.55)
    c_seg.plot(hours_pred, transition, color=PALETTE["seg_dark"], linewidth=0.85, alpha=0.55)
    c_seg.fill_between(hours_pred, 0, seg, color=PALETTE["seg"], alpha=0.10)
    c_seg.set_ylim(-0.04, 1.04)
    c_seg.set_yticks([0, 1])
    c_seg.set_yticklabels(["0", "1"], fontsize=6.8)
    c_seg.set_ylabel("")

    c_bdl.plot(hours_pred, bdl_on, color=PALETTE["bdl"], linewidth=1.65)
    c_bdl.plot(hours_pred, bdl_off, color=PALETTE["accent"], linewidth=1.65)
    c_bdl.fill_between(hours_pred, 0, bdl_on, color=PALETTE["bdl"], alpha=0.14)
    c_bdl.fill_between(hours_pred, 0, bdl_off, color=PALETTE["accent"], alpha=0.13)
    c_bdl.scatter(
        [onset_hour, wake_hour],
        [onset_peak, wake_peak],
        s=46,
        color=[PALETTE["bdl"], PALETTE["accent"]],
        edgecolor="white",
        linewidth=0.8,
        zorder=5,
    )
    c_bdl.set_ylim(-0.04, 1.05)
    c_bdl.set_yticks([0, 1])
    c_bdl.set_yticklabels(["0", "1"], fontsize=6.8)
    c_bdl.set_ylabel("")
    c_bdl.set_xlabel("Hours relative to onset", fontsize=7.6)
    c_bdl.set_xlim(hours_raw.min(), hours_raw.max())
    for ax in c_axes:
        ax.tick_params(axis="both", labelsize=6.8)
    plt.setp(c_signal.get_xticklabels(), visible=False)
    plt.setp(c_seg.get_xticklabels(), visible=False)
    fig_col.subplots_adjust(left=0.140, right=0.985, top=0.965, bottom=0.095)
    fig_col.savefig(outdir / "sleep_prediction_column.png", dpi=260)
    plt.close(fig_col)


def plot_sleep_results_column(scores, outdir):
    objectives = [
        "density_hard",
        "density_gau",
        "density_custom",
        "seg",
        "seg_weighted",
        "seg_focal",
    ]
    objective_frame = tuned_map(
        scores,
        "sleep",
        models=["gru"],
        objectives=objectives,
        run_tags=[MAIN_RUN_TAG, "lr3e3_e20_gpu_bs32_eval5"],
    )
    arch_frame = sleep_architecture_frame(scores)
    tolerance_frame = scores[
        (scores["row_type"] == "tolerance")
        & (scores["stage"] == "tuned")
        & (scores["optimized_metric"] == "mAP")
        & (scores["metric"] == "mAP")
        & (scores["dataset"] == "sleep")
        & (scores["model"] == "gru")
        & (scores["run_tag"].isin([MAIN_RUN_TAG, "lr3e3_e20_gpu_bs32_eval5"]))
        & (scores["objective"].isin(["density_hard", "seg"]))
    ].copy()
    if objective_frame.empty or arch_frame.empty or tolerance_frame.empty:
        save_placeholder(outdir, "sleep_results_column.png", "Sleep benchmark main results")
        return

    objective_frame["rank"] = objective_frame["objective"].map({name: idx for idx, name in enumerate(objectives)})
    objective_frame = objective_frame.sort_values("rank")
    objective_frame["score"] = pd.to_numeric(objective_frame["score"], errors="coerce")
    selected_postprocess = objective_frame.set_index("objective")["postprocess_objective"].to_dict()
    tolerance_frame = tolerance_frame[
        tolerance_frame.apply(
            lambda row: row.get("postprocess_objective") == selected_postprocess.get(row["objective"]),
            axis=1,
        )
    ].copy()
    if tolerance_frame.empty:
        save_placeholder(outdir, "sleep_results_column.png", "Sleep benchmark main results")
        return
    seg = float(objective_frame[objective_frame["objective"] == "seg"]["score"].iloc[0])

    tolerance_frame["tolerance"] = pd.to_numeric(tolerance_frame["tolerance"])
    tolerance_frame["score"] = pd.to_numeric(tolerance_frame["score"], errors="coerce")
    tolerance_frame["display_tolerance"] = tolerance_frame["tolerance"] / 12.0

    tol_summary = (
        tolerance_frame.groupby(["objective", "display_tolerance"], as_index=False)["score"]
        .mean()
        .sort_values("display_tolerance")
    )
    tol_colors = {"density_hard": objective_color("density_hard"), "seg": objective_color("seg")}
    colors = {objective: objective_color(objective) for objective in objectives}

    rows = []
    models = sleep_architecture_models(arch_frame)
    for model in models:
        bdl = arch_frame[(arch_frame["model"] == model) & (arch_frame["objective"] == "density_hard")]
        ce = arch_frame[(arch_frame["model"] == model) & (arch_frame["objective"] == "seg")]
        if not bdl.empty and not ce.empty:
            rows.append((model_short_label(model), float(ce.iloc[0]["score"]), float(bdl.iloc[0]["score"])))
    fig = plt.figure(figsize=(3.45, 6.70))
    gs = fig.add_gridspec(3, 1, height_ratios=[2.00, 2.25, 1.25], hspace=0.48)
    ax_tol = fig.add_subplot(gs[0])
    ax_obj = fig.add_subplot(gs[1])
    ax_arch = fig.add_subplot(gs[2])
    hard_curve = tol_summary[tol_summary["objective"] == "density_hard"]
    seg_curve = tol_summary[tol_summary["objective"] == "seg"]
    for objective, label in [("density_hard", "BDL-Hard"), ("seg", "Cross-entropy")]:
        curve = tol_summary[tol_summary["objective"] == objective]
        if curve.empty:
            continue
        ax_tol.plot(
            curve["display_tolerance"],
            curve["score"],
            marker="o",
            linewidth=2.25 if objective == "density_hard" else 1.85,
            markersize=4.8,
            color=tol_colors[objective],
            label=label,
            zorder=3 if objective == "density_hard" else 2,
        )
    if not hard_curve.empty and not seg_curve.empty:
        common_x = np.intersect1d(hard_curve["display_tolerance"], seg_curve["display_tolerance"])
        if common_x.size:
            hard_y = np.interp(common_x, hard_curve["display_tolerance"], hard_curve["score"])
            seg_y = np.interp(common_x, seg_curve["display_tolerance"], seg_curve["score"])
            ax_tol.fill_between(common_x, seg_y, hard_y, where=hard_y >= seg_y, color=PALETTE["bdl_light"], alpha=0.34, zorder=0)
    ax_tol.set_title("A  AP by matching tolerance", loc="left", fontsize=8.4, weight="bold")
    ax_tol.set_xscale("log")
    ax_tol.set_xticks([1, 3, 10, 30])
    ax_tol.set_xticklabels(["1", "3", "10", "30"], fontsize=7)
    ax_tol.set_xlim(0.85, 34)
    ax_tol.set_ylim(0, 0.92)
    ax_tol.set_xlabel("Tolerance (minutes)", fontsize=7.6)
    ax_tol.set_ylabel("AP", fontsize=8.0)
    ax_tol.tick_params(axis="y", labelsize=7.4)
    ax_tol.legend(loc="lower right", fontsize=6.7, handlelength=1.5, borderaxespad=0.2)
    y = np.arange(len(objective_frame))[::-1]
    ax_obj.axvline(seg, color=PALETTE["seg_dark"], linewidth=0.8, linestyle="--", alpha=0.75)
    ax_obj.barh(
        y,
        objective_frame["score"],
        height=0.68,
        color=[colors.get(obj, PALETTE["raw"]) for obj in objective_frame["objective"]],
        edgecolor="white",
        linewidth=0.8,
    )
    ax_obj.set_yticks(y)
    ax_obj.set_yticklabels([objective_label(obj) for obj in objective_frame["objective"]], fontsize=7.0)
    ax_obj.set_xlim(0, 0.78)
    ax_obj.set_xlabel("mAP", fontsize=7.6)
    ax_obj.set_title("B  Objectives", loc="left", fontsize=8.4, weight="bold")
    ax_obj.tick_params(axis="x", labelsize=7)
    ax_obj.xaxis.grid(True)
    ax_obj.yaxis.grid(False)
    ax_obj.text(seg + 0.012, y.max() + 0.42, "CE", fontsize=6.4, color=PALETTE["seg_dark"], va="center")
    for yi, score, objective in zip(y, objective_frame["score"], objective_frame["objective"]):
        ax_obj.text(
            score + 0.014,
            yi,
            f"{score:.3f}",
            va="center",
            fontsize=6.9,
            color=PALETTE["bdl_dark"] if objective == "density_hard" else PALETTE["text"],
            weight="bold" if objective == "density_hard" else "normal",
        )

    y_arch = np.arange(len(rows))[::-1]
    ax_arch.set_title("C  Backbones", loc="left", fontsize=8.4, weight="bold")
    for yi, (name, ce, bdl) in zip(y_arch, rows):
        gain_value = bdl - ce
        ax_arch.plot([ce, bdl], [yi, yi], color=PALETTE["bdl_dark"], linewidth=1.6, alpha=0.72)
        ax_arch.scatter(ce, yi, s=34, color=PALETTE["seg"], edgecolor="white", linewidth=0.6, zorder=3)
        ax_arch.scatter(bdl, yi, s=40, color=PALETTE["bdl"], edgecolor="white", linewidth=0.6, zorder=4)
        ax_arch.text(bdl + 0.018, yi, f"+{gain_value:.3f}", va="center", fontsize=6.9, color=PALETTE["bdl_dark"], weight="bold")
    ax_arch.set_yticks(y_arch)
    ax_arch.set_yticklabels([name for name, _, _ in rows], fontsize=7.0, fontweight="bold")
    ax_arch.set_xlim(0.20, 0.74)
    ax_arch.set_ylim(-0.35, max(len(rows) - 0.65, 0.65))
    ax_arch.set_xlabel("mAP", fontsize=7.6)
    ax_arch.tick_params(axis="x", labelsize=7)
    ax_arch.xaxis.grid(True)
    ax_arch.yaxis.grid(False)
    ax_arch.scatter([0.53], [1.06], transform=ax_arch.transAxes, s=22, color=PALETTE["seg"], clip_on=False)
    ax_arch.text(0.58, 1.06, "CE", transform=ax_arch.transAxes, fontsize=6.3, color=PALETTE["muted"], va="center")
    ax_arch.scatter([0.75], [1.06], transform=ax_arch.transAxes, s=22, color=PALETTE["bdl"], clip_on=False)
    ax_arch.text(0.80, 1.06, "BDL", transform=ax_arch.transAxes, fontsize=6.3, color=PALETTE["muted"], va="center")

    for ax in (ax_tol, ax_obj, ax_arch):
        ax.set_facecolor("#FAFBFC")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.subplots_adjust(left=0.165, right=0.985, top=0.905, bottom=0.075)
    fig.savefig(outdir / "sleep_results_column.png", dpi=240)
    plt.close(fig)


def plot_sleep_architecture_delta(scores, outdir):
    frame = sleep_architecture_frame(scores)
    models = sleep_architecture_models(frame)
    rows = []
    for model in models:
        bdl = frame[(frame["model"] == model) & (frame["objective"] == "density_hard")]
        seg = frame[(frame["model"] == model) & (frame["objective"] == "seg")]
        if not bdl.empty and not seg.empty:
            rows.append(
                {
                    "model": model,
                    "delta": float(bdl.iloc[0]["score"] - seg.iloc[0]["score"]),
                }
            )
    if not rows:
        save_placeholder(outdir, "sleep_architecture_delta.png", "Sleep architecture comparison")
        return
    deltas = pd.DataFrame(rows)
    colors = np.where(deltas["delta"] >= 0, PALETTE["positive"], PALETTE["negative"])
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    ax.axhline(0, color=PALETTE["text"], linewidth=0.8)
    bars = ax.bar(np.arange(len(deltas)), deltas["delta"], color=colors)
    ax.xaxis.grid(False)
    ax.yaxis.grid(True)
    for bar, delta in zip(bars, deltas["delta"]):
        label = f"{delta:+.3f}"
        va = "bottom" if delta >= 0 else "top"
        offset = 0.006 if delta >= 0 else -0.006
        ax.text(bar.get_x() + bar.get_width() / 2, delta + offset, label, ha="center", va=va, fontsize=8.5)
    ax.set_xticks(np.arange(len(deltas)))
    ax.set_xticklabels([model_short_label(model) for model in deltas["model"]])
    ax.set_ylabel("BDL-Hard mAP minus segmentation mAP")
    ax.set_title("Sleep architecture robustness")
    fig.tight_layout()
    fig.savefig(outdir / "sleep_architecture_delta.png", dpi=200)
    plt.close(fig)


def plot_run_tag_bars(scores, outdir, filename, title, run_tag_labels):
    frame = tuned_map(
        scores,
        "sleep",
        models=["gru"],
        objectives=["density_custom"],
        run_tags=list(run_tag_labels),
        dedupe_cols=["model", "objective", "run_tag"],
    )
    rows = []
    for run_tag, label in run_tag_labels.items():
        row = frame[
            (frame["postprocess_objective"].notna())
            & (frame["objective"] == "density_custom")
            & (frame["run_tag"] == run_tag)
        ]
        if row.empty:
            rows.append({"label": label, "score": np.nan})
        else:
            rows.append({"label": label, "score": float(row.iloc[0]["score"])})
    data = pd.DataFrame(rows)
    if data["score"].notna().sum() == 0:
        save_placeholder(outdir, filename, title)
        return
    fig, ax = plt.subplots(figsize=(5.5, 3.3))
    x = np.arange(len(data))
    bars = ax.bar(x, data["score"].fillna(0.0), color=PALETTE["bdl"])
    ax.xaxis.grid(False)
    ax.yaxis.grid(True)
    for bar, score in zip(bars, data["score"]):
        if not pd.isna(score):
            ax.text(bar.get_x() + bar.get_width() / 2, score + 0.01, f"{score:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(data["label"])
    ax.set_ylabel("Tuned mAP")
    ax.set_title(title)
    for idx, score in enumerate(data["score"]):
        if pd.isna(score):
            ax.text(idx, 0.02, "n/a", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / filename, dpi=200)
    plt.close(fig)


def plot_online_ablation(scores, outdir):
    models = ["fgru", "flstm", "causal_transformer"]
    frame = tuned_map(
        scores,
        "sleep",
        models=models,
        objectives=["density_hard", "seg"],
        run_tags=[ONLINE_RUN_TAG],
    )
    rows = []
    for model in models:
        for objective in ["density_hard", "seg"]:
            row = frame[(frame["model"] == model) & (frame["objective"] == objective)]
            if not row.empty:
                rows.append({"model": model, "objective": objective, "score": float(row.iloc[0]["score"])})
    if not rows:
        save_placeholder(outdir, "sleep_online_ablation.png", "Sleep online-context ablation")
        return
    data = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6.8, 3.25))
    x = np.arange(len(models))
    width = 0.35
    for offset, objective in [(-width / 2, "density_hard"), (width / 2, "seg")]:
        values = []
        for model in models:
            row = data[(data["model"] == model) & (data["objective"] == objective)]
            values.append(float(row.iloc[0]["score"]) if not row.empty else 0.0)
        color = objective_color(objective)
        ax.bar(
            x + offset,
            values,
            width=width,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            label=objective_label(objective),
        )
    ax.set_xticks(x)
    ax.set_xticklabels([model_short_label(model) for model in models])
    ax.set_ylabel("Tuned mAP")
    ax.legend(frameon=False, loc="upper left", ncols=2, borderaxespad=0.25)
    ax.set_ylim(0, max(data["score"]) * 1.25)
    ax.xaxis.grid(False)
    ax.yaxis.grid(True)
    fig.tight_layout()
    fig.savefig(outdir / "sleep_online_ablation.png", dpi=200)
    plt.close(fig)


def plot_seizure_replication(scores, outdir):
    frame = tuned_map(
        scores,
        "seizure",
        models=["gru", "unet"],
        objectives=["density_hard", "density_gau", "seg"],
        run_tags=[SEIZURE_RUN_TAG],
    )
    if frame.empty:
        save_placeholder(outdir, "seizure_replication.png", "Seizure benchmark replication")
        return
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    labels = []
    values = []
    colors = []
    for row in frame.sort_values(["model", "objective"]).itertuples():
        labels.append(f"{model_short_label(row.model)}\n{objective_label(row.objective)}")
        values.append(row.score)
        colors.append(objective_color(row.objective))
    bars = ax.bar(np.arange(len(values)), values, color=colors)
    ax.xaxis.grid(False)
    ax.yaxis.grid(True)
    ax.set_ylim(0, max(values) * 1.15)
    ax.set_xticks(np.arange(len(values)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Tuned mAP")
    fig.tight_layout()
    fig.savefig(outdir / "seizure_replication.png", dpi=200)
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
        ax.plot(values, color=PALETTE["bdl"], linewidth=1.2)
        ax.scatter(peaks, values[peaks], color=PALETTE["negative"], s=16, zorder=3)
        ax.set_ylabel(f"Channel {channel}")
    axes[-1].set_xlabel("Timestep")
    fig.suptitle(f"Predicted event scores: {prediction_path.name}")
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
    apply_plot_style()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    plot_method_overview(outdir)

    dataclass = apply_objective_overrides(
        get_dataclass(args.dataset),
        gaussian_sigma=args.gaussian_sigma,
        tolerance_scale=args.tolerance_scale,
    )
    plot_target_kernels(dataclass, outdir, args.dataset)
    if args.dataset == "sleep":
        plot_target_pipeline_column(dataclass, outdir)

    scores = load_scores(args.results_root)
    plot_summary_bars(scores, outdir, args.dataset)
    plot_tolerance_curves(scores, outdir, args.dataset)
    if args.dataset == "sleep":
        plot_sleep_main_results(scores, outdir)
        plot_sleep_prediction_example(outdir)
        plot_sleep_results_column(scores, outdir)
        plot_sleep_objective_sweep(scores, outdir)
        plot_sleep_architecture_delta(scores, outdir)
        plot_run_tag_bars(
            scores,
            outdir,
            "sleep_prior_ablation.png",
            "Sleep sparse-prior ablation",
            {
                "prior_sparse_e20_bs32_eval5": "Sparse",
                "prior_none_e20_bs32_eval5": "None",
            },
        )
        plot_run_tag_bars(
            scores,
            outdir,
            "sleep_target_width_ablation.png",
            "Sleep target-width ablation",
            {
                "tol05_e20_bs32_eval5": "0.5x",
                MAIN_RUN_TAG: "1.0x",
                "tol15_e20_bs32_eval5": "1.5x",
            },
        )
        plot_online_ablation(scores, outdir)
    if args.dataset == "seizure":
        plot_seizure_replication(scores, outdir)

    if args.prediction:
        plot_prediction_peaks(Path(args.prediction), outdir, args.max_steps)


if __name__ == "__main__":
    main()
