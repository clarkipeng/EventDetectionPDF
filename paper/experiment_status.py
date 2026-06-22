"""Build a paper experiment matrix and fill it from local run artifacts."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import pandas as pd


SEEDS_FINAL = [0, 1, 2]
CORE_MODELS = ["gru", "unet", "unet_t"]
ONLINE_MODELS = ["fgru", "flstm", "causal_transformer"]
ALL_OBJECTIVES = [
    "density_hard",
    "density_gau",
    "density_custom",
    "seg",
    "seg_weighted",
    "seg_focal",
]
OBJECTIVE_SWEEP_REMAINING = [
    objective for objective in ALL_OBJECTIVES if objective != "density_custom"
]

OBJECTIVE_LABELS = {
    "density_hard": "BDL-Hard",
    "density_gau": "BDL-Gaussian",
    "density_custom": "BDL-Tolerance",
    "seg": "Cross-entropy",
    "seg_weighted": "Weighted cross-entropy",
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
    "gru_3l_128h": "GRU 3x128",
}

RUN_LABELS = {
    "objective_e20_bs32_eval5": "Main sleep sweep",
    "lr3e3_e20_gpu_bs32_eval5": "BDL-Tolerance candidate",
    "lr3e3_e2": "Short BDL-Tolerance smoke",
    "prior_sparse_e20_bs32_eval5": "Sparse prior",
    "prior_none_e20_bs32_eval5": "No sparse prior",
    "tol05_e20_bs32_eval5": "Tolerance width 0.5x",
    "tol15_e20_bs32_eval5": "Tolerance width 1.5x",
    "online_e20_bs32_eval5": "Online-context ablation",
    "seizure_main_e20_bs32_eval5": "CHB-MIT sweep",
    "transformer_offline_e20_bs32_eval5": "Offline Transformer diagnostic",
    "online_smoothing_e20_bs32_eval5": "Online smoothing diagnostic",
    "seizure_rescue_lr3e4_ds5_bs8_e20_eval5": "CHB-MIT fine-stride sensitivity",
    "seizure_highscore_gru3l128h_ds512_bs8_e20": "CHB-MIT high-score reproduction",
    "seizure_highscore_gru3l128h_ds256_bs8_e20": "CHB-MIT high-score stride ablation",
    "transformer_main_candidate_lr1e3_e20_bs32_eval5": "Transformer main-table candidate",
    "transformer_strong_lr5e4_wd1e2_e20_bs8_eval5": "Strong Transformer candidate",
    "point_event_ablation_e20": "Point-event ablation",
}

PLAN_LABELS = {
    "lr_screen_density_custom": "Learning-rate screen",
    "current_density_candidate": "BDL-Tolerance candidate",
    "gru_objective_sweep_seed0": "GRU objective sweep",
    "core_model_main_table": "Main architecture table",
    "kernel_and_segmentation_ablation": "Kernel and segmentation ablation",
    "gaussian_kernel_architecture_ablation": "Gaussian architecture ablation",
    "prior_ablation": "Sparse-prior ablation",
    "target_width_ablation": "Target-width ablation",
    "online_model_ablation": "Online-context ablation",
    "second_dataset_replication": "CHB-MIT replication",
    "seizure_gaussian_kernel_ablation": "CHB-MIT Gaussian ablation",
    "offline_transformer_architecture_swap": "Offline Transformer diagnostic",
    "transformer_main_table_candidate": "Transformer main-table candidate",
    "transformer_strong_candidate": "Strong Transformer candidate",
    "online_smoothing_diagnostic": "Online smoothing diagnostic",
    "seizure_fine_stride_sensitivity": "CHB-MIT fine-stride sensitivity",
    "seizure_highscore_reproduction": "CHB-MIT high-score reproduction",
    "seizure_highscore_stride_ablation": "CHB-MIT high-score stride ablation",
    "bowshock_point_event_ablation": "Bowshock point-event ablation",
    "fraud_point_event_ablation": "Fraud point-event ablation",
}


PLAN_GROUPS = [
    {
        "phase": "P0",
        "name": "lr_screen_density_custom",
        "purpose": "Choose optimizer scale after switching to Poisson likelihood.",
        "dataset": ["sleep"],
        "model": ["gru"],
        "objective": ["density_custom"],
        "seed": [0],
        "run_tag": ["lr3e4_e2", "lr1e3_e2", "lr3e3_e2"],
        "epochs": 2,
        "folds": 2,
        "batch_size": 10,
        "eval_every": 1,
        "score_required": False,
    },
    {
        "phase": "P1",
        "name": "current_density_candidate",
        "purpose": "First real multi-epoch CUDA run for the main method.",
        "dataset": ["sleep"],
        "model": ["gru"],
        "objective": ["density_custom"],
        "seed": [0],
        "run_tag": ["lr3e3_e20_gpu_bs32_eval5"],
        "epochs": 20,
        "folds": 4,
        "batch_size": 32,
        "eval_every": 5,
        "score_required": True,
    },
    {
        "phase": "P2",
        "name": "gru_objective_sweep_seed0",
        "purpose": "Screen BDL kernels and segmentation losses; P1 supplies density_custom.",
        "dataset": ["sleep"],
        "model": ["gru"],
        "objective": OBJECTIVE_SWEEP_REMAINING,
        "seed": [0],
        "run_tag": ["objective_e20_bs32_eval5"],
        "epochs": 20,
        "folds": 4,
        "batch_size": 32,
        "eval_every": 5,
        "score_required": True,
    },
    {
        "phase": "P3",
        "name": "core_model_main_table",
        "purpose": "Main table: density likelihood versus strongest non-density baselines.",
        "dataset": ["sleep"],
        "model": CORE_MODELS,
        "objective": ["density_hard", "seg"],
        "seed": SEEDS_FINAL,
        "run_tag": ["objective_e20_bs32_eval5"],
        "epochs": 20,
        "folds": 4,
        "batch_size": 32,
        "eval_every": 5,
        "score_required": True,
    },
    {
        "phase": "P4",
        "name": "kernel_and_segmentation_ablation",
        "purpose": "Appendix ablations for BDL kernel choice and stronger segmentation losses.",
        "dataset": ["sleep"],
        "model": ["gru"],
        "objective": [
            "density_gau",
            "density_custom",
            "seg_weighted",
            "seg_focal",
        ],
        "seed": SEEDS_FINAL,
        "run_tag": ["objective_e20_bs32_eval5"],
        "epochs": 20,
        "folds": 4,
        "batch_size": 32,
        "eval_every": 5,
        "score_required": True,
    },
    {
        "phase": "P4G",
        "name": "gaussian_kernel_architecture_ablation",
        "purpose": "Test the Gaussian label-error target across non-GRU core architectures.",
        "dataset": ["sleep"],
        "model": ["unet", "unet_t"],
        "objective": ["density_gau"],
        "seed": SEEDS_FINAL,
        "run_tag": ["objective_e20_bs32_eval5"],
        "epochs": 20,
        "folds": 4,
        "batch_size": 32,
        "eval_every": 5,
        "score_required": True,
    },
    {
        "phase": "P5",
        "name": "prior_ablation",
        "purpose": "Measure whether sparse-intensity initialization matters.",
        "dataset": ["sleep"],
        "model": ["gru"],
        "objective": ["density_custom"],
        "seed": SEEDS_FINAL,
        "run_tag": ["prior_sparse_e20_bs32_eval5", "prior_none_e20_bs32_eval5"],
        "epochs": 20,
        "folds": 4,
        "batch_size": 32,
        "eval_every": 5,
        "score_required": True,
    },
    {
        "phase": "P6",
        "name": "target_width_ablation",
        "purpose": "Appendix sensitivity of tolerance-derived density target width.",
        "dataset": ["sleep"],
        "model": ["gru"],
        "objective": ["density_custom"],
        "seed": [0],
        "run_tag": ["tol05_e20_bs32_eval5", "tol15_e20_bs32_eval5"],
        "epochs": 20,
        "folds": 4,
        "batch_size": 32,
        "eval_every": 5,
        "score_required": True,
    },
    {
        "phase": "P7",
        "name": "online_model_ablation",
        "purpose": "Show whether the likelihood gain persists in causal/online models.",
        "dataset": ["sleep"],
        "model": ONLINE_MODELS,
        "objective": ["density_hard", "seg"],
        "seed": SEEDS_FINAL,
        "run_tag": ["online_e20_bs32_eval5"],
        "epochs": 20,
        "folds": 4,
        "batch_size": 32,
        "eval_every": 5,
        "score_required": True,
    },
    {
        "phase": "P8",
        "name": "second_dataset_replication",
        "purpose": "Replicate the main comparison on seizure once that data is available.",
        "dataset": ["seizure"],
        "model": ["gru", "unet"],
        "objective": ["density_hard", "seg"],
        "seed": [0],
        "run_tag": ["seizure_main_e20_bs32_eval5"],
        "epochs": 20,
        "folds": 4,
        "batch_size": 32,
        "eval_every": 5,
        "score_required": True,
    },
    {
        "phase": "P8G",
        "name": "seizure_gaussian_kernel_ablation",
        "purpose": "Check whether Gaussian label-error smoothing transfers to the seizure benchmark.",
        "dataset": ["seizure"],
        "model": ["gru", "unet"],
        "objective": ["density_gau"],
        "seed": [0],
        "run_tag": ["seizure_main_e20_bs32_eval5"],
        "epochs": 20,
        "folds": 4,
        "batch_size": 32,
        "eval_every": 5,
        "score_required": True,
    },
    {
        "phase": "P9",
        "name": "offline_transformer_architecture_swap",
        "purpose": "Optional architecture check: compare BDL and segmentation on a matched offline Transformer.",
        "dataset": ["sleep"],
        "model": ["transformer_4l_64h_4a"],
        "objective": ["density_hard", "seg"],
        "seed": SEEDS_FINAL,
        "run_tag": ["transformer_offline_e20_bs32_eval5"],
        "epochs": 20,
        "folds": 4,
        "batch_size": 32,
        "eval_every": 5,
        "score_required": True,
    },
    {
        "phase": "P10",
        "name": "online_smoothing_diagnostic",
        "purpose": "Optional causal-context check: test whether smoothed BDL targets help online models.",
        "dataset": ["sleep"],
        "model": ONLINE_MODELS,
        "objective": ["density_gau", "density_custom", "seg"],
        "seed": [0],
        "run_tag": ["online_smoothing_e20_bs32_eval5"],
        "epochs": 20,
        "folds": 4,
        "batch_size": 32,
        "eval_every": 5,
        "score_required": True,
    },
    {
        "phase": "P15",
        "name": "transformer_main_table_candidate",
        "purpose": "Tuned offline Transformer candidate for replacing the attention-gated U-Net architecture control if the scored comparison is defensible.",
        "dataset": ["sleep"],
        "model": ["transformer_4l_64h_4a"],
        "objective": ["density_gau", "density_custom", "seg"],
        "seed": [0],
        "run_tag": ["transformer_main_candidate_lr1e3_e20_bs32_eval5"],
        "epochs": 20,
        "folds": 4,
        "batch_size": 32,
        "eval_every": 5,
        "score_required": True,
    },
    {
        "phase": "P16",
        "name": "transformer_strong_candidate",
        "purpose": "Stronger offline Transformer candidate for replacing the attention-gated U-Net architecture control if matched objective scores are competitive.",
        "dataset": ["sleep"],
        "model": ["transformer_6l_128h_8a_10d"],
        "objective": ["density_hard", "density_gau", "density_custom", "seg"],
        "seed": [0],
        "run_tag": ["transformer_strong_lr5e4_wd1e2_e20_bs8_eval5"],
        "epochs": 20,
        "folds": 4,
        "batch_size": 8,
        "eval_every": 5,
        "score_required": True,
    },
    {
        "phase": "P11",
        "name": "seizure_fine_stride_sensitivity",
        "purpose": "Optional CHB-MIT sensitivity check with lower learning rate and finer temporal downsampling.",
        "dataset": ["seizure"],
        "model": ["gru", "unet"],
        "objective": ["density_hard", "density_gau", "seg"],
        "seed": [0],
        "run_tag": ["seizure_rescue_lr3e4_ds5_bs8_e20_eval5"],
        "epochs": 20,
        "folds": 4,
        "batch_size": 8,
        "downsample": 5,
        "eval_every": 5,
        "score_required": False,
    },
    {
        "phase": "P12",
        "name": "seizure_highscore_reproduction",
        "purpose": "Reproduce the high-scoring CHB-MIT segmentation setting identified during development and compare matched BDL objectives.",
        "dataset": ["seizure"],
        "model": ["gru_3l_128h"],
        "objective": ["seg", "density_hard", "density_gau", "density_custom"],
        "seed": [0],
        "run_tag": ["seizure_highscore_gru3l128h_ds512_bs8_e20"],
        "epochs": 20,
        "folds": 4,
        "batch_size": 8,
        "downsample": 512,
        "eval_every": 1,
        "score_required": True,
    },
    {
        "phase": "P12b",
        "name": "seizure_highscore_stride_ablation",
        "purpose": "Test whether a one-second output stride improves the high-score CHB-MIT configuration enough to offset longer recurrent sequences.",
        "dataset": ["seizure"],
        "model": ["gru_3l_128h"],
        "objective": ["seg", "density_hard", "density_gau", "density_custom"],
        "seed": [0],
        "run_tag": ["seizure_highscore_gru3l128h_ds256_bs8_e20"],
        "epochs": 20,
        "folds": 4,
        "batch_size": 8,
        "downsample": 256,
        "eval_every": 1,
        "score_required": True,
    },
    {
        "phase": "P13",
        "name": "bowshock_point_event_ablation",
        "purpose": "Appendix point-event ablation on the Martian bow-shock detection benchmark.",
        "dataset": ["bowshock"],
        "model": ["gru"],
        "objective": ["density_hard", "density_gau", "density_custom", "seg"],
        "seed": [0],
        "run_tag": ["point_event_ablation_e20"],
        "epochs": 20,
        "folds": 4,
        "batch_size": 32,
        "downsample": 10,
        "eval_every": 5,
        "score_required": True,
    },
    {
        "phase": "P14",
        "name": "fraud_point_event_ablation",
        "purpose": "Appendix point-event ablation on the credit-card fraud detection benchmark.",
        "dataset": ["fraud"],
        "model": ["gru"],
        "objective": ["density_hard", "density_gau", "density_custom", "seg"],
        "seed": [0],
        "run_tag": ["point_event_ablation_e20"],
        "epochs": 20,
        "folds": 4,
        "batch_size": 32,
        "downsample": 1,
        "eval_every": 5,
        "score_required": True,
    },
]


def expand_plan():
    rows = []
    for group in PLAN_GROUPS:
        keys = ["dataset", "model", "objective", "seed", "run_tag"]
        for values in itertools.product(*[group[key] for key in keys]):
            row = {key: value for key, value in zip(keys, values)}
            row.update(
                {
                    "phase": group["phase"],
                    "name": group["name"],
                    "purpose": group["purpose"],
                    "target_epochs": group["epochs"],
                    "target_folds": group["folds"],
                    "target_batch_size": group["batch_size"],
                    "target_downsample": group.get("downsample", 10),
                    "target_eval_every": group["eval_every"],
                    "score_required": group["score_required"],
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def read_json(path):
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def is_quarantined(path):
    return "quarantine" in Path(path).parts


def load_observed_runs(results_root):
    rows = []
    for config_path in sorted(Path(results_root).glob("**/results/run_config.json")):
        if is_quarantined(config_path):
            continue
        run_dir = config_path.parents[1]
        config = read_json(config_path)
        fold_path = config_path.parent / "fold_results.csv"
        epoch_path = config_path.parent / "epoch_results.csv"
        score_path = config_path.parent / "scores.csv"

        fold_count = 0
        mean_valid_mAP = None
        mean_valid_loss = None
        mean_best_epoch = None
        if fold_path.exists():
            folds = pd.read_csv(fold_path)
            fold_count = int(len(folds))
            mean_valid_mAP = pd.to_numeric(
                folds.get("best_valid_mAP"), errors="coerce"
            ).mean()
            mean_valid_loss = pd.to_numeric(
                folds.get("best_valid_loss"), errors="coerce"
            ).mean()
            mean_best_epoch = pd.to_numeric(
                folds.get("best_epoch"), errors="coerce"
            ).mean()

        epoch_count = 0
        latest_fold = None
        latest_epoch = None
        latest_epoch_total = None
        latest_train_loss = None
        latest_valid_mAP = None
        latest_eval_fold = None
        latest_eval_epoch = None
        latest_eval_valid_mAP = None
        best_eval_fold = None
        best_eval_epoch = None
        best_eval_valid_mAP = None
        avg_epoch_runtime_sec = None
        avg_eval_epoch_runtime_sec = None
        avg_train_epoch_runtime_sec = None
        estimated_current_fold_remaining_min = None
        if epoch_path.exists():
            epochs = pd.read_csv(epoch_path)
            epoch_count = int(len(epochs))
            if not epochs.empty:
                latest = epochs.tail(1).iloc[0]
                latest_fold = latest.get("fold")
                latest_epoch = latest.get("epoch")
                latest_epoch_total = latest.get("epochs")
                latest_train_loss = latest.get("train_loss")
                latest_valid_mAP = latest.get("valid_mAP")
                if "epoch_runtime_sec" in epochs:
                    runtime_sec = pd.to_numeric(
                        epochs["epoch_runtime_sec"], errors="coerce"
                    )
                else:
                    runtime_sec = pd.Series(float("nan"), index=epochs.index)
                avg_epoch_runtime_sec = runtime_sec.mean()
                if "evaluated" in epochs:
                    evaluated = (
                        epochs["evaluated"].astype(str).str.lower().isin(["true", "1"])
                    )
                elif "valid_mAP" in epochs:
                    evaluated = pd.to_numeric(
                        epochs["valid_mAP"], errors="coerce"
                    ).notna()
                else:
                    evaluated = pd.Series(False, index=epochs.index)
                avg_eval_epoch_runtime_sec = runtime_sec[evaluated].mean()
                avg_train_epoch_runtime_sec = runtime_sec[~evaluated].mean()
                eval_every = config.get("eval_every")
                try:
                    start_epoch = int(float(latest_epoch))
                    total_epochs = int(float(latest_epoch_total or config.get("epochs")))
                    eval_every_int = int(float(eval_every)) if eval_every else None
                except (TypeError, ValueError):
                    start_epoch = None
                    total_epochs = None
                    eval_every_int = None
                if (
                    start_epoch is not None
                    and total_epochs is not None
                    and start_epoch < total_epochs
                    and pd.notna(avg_epoch_runtime_sec)
                ):
                    remaining_sec = 0.0
                    for future_epoch in range(start_epoch + 1, total_epochs + 1):
                        will_eval = (
                            bool(eval_every_int)
                            and eval_every_int > 0
                            and future_epoch % eval_every_int == 0
                        )
                        if will_eval and pd.notna(avg_eval_epoch_runtime_sec):
                            remaining_sec += float(avg_eval_epoch_runtime_sec)
                        elif pd.notna(avg_train_epoch_runtime_sec):
                            remaining_sec += float(avg_train_epoch_runtime_sec)
                        else:
                            remaining_sec += float(avg_epoch_runtime_sec)
                    estimated_current_fold_remaining_min = remaining_sec / 60.0
                if "valid_mAP" in epochs:
                    eval_epochs = epochs.copy()
                    eval_epochs["valid_mAP_numeric"] = pd.to_numeric(
                        eval_epochs["valid_mAP"], errors="coerce"
                    )
                    eval_epochs = eval_epochs.dropna(subset=["valid_mAP_numeric"])
                    if not eval_epochs.empty:
                        latest_eval = eval_epochs.tail(1).iloc[0]
                        latest_eval_fold = latest_eval.get("fold")
                        latest_eval_epoch = latest_eval.get("epoch")
                        latest_eval_valid_mAP = latest_eval.get("valid_mAP_numeric")
                        best_eval = eval_epochs.sort_values(
                            "valid_mAP_numeric", ascending=False
                        ).iloc[0]
                        best_eval_fold = best_eval.get("fold")
                        best_eval_epoch = best_eval.get("epoch")
                        best_eval_valid_mAP = best_eval.get("valid_mAP_numeric")

        tuned_mAP, tuned_mf1 = read_tuned_score_metrics(score_path)
        strict3_path = config_path.parent / "scores_strict3.csv"
        strict3_tuned_mAP, strict3_tuned_mf1 = read_tuned_score_metrics(strict3_path)

        rows.append(
            {
                "dataset": config.get("dataset"),
                "model": config.get("model"),
                "objective": config.get("objective"),
                "seed": config.get("seed"),
                "run_tag": config.get("run_tag"),
                "actual_epochs": config.get("epochs"),
                "actual_folds": config.get("folds"),
                "actual_batch_size": config.get("batch_size"),
                "actual_downsample": config.get("downsample"),
                "actual_eval_every": config.get("eval_every"),
                "lr": config.get("lr"),
                "density_prior": config.get("density_prior"),
                "score_after_train": config.get("score_after_train"),
                "completed_folds": fold_count,
                "recorded_epochs": epoch_count,
                "latest_fold": latest_fold,
                "latest_epoch": latest_epoch,
                "latest_epoch_total": latest_epoch_total,
                "latest_train_loss": latest_train_loss,
                "latest_valid_mAP": latest_valid_mAP,
                "latest_eval_fold": latest_eval_fold,
                "latest_eval_epoch": latest_eval_epoch,
                "latest_eval_valid_mAP": latest_eval_valid_mAP,
                "best_eval_fold": best_eval_fold,
                "best_eval_epoch": best_eval_epoch,
                "best_eval_valid_mAP": best_eval_valid_mAP,
                "avg_epoch_runtime_sec": avg_epoch_runtime_sec,
                "avg_eval_epoch_runtime_sec": avg_eval_epoch_runtime_sec,
                "avg_train_epoch_runtime_sec": avg_train_epoch_runtime_sec,
                "estimated_current_fold_remaining_min": estimated_current_fold_remaining_min,
                "mean_valid_mAP": mean_valid_mAP,
                "mean_valid_loss": mean_valid_loss,
                "mean_best_epoch": mean_best_epoch,
                "tuned_mAP": tuned_mAP,
                "tuned_mf1": tuned_mf1,
                "strict3_tuned_mAP": strict3_tuned_mAP,
                "strict3_tuned_mf1": strict3_tuned_mf1,
                "strict3_score_file": str(strict3_path) if strict3_path.exists() else "",
                "run_dir": str(run_dir),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def read_tuned_score_metrics(score_path):
    if not score_path.exists():
        return None, None
    scores = pd.read_csv(score_path)
    tuned = scores[
        (scores["row_type"] == "summary")
        & (scores["stage"] == "tuned")
        & (scores["optimized_metric"] == "mAP")
    ].copy()
    if tuned.empty:
        return None, None
    tuned["score"] = pd.to_numeric(tuned["score"], errors="coerce")
    mAP = tuned[tuned["metric"] == "mAP"]
    mf1 = tuned[tuned["metric"] == "mf1"]
    tuned_mAP = float(mAP["score"].max()) if not mAP.empty else None
    tuned_mf1 = float(mf1["score"].max()) if not mf1.empty else None
    return tuned_mAP, tuned_mf1


def status_for(row):
    if pd.notna(row.get("tuned_mAP")):
        return "scored"
    if row.get("completed_folds", 0) >= row.get("target_folds", 0):
        return "trained"
    if row.get("completed_folds", 0) > 0:
        return "partial"
    if pd.notna(row.get("run_dir")):
        return "started"
    return "pending"


def phase_sort_key(phase):
    text = str(phase)
    if text.startswith("P"):
        suffix = text[1:]
        digits = ""
        rest = ""
        for char in suffix:
            if char.isdigit() and not rest:
                digits += char
            else:
                rest += char
        if digits:
            return int(digits), rest
    return 10_000, text


def merge_status(plan, observed):
    if observed.empty:
        out = plan.copy()
        out["status"] = "pending"
        return out

    keys = ["dataset", "model", "objective", "seed", "run_tag"]
    merged = plan.merge(observed, on=keys, how="left")
    merged["status"] = merged.apply(status_for, axis=1)
    ordered = [
        "phase",
        "name",
        "status",
        "dataset",
        "model",
        "objective",
        "seed",
        "run_tag",
        "target_epochs",
        "target_folds",
        "target_batch_size",
        "target_downsample",
        "target_eval_every",
        "completed_folds",
        "recorded_epochs",
        "latest_fold",
        "latest_epoch",
        "latest_epoch_total",
        "latest_train_loss",
        "latest_valid_mAP",
        "latest_eval_fold",
        "latest_eval_epoch",
        "latest_eval_valid_mAP",
        "best_eval_fold",
        "best_eval_epoch",
        "best_eval_valid_mAP",
        "avg_epoch_runtime_sec",
        "avg_eval_epoch_runtime_sec",
        "avg_train_epoch_runtime_sec",
        "estimated_current_fold_remaining_min",
        "actual_epochs",
        "actual_batch_size",
        "actual_downsample",
        "actual_eval_every",
        "lr",
        "density_prior",
        "mean_best_epoch",
        "mean_valid_mAP",
        "mean_valid_loss",
        "tuned_mAP",
        "tuned_mf1",
        "strict3_tuned_mAP",
        "strict3_tuned_mf1",
        "strict3_score_file",
        "score_required",
        "purpose",
        "run_dir",
    ]
    return merged.loc[:, ordered]


def write_text_summary(status, path):
    def markdown_table(frame, floatfmt=".3f"):
        if frame.empty:
            return ""
        columns = list(frame.columns)
        rows = []
        for _, row in frame.iterrows():
            values = []
            for column in columns:
                value = row[column]
                if pd.isna(value):
                    values.append("")
                elif isinstance(value, float):
                    values.append(format(value, floatfmt))
                else:
                    values.append(str(value))
            rows.append(values)
        lines = [
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join(["---"] * len(columns)) + " |",
        ]
        lines.extend("| " + " | ".join(row) + " |" for row in rows)
        return "\n".join(lines)

    counts = (
        status.groupby(["phase", "status"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    counts["_phase_order"] = counts["phase"].map(phase_sort_key)
    counts = counts.sort_values("_phase_order").drop(columns=["_phase_order"])
    lines = [
        "Experiment Status",
        "",
        "Generated from `paper/experiment_status.py`.",
        "Run directories under `experiments/quarantine/` are excluded.",
        "",
        "Phase Counts",
        "",
        markdown_table(counts),
        "",
        "Best Observed Runs",
        "",
    ]
    observed = status.dropna(subset=["mean_valid_mAP"]).copy()
    if observed.empty:
        lines.append("No completed fold metrics yet.")
    else:
        display = observed.copy()
        display["name"] = display["name"].map(PLAN_LABELS).fillna(display["name"])
        display["model"] = display["model"].map(MODEL_LABELS).fillna(display["model"])
        display["objective"] = (
            display["objective"].map(OBJECTIVE_LABELS).fillna(display["objective"])
        )
        display["run_tag"] = display["run_tag"].map(RUN_LABELS).fillna(display["run_tag"])
        cols = [
            "phase",
            "name",
            "status",
            "model",
            "objective",
            "seed",
            "run_tag",
            "mean_valid_mAP",
            "tuned_mAP",
        ]
        top = display.sort_values("mean_valid_mAP", ascending=False).head(12)
        lines.append(markdown_table(top.loc[:, cols]))

    active = status[status["status"].isin(["started", "partial"])].copy()
    if not active.empty:
        active["name"] = active["name"].map(PLAN_LABELS).fillna(active["name"])
        active["model"] = active["model"].map(MODEL_LABELS).fillna(active["model"])
        active["objective"] = (
            active["objective"].map(OBJECTIVE_LABELS).fillna(active["objective"])
        )
        active["run_tag"] = active["run_tag"].map(RUN_LABELS).fillna(active["run_tag"])
        active_cols = [
            "phase",
            "name",
            "status",
            "model",
            "objective",
            "seed",
            "run_tag",
            "completed_folds",
            "target_folds",
            "latest_fold",
            "latest_epoch",
            "latest_epoch_total",
            "latest_eval_epoch",
            "latest_eval_valid_mAP",
            "best_eval_epoch",
            "best_eval_valid_mAP",
            "estimated_current_fold_remaining_min",
        ]
        active = active.sort_values(["phase", "name", "model", "objective"])
        lines.extend(
            [
                "",
                "Active Rows",
                "",
                "Rows here have local run artifacts but are not yet complete.",
                "",
                markdown_table(active.loc[:, active_cols]),
            ]
        )

    incomplete = status[
        (status["score_required"].fillna(False).astype(bool))
        & (status["status"] != "scored")
    ].copy()
    if not incomplete.empty:
        incomplete["name"] = incomplete["name"].map(PLAN_LABELS).fillna(
            incomplete["name"]
        )
        incomplete["model"] = incomplete["model"].map(MODEL_LABELS).fillna(
            incomplete["model"]
        )
        incomplete["objective"] = (
            incomplete["objective"].map(OBJECTIVE_LABELS).fillna(incomplete["objective"])
        )
        incomplete["run_tag"] = incomplete["run_tag"].map(RUN_LABELS).fillna(
            incomplete["run_tag"]
        )
        incomplete_cols = [
            "phase",
            "name",
            "status",
            "model",
            "objective",
            "seed",
            "run_tag",
            "completed_folds",
            "target_folds",
            "latest_fold",
            "latest_epoch",
            "latest_epoch_total",
            "latest_valid_mAP",
            "latest_eval_epoch",
            "latest_eval_valid_mAP",
            "best_eval_epoch",
            "best_eval_valid_mAP",
        ]
        incomplete = incomplete.sort_values(["phase", "name", "model", "objective"])
        lines.extend(
            [
                "",
                "Incomplete Rows",
                "",
                "Rows here still need training folds or score files before they can support manuscript tables.",
                "",
                markdown_table(incomplete.loc[:, incomplete_cols]),
            ]
        )
    strict = status[
        status["run_tag"].isin(
            [
                "seizure_highscore_gru3l128h_ds512_bs8_e20",
                "seizure_highscore_gru3l128h_ds256_bs8_e20",
            ]
        )
    ].copy()
    if not strict.empty:
        strict["strict3_status"] = strict["strict3_tuned_mAP"].apply(
            lambda value: "scored" if pd.notna(value) else "pending"
        )
        strict["model"] = strict["model"].map(MODEL_LABELS).fillna(strict["model"])
        strict["objective"] = (
            strict["objective"].map(OBJECTIVE_LABELS).fillna(strict["objective"])
        )
        strict["run_tag"] = strict["run_tag"].map(RUN_LABELS).fillna(strict["run_tag"])
        strict_cols = [
            "phase",
            "model",
            "objective",
            "run_tag",
            "status",
            "strict3_status",
            "tuned_mAP",
            "strict3_tuned_mAP",
        ]
        strict = strict.sort_values(["phase", "run_tag", "objective"])
        lines.extend(
            [
                "",
                "Strict CHB-MIT Rescoring",
                "",
                "Strict rows use one-to-three-second boundary tolerances and are stored in `scores_strict3.csv`.",
                "",
                markdown_table(strict.loc[:, strict_cols]),
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default="experiments")
    parser.add_argument("--outdir", default="paper/results/generated")
    return parser.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    plan = expand_plan()
    observed = load_observed_runs(args.results_root)
    status = merge_status(plan, observed)
    plan.to_csv(outdir / "experiment_plan.csv", index=False)
    observed.to_csv(outdir / "observed_runs.csv", index=False)
    status.to_csv(outdir / "experiment_status.csv", index=False)
    write_text_summary(status, outdir / "experiment_status.txt")
    print(f"Planned experiment rows: {len(plan)}")
    print(f"Observed run rows: {len(observed)}")
    print(f"Wrote status tables to {outdir}")


if __name__ == "__main__":
    main()
