"""Collect experiment artifacts into paper-ready result tables.

The training/evaluation code writes one `scores.csv` and one `fold_results.csv`
per run under `experiments/`. This script consolidates those files into a
small set of CSV and LaTeX tables that can be regenerated after each batch of
GPU jobs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from paired_fold_analysis import write_paired_fold_outputs


OBJECTIVE_ORDER = [
    "density_hard",
    "density_custom",
    "density_gau",
    "seg",
    "seg_weighted",
    "seg_focal",
]

SLEEP_OBJECTIVE_SWEEP_OBJECTIVES = {
    "density_hard",
    "density_gau",
    "density_custom",
    "seg",
    "seg_weighted",
    "seg_focal",
}

OBJECTIVE_FAMILIES = {
    "density_hard": "BDL",
    "density_gau": "BDL",
    "density_custom": "BDL",
    "seg": "Segmentation",
    "seg_weighted": "Segmentation",
    "seg_focal": "Segmentation",
}

OBJECTIVE_LABELS = {
    "density_hard": "BDL-Hard",
    "density_gau": "BDL-Gaussian",
    "density_custom": "BDL-Tolerance",
    "seg": "Cross-entropy",
    "seg_weighted": "Weighted cross-entropy",
    "seg_focal": "Focal loss",
}

POSTPROCESS_LABELS = {
    "density_hard": "Event-score peaks",
    "density_gau": "Event-score peaks",
    "density_custom": "Event-score peaks",
    "seg": "Segmentation transitions",
    "seg1": "Segmentation transitions",
    "seg2": "Segmentation transitions",
    "seg_weighted": "Weighted transitions",
    "seg_weighted1": "Weighted transitions",
    "seg_weighted2": "Weighted transitions",
    "seg_focal": "Focal transitions",
    "seg_focal1": "Focal transitions",
    "seg_focal2": "Focal transitions",
}

RUN_LABELS = {
    "objective_e20_bs32_eval5": "Main sleep sweep",
    "lr3e3_e20_gpu_bs32_eval5": "BDL-Tolerance candidate",
    "lr3e3_e2": "Short BDL-Tolerance smoke",
    "seizure_main_e20_bs32_eval5": "CHB-MIT sweep",
    "seizure_highscore_gru3l128h_ds512_bs8_e20": "High-capacity GRU, 2 s stride",
    "seizure_highscore_gru3l128h_ds256_bs8_e20": "High-capacity GRU, 1 s stride",
    "online_e20_bs32_eval5": "Online-context ablation",
    "prior_none_e20_bs32_eval5": "No sparse prior",
    "prior_sparse_e20_bs32_eval5": "Sparse prior",
    "tol05_e20_bs32_eval5": "Tolerance width 0.5x",
    "tol15_e20_bs32_eval5": "Tolerance width 1.5x",
    "transformer_offline_e20_bs32_eval5": "Offline Transformer diagnostic",
    "transformer_main_candidate_lr1e3_e20_bs32_eval5": "Transformer main-table candidate",
    "transformer_strong_lr5e4_wd1e2_e20_bs8_eval5": "Strong Transformer candidate",
    "point_event_ablation_e20": "Point-event ablation",
}

DATASET_LABELS = {
    "sleep": "Sleep",
    "seizure": "CHB-MIT",
    "bowshock": "Bowshock",
    "fraud": "Fraud",
}

MODEL_LABELS = {
    "gru": "GRU",
    "bigru": "BiGRU",
    "unet": "U-Net",
    "unet_t": "Attention-gated U-Net",
    "unet_atten": "Attention-gated U-Net",
    "fgru": "Forward GRU",
    "flstm": "Forward LSTM",
    "causal_transformer": "Causal Transformer",
    "gru_3l_128h": "GRU 3x128",
    "transformer_4l_64h_4a": "Offline Transformer",
    "transformer_6l_128h_8a_10d": "Offline Transformer",
    "mamba": "Mamba",
    "transformer": "Offline Transformer",
}

DISPLAY_COLUMNS = {
    "dataset": "Dataset",
    "model": "Model",
    "family": "Family",
    "method": "Method",
    "post-processing": "Candidate decoder",
    "objective": "Objective",
    "postprocess_objective": "Candidate decoder",
    "run_tag": "Run",
    "setting": "Setting",
    "width": "Target width",
    "mAP": "mAP",
    "mf1": "F1",
    "mean_std": "mAP",
    "count": "Runs",
    "seeds": "Runs",
}

SLEEP_SWEEP_TAGS = {"objective_e20_bs32_eval5", "lr3e3_e20_gpu_bs32_eval5"}
MAIN_RUN_TAG = "objective_e20_bs32_eval5"
SEIZURE_RUN_TAG = "seizure_main_e20_bs32_eval5"
SEIZURE_HIGH_SCORE_TAG = "seizure_highscore_gru3l128h_ds512_bs8_e20"
SEIZURE_HIGH_SCORE_DS256_TAG = "seizure_highscore_gru3l128h_ds256_bs8_e20"
ONLINE_RUN_TAG = "online_e20_bs32_eval5"
ONLINE_SMOOTHING_RUN_TAG = "online_smoothing_e20_bs32_eval5"
TRANSFORMER_MAIN_CANDIDATE_TAG = "transformer_main_candidate_lr1e3_e20_bs32_eval5"
TRANSFORMER_STRONG_CANDIDATE_TAG = "transformer_strong_lr5e4_wd1e2_e20_bs8_eval5"
POINT_EVENT_RUN_TAG = "point_event_ablation_e20"


def is_quarantined(path):
    return "quarantine" in Path(path).parts


def read_csvs(paths):
    frames = []
    for path in paths:
        if is_quarantined(path):
            continue
        frame = pd.read_csv(path)
        frame["source_path"] = str(path)
        add_path_metadata(frame, path)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def parse_seed_dir(seed_dir):
    if not seed_dir.startswith("seed_"):
        return None, ""
    remainder = seed_dir.removeprefix("seed_")
    seed_token, sep, run_tag = remainder.partition("_")
    try:
        seed = int(seed_token)
    except ValueError:
        seed = None
    return seed, run_tag if sep else ""


def add_path_metadata(frame, path):
    path = Path(path)
    run_dir = path.parents[1]
    seed, path_run_tag = parse_seed_dir(run_dir.name)
    path_objective = run_dir.parent.name
    path_model = run_dir.parent.parent.name
    path_dataset = run_dir.parent.parent.parent.name

    if "dataset" not in frame:
        frame["dataset"] = path_dataset
    else:
        frame["dataset"] = frame["dataset"].fillna(path_dataset)
    if "model" not in frame:
        frame["model"] = path_model
    else:
        frame["model"] = frame["model"].fillna(path_model)
    if "objective" not in frame:
        frame["objective"] = path_objective
    else:
        frame["objective"] = frame["objective"].fillna(path_objective)
    if "seed" not in frame:
        frame["seed"] = seed
    else:
        frame["seed"] = frame["seed"].fillna(seed)
    if "run_tag" not in frame:
        frame["run_tag"] = path_run_tag
    else:
        frame["run_tag"] = frame["run_tag"].fillna(path_run_tag)
        frame.loc[frame["run_tag"].astype(str) == "None", "run_tag"] = path_run_tag
    frame["run_tag"] = frame["run_tag"].fillna("").astype(str)
    frame["run_id"] = run_dir.name
    frame["run_dir"] = str(run_dir)
    if path.name.startswith("scores_") and path.name.endswith(".csv"):
        score_suffix = path.name.removeprefix("scores_").removesuffix(".csv")
    else:
        score_suffix = ""
    if "score_suffix" not in frame:
        frame["score_suffix"] = score_suffix
    else:
        frame["score_suffix"] = frame["score_suffix"].fillna(score_suffix)
        frame.loc[frame["score_suffix"].astype(str) == "None", "score_suffix"] = score_suffix
    frame["score_suffix"] = frame["score_suffix"].fillna("").astype(str)
    frame["score_file"] = path.name


def load_scores(results_root):
    return read_csvs(sorted(Path(results_root).glob("**/results/scores*.csv")))


def load_fold_results(results_root):
    return read_csvs(sorted(Path(results_root).glob("**/results/fold_results.csv")))


def default_score_rows(scores):
    if scores.empty or "score_suffix" not in scores:
        return scores
    return scores[scores["score_suffix"].fillna("").astype(str) == ""].copy()


def suffixed_score_rows(scores, suffix):
    if scores.empty or "score_suffix" not in scores:
        return pd.DataFrame()
    return scores[scores["score_suffix"].fillna("").astype(str) == suffix].copy()


def objective_sort_key(objective):
    try:
        return OBJECTIVE_ORDER.index(objective)
    except ValueError:
        return len(OBJECTIVE_ORDER), objective


def format_mean_std(mean, std, count):
    if pd.isna(mean):
        return ""
    return f"{mean:.3f}"


def tuned_summary(scores):
    if scores.empty:
        return pd.DataFrame()
    frame = scores[
        (scores["row_type"] == "summary")
        & (scores["stage"] == "tuned")
        & (scores["optimized_metric"] == "mAP")
        & (scores["metric"].isin(["mAP", "mf1", "maxf1"]))
    ].copy()
    frame = frame[frame["objective"].isin(OBJECTIVE_ORDER)]
    if frame.empty:
        return frame
    frame["score"] = pd.to_numeric(frame["score"], errors="coerce")
    grouped = (
        frame.groupby(
            [
                "dataset",
                "model",
                "objective",
                "run_tag",
                "postprocess_objective",
                "metric",
            ],
            dropna=False,
        )["score"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    grouped["objective_rank"] = grouped["objective"].map(objective_sort_key)
    grouped = grouped.sort_values(
        ["dataset", "model", "run_tag", "metric", "objective_rank"]
    )
    grouped = grouped.drop(columns=["objective_rank"])
    grouped["mean_std"] = [
        format_mean_std(row.mean, row.std, int(row.count))
        for row in grouped.itertuples()
    ]
    return grouped


def tolerance_summary(scores):
    if scores.empty:
        return pd.DataFrame()
    frame = scores[
        (scores["row_type"] == "tolerance")
        & (scores["stage"] == "tuned")
        & (scores["optimized_metric"] == "mAP")
        & (scores["metric"] == "mAP")
    ].copy()
    frame = frame[frame["objective"].isin(OBJECTIVE_ORDER)]
    if frame.empty:
        return frame
    frame["score"] = pd.to_numeric(frame["score"], errors="coerce")
    frame["tolerance"] = pd.to_numeric(frame["tolerance"], errors="coerce")
    return (
        frame.groupby(
            [
                "dataset",
                "model",
                "objective",
                "run_tag",
                "postprocess_objective",
                "tolerance",
            ],
            dropna=False,
        )["score"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values(["dataset", "model", "run_tag", "objective", "tolerance"])
    )


def best_sleep_objective_sweep(summary):
    if summary.empty:
        return pd.DataFrame()
    frame = summary[
        (summary["dataset"] == "sleep")
        & (summary["model"] == "gru")
        & (summary["run_tag"].isin(SLEEP_SWEEP_TAGS))
        & (summary["metric"] == "mAP")
        & (summary["objective"].isin(SLEEP_OBJECTIVE_SWEEP_OBJECTIVES))
    ].copy()
    if frame.empty:
        return frame

    metric_table = frame.pivot_table(
        index=[
            "dataset",
            "model",
            "objective",
            "run_tag",
            "postprocess_objective",
            "count",
        ],
        columns="metric",
        values="mean",
        aggfunc="first",
    ).reset_index()
    if "mAP" not in metric_table:
        return pd.DataFrame()

    metric_table["family"] = metric_table["objective"].map(OBJECTIVE_FAMILIES).fillna("Other")
    metric_table["objective_rank"] = metric_table["objective"].map(objective_sort_key)
    metric_table = metric_table.sort_values(
        ["dataset", "model", "objective", "mAP"],
        ascending=[True, True, True, False],
    )
    best = metric_table.drop_duplicates(["dataset", "model", "objective"], keep="first")
    best = best.sort_values("mAP", ascending=False)
    best["mAP"] = best["mAP"].round(3)
    best["method"] = best["objective"].map(OBJECTIVE_LABELS).fillna(best["objective"])
    best["post-processing"] = (
        best["postprocess_objective"]
        .map(POSTPROCESS_LABELS)
        .fillna(best["postprocess_objective"])
    )
    best["seeds"] = best["count"].astype(int)
    return best[
        [
            "family",
            "method",
            "post-processing",
            "objective",
            "postprocess_objective",
            "run_tag",
            "mAP",
            "seeds",
        ]
    ]


def fold_summary(folds):
    if folds.empty:
        return pd.DataFrame()
    frame = folds.copy()
    metric_cols = ["best_valid_loss", "best_valid_mAP", "fold_runtime_sec"]
    for col in metric_cols:
        if col in frame:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    grouped = (
        frame.groupby(["dataset", "model", "objective", "run_tag", "seed"], dropna=False)
        .agg(
            folds=("fold", "count"),
            mean_best_epoch=("best_epoch", "mean"),
            mean_valid_loss=("best_valid_loss", "mean"),
            mean_valid_mAP=("best_valid_mAP", "mean"),
            total_runtime_sec=("fold_runtime_sec", "sum"),
        )
        .reset_index()
        .sort_values(["dataset", "model", "objective", "run_tag", "seed"])
    )
    return grouped


def best_metric_rows(
    summary,
    *,
    dataset=None,
    models=None,
    objectives=None,
    run_tags=None,
    metric="mAP",
    group_cols=None,
):
    if summary.empty:
        return pd.DataFrame()
    frame = summary[summary["metric"] == metric].copy()
    if dataset is not None:
        frame = frame[frame["dataset"] == dataset]
    if models is not None:
        frame = frame[frame["model"].isin(models)]
    if objectives is not None:
        frame = frame[frame["objective"].isin(objectives)]
    if run_tags is not None:
        frame = frame[frame["run_tag"].isin(run_tags)]
    if frame.empty:
        return frame
    frame["mean"] = pd.to_numeric(frame["mean"], errors="coerce")
    if group_cols is None:
        group_cols = ["dataset", "model", "objective"]
    return (
        frame.sort_values("mean", ascending=False)
        .drop_duplicates(group_cols, keep="first")
        .sort_values(group_cols)
    )


def table_cell(frame, **selectors):
    if frame.empty:
        return "--", None, None
    rows = frame.copy()
    for key, value in selectors.items():
        rows = rows[rows[key] == value]
    if rows.empty:
        return "--", None, None
    row = rows.iloc[0]
    return row["mean_std"], float(row["mean"]), int(row["count"])


def format_delta(lhs, rhs):
    if lhs is None or rhs is None:
        return "--"
    return f"{lhs - rhs:+.3f}"


def architecture_row(best, model):
    bdl_cell, bdl_mean, bdl_count = table_cell(best, model=model, objective="density_hard")
    seg_cell, seg_mean, seg_count = table_cell(best, model=model, objective="seg")
    counts = [count for count in [bdl_count, seg_count] if count is not None]
    return {
        "model": model,
        "BDL-Hard": bdl_cell,
        "Cross-entropy": seg_cell,
        "$\\Delta$": format_delta(bdl_mean, seg_mean),
        "seeds": min(counts) if counts else "--",
    }, bdl_mean, seg_mean


def sleep_main_architecture(summary):
    models = ["gru", "unet", "unet_t"]
    run_tags = [MAIN_RUN_TAG]
    best = best_metric_rows(
        summary,
        dataset="sleep",
        models=models,
        objectives=["density_hard", "seg"],
        run_tags=run_tags,
        group_cols=["dataset", "model", "objective", "run_tag"],
    )
    rows = []
    for model in models:
        row, _, _ = architecture_row(best, model)
        rows.append(row)
    return pd.DataFrame(rows)


def write_attention_unet_control(_summary, outdir):
    path = outdir / "attention_unet_control.tex"
    csv_path = outdir / "attention_unet_control.csv"
    if path.exists():
        path.unlink()
    if csv_path.exists():
        csv_path.unlink()
    return False


def seizure_replication(summary):
    models = ["gru", "unet"]
    objectives = ["density_hard", "density_gau", "seg"]
    best = best_metric_rows(
        summary,
        dataset="seizure",
        models=models,
        objectives=objectives,
        run_tags=[SEIZURE_RUN_TAG],
    )
    rows = []
    for model in models:
        row = {"model": model}
        counts = []
        for objective, label in [
            ("density_hard", "BDL-Hard"),
            ("density_gau", "BDL-Gaussian"),
            ("seg", "Cross-entropy"),
        ]:
            cell, _, count = table_cell(best, model=model, objective=objective)
            row[label] = cell
            if count is not None:
                counts.append(count)
        row["seeds"] = min(counts) if counts else "--"
        rows.append(row)
    return pd.DataFrame(rows)


def seizure_highscore_table(summary):
    settings = [
        (
            "GRU 3x128, 2 s stride",
            "gru_3l_128h",
            SEIZURE_HIGH_SCORE_TAG,
        ),
        (
            "GRU 3x128, 1 s stride",
            "gru_3l_128h",
            SEIZURE_HIGH_SCORE_DS256_TAG,
        ),
    ]
    objectives = ["density_hard", "density_gau", "density_custom", "seg"]
    best = best_metric_rows(
        summary,
        dataset="seizure",
        models=["gru_3l_128h"],
        objectives=objectives,
        run_tags=[tag for _, _, tag in settings],
        group_cols=["dataset", "model", "objective", "run_tag"],
    )
    rows = []
    for setting, model, run_tag in settings:
        row = {"setting": setting, "run_tag": run_tag}
        counts = []
        for objective in objectives:
            label = OBJECTIVE_LABELS[objective]
            cell, _, count = table_cell(
                best,
                model=model,
                objective=objective,
                run_tag=run_tag,
            )
            row[label] = cell
            if count is not None:
                counts.append(count)
        row["seeds"] = min(counts) if counts else "--"
        rows.append(row)
    return pd.DataFrame(rows)


def online_ablation(summary):
    models = ["fgru", "flstm", "causal_transformer"]
    best = best_metric_rows(
        summary,
        dataset="sleep",
        models=models,
        objectives=["density_hard", "seg"],
        run_tags=[ONLINE_RUN_TAG],
    )
    rows = []
    for model in models:
        bdl_cell, bdl_mean, bdl_count = table_cell(best, model=model, objective="density_hard")
        seg_cell, seg_mean, seg_count = table_cell(best, model=model, objective="seg")
        counts = [count for count in [bdl_count, seg_count] if count is not None]
        rows.append(
            {
                "model": model,
                "BDL-Hard": bdl_cell,
                "Cross-entropy": seg_cell,
                "$\\Delta$": format_delta(bdl_mean, seg_mean),
                "seeds": min(counts) if counts else "--",
            }
        )
    return pd.DataFrame(rows)


def online_smoothing_ablation(summary):
    models = ["fgru", "flstm", "causal_transformer"]
    objectives = ["density_gau", "density_custom", "seg"]
    best = best_metric_rows(
        summary,
        dataset="sleep",
        models=models,
        objectives=objectives,
        run_tags=[ONLINE_SMOOTHING_RUN_TAG],
    )
    rows = []
    for model in models:
        gau_cell, gau_mean, gau_count = table_cell(best, model=model, objective="density_gau")
        tol_cell, tol_mean, tol_count = table_cell(best, model=model, objective="density_custom")
        seg_cell, seg_mean, seg_count = table_cell(best, model=model, objective="seg")
        bdl_means = [value for value in [gau_mean, tol_mean] if value is not None]
        best_bdl = max(bdl_means) if bdl_means else None
        counts = [count for count in [gau_count, tol_count, seg_count] if count is not None]
        rows.append(
            {
                "model": model,
                "BDL-Gaussian": gau_cell,
                "BDL-Tolerance": tol_cell,
                "Cross-entropy": seg_cell,
                "$\\Delta$": format_delta(best_bdl, seg_mean),
                "seeds": min(counts) if counts else "--",
            }
        )
    return pd.DataFrame(rows)


def offline_transformer_diagnostic(summary):
    best = best_metric_rows(
        summary,
        dataset="sleep",
        models=["transformer_4l_64h_4a"],
        objectives=["density_hard", "seg"],
        run_tags=["transformer_offline_e20_bs32_eval5"],
        group_cols=["dataset", "model", "objective"],
    )
    rows = []
    for objective in ["density_hard", "seg"]:
        cell, _, count = table_cell(best, objective=objective)
        rows.append(
            {
                "objective": objective,
                "mAP": cell,
                "seeds": count if count is not None else "--",
            }
        )
    return pd.DataFrame(rows)


def transformer_main_candidate(summary):
    objectives = ["density_gau", "density_custom", "seg"]
    best = best_metric_rows(
        summary,
        dataset="sleep",
        models=["transformer_4l_64h_4a"],
        objectives=objectives,
        run_tags=[TRANSFORMER_MAIN_CANDIDATE_TAG],
        group_cols=["dataset", "model", "objective", "run_tag"],
    )
    row = {"model": "transformer_4l_64h_4a"}
    counts = []
    bdl_means = []
    seg_mean = None
    for objective in objectives:
        label = OBJECTIVE_LABELS[objective]
        cell, mean, count = table_cell(best, objective=objective)
        row[label] = cell
        if objective.startswith("density_") and mean is not None:
            bdl_means.append(mean)
        if count is not None:
            counts.append(count)
        if objective == "seg":
            seg_mean = mean
    best_bdl = max(bdl_means) if bdl_means else None
    row["$\\Delta$"] = format_delta(best_bdl, seg_mean)
    row["seeds"] = min(counts) if counts else "--"
    return pd.DataFrame([row])


def transformer_strong_candidate(summary):
    objectives = ["density_hard", "density_gau", "density_custom", "seg"]
    model = "transformer_6l_128h_8a_10d"
    best = best_metric_rows(
        summary,
        dataset="sleep",
        models=[model],
        objectives=objectives,
        run_tags=[TRANSFORMER_STRONG_CANDIDATE_TAG],
        group_cols=["dataset", "model", "objective", "run_tag"],
    )
    row = {"model": model}
    counts = []
    bdl_means = []
    seg_mean = None
    for objective in objectives:
        label = OBJECTIVE_LABELS[objective]
        cell, mean, count = table_cell(best, model=model, objective=objective)
        row[label] = cell
        if objective.startswith("density_") and mean is not None:
            bdl_means.append(mean)
        if count is not None:
            counts.append(count)
        if objective == "seg":
            seg_mean = mean
    best_bdl = max(bdl_means) if bdl_means else None
    row["$\\Delta$"] = format_delta(best_bdl, seg_mean)
    row["seeds"] = min(counts) if counts else "--"
    return pd.DataFrame([row])


def point_event_ablation(summary):
    datasets = ["bowshock", "fraud"]
    objectives = ["density_hard", "density_gau", "density_custom", "seg"]
    rows = []
    map_best = best_metric_rows(
        summary,
        dataset=None,
        models=["gru"],
        objectives=objectives,
        run_tags=[POINT_EVENT_RUN_TAG],
        metric="mAP",
        group_cols=["dataset", "model", "objective"],
    )
    f1_best = best_metric_rows(
        summary,
        dataset=None,
        models=["gru"],
        objectives=objectives,
        run_tags=[POINT_EVENT_RUN_TAG],
        metric="mf1",
        group_cols=["dataset", "model", "objective"],
    )
    for dataset in datasets:
        for objective in objectives:
            map_cell, _, map_count = table_cell(
                map_best,
                dataset=dataset,
                model="gru",
                objective=objective,
            )
            f1_cell, _, f1_count = table_cell(
                f1_best,
                dataset=dataset,
                model="gru",
                objective=objective,
            )
            counts = [count for count in [map_count, f1_count] if count is not None]
            rows.append(
                {
                    "dataset": dataset,
                    "objective": objective,
                    "mAP": map_cell,
                    "mf1": f1_cell,
                    "seeds": min(counts) if counts else "--",
                }
            )
    return pd.DataFrame(rows)


def prior_ablation(summary):
    labels = [
        ("prior_sparse_e20_bs32_eval5", "Sparse prior"),
        ("prior_none_e20_bs32_eval5", "No prior"),
    ]
    best = best_metric_rows(
        summary,
        dataset="sleep",
        models=["gru"],
        objectives=["density_custom"],
        run_tags=[tag for tag, _ in labels],
        group_cols=["dataset", "model", "objective", "run_tag"],
    )
    rows = []
    for run_tag, label in labels:
        cell, _, count = table_cell(best, run_tag=run_tag)
        rows.append({"setting": label, "mAP": cell, "seeds": count if count is not None else "--"})
    return pd.DataFrame(rows)


def target_width_ablation(summary):
    labels = [
        ("tol05_e20_bs32_eval5", "$0.5\\times$"),
        (MAIN_RUN_TAG, "$1.0\\times$"),
        ("tol15_e20_bs32_eval5", "$1.5\\times$"),
    ]
    best = best_metric_rows(
        summary,
        dataset="sleep",
        models=["gru"],
        objectives=["density_custom"],
        run_tags=[tag for tag, _ in labels],
        group_cols=["dataset", "model", "objective", "run_tag"],
    )
    rows = []
    for run_tag, label in labels:
        cell, _, count = table_cell(best, run_tag=run_tag)
        rows.append({"width": label, "mAP": cell, "seeds": count if count is not None else "--"})
    return pd.DataFrame(rows)


def write_latex_table(frame, path, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    if frame.empty:
        path.write_text("% No results available yet.\n", encoding="utf-8")
        return
    table = frame.loc[:, columns].copy()
    if "dataset" in table:
        table["dataset"] = table["dataset"].map(DATASET_LABELS).fillna(table["dataset"])
    if "model" in table:
        table["model"] = table["model"].map(MODEL_LABELS).fillna(table["model"])
    if "objective" in table:
        table["objective"] = table["objective"].map(OBJECTIVE_LABELS).fillna(table["objective"])
    if "postprocess_objective" in table:
        table["postprocess_objective"] = (
            table["postprocess_objective"]
            .map(POSTPROCESS_LABELS)
            .fillna(table["postprocess_objective"])
        )
    if "run_tag" in table:
        table["run_tag"] = table["run_tag"].map(RUN_LABELS).fillna(table["run_tag"])
    for col in table.select_dtypes(include=["object"]).columns:
        if col == "mean_std":
            continue
        table[col] = table[col].astype(str).str.replace("_", r"\_", regex=False)
    table = table.rename(columns=DISPLAY_COLUMNS)
    table.to_latex(path, index=False, escape=False, float_format="%.3f")


def table_values_complete(frame, value_columns):
    if frame.empty:
        return False
    if any(column not in frame for column in value_columns):
        return False
    values = frame.loc[:, value_columns].astype(str)
    missing = values.isin(["", "--", "nan", "None"])
    return not missing.any().any()


def write_ready_latex_table(frame, path, columns, value_columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    if table_values_complete(frame, value_columns):
        write_latex_table(frame, path, columns)
        return True
    if path.exists():
        path.unlink()
    return False


def complete_rows(frame, value_columns):
    if frame.empty:
        return frame.copy()
    if any(column not in frame for column in value_columns):
        return frame.iloc[0:0].copy()
    complete = frame.copy()
    values = complete.loc[:, value_columns].astype(str)
    missing = values.isin(["", "--", "nan", "None"])
    return complete.loc[~missing.any(axis=1)].copy()


def write_empty_outputs(outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    csv_outputs = {
        "all_scores.csv": [],
        "all_scores_standard.csv": [],
        "summary_tuned.csv": [
            "dataset",
            "model",
            "objective",
            "run_tag",
            "postprocess_objective",
            "metric",
            "mean",
            "std",
            "count",
            "mean_std",
        ],
        "tolerance_curves.csv": [
            "dataset",
            "model",
            "objective",
            "run_tag",
            "postprocess_objective",
            "tolerance",
            "mean",
            "std",
            "count",
        ],
        "sleep_main_architecture.csv": [
            "model",
            "BDL-Hard",
            "Cross-entropy",
            "$\\Delta$",
        ],
        "sleep_objective_sweep.csv": [
            "family",
            "method",
            "post-processing",
            "mAP",
        ],
        "seizure_replication.csv": [
            "model",
            "BDL-Hard",
            "BDL-Gaussian",
            "Cross-entropy",
        ],
        "seizure_highscore_tracker.csv": [
            "setting",
            "BDL-Hard",
            "BDL-Gaussian",
            "BDL-Tolerance",
            "Cross-entropy",
        ],
        "seizure_highscore_strict3_tracker.csv": [
            "setting",
            "BDL-Hard",
            "BDL-Gaussian",
            "BDL-Tolerance",
            "Cross-entropy",
        ],
        "online_ablation.csv": [
            "model",
            "BDL-Hard",
            "Cross-entropy",
            "$\\Delta$",
        ],
        "online_smoothing_ablation.csv": [
            "model",
            "BDL-Gaussian",
            "BDL-Tolerance",
            "Cross-entropy",
            "$\\Delta$",
        ],
        "offline_transformer_diagnostic.csv": [
            "objective",
            "mAP",
            "seeds",
        ],
        "point_event_ablation.csv": [
            "dataset",
            "objective",
            "mAP",
            "mf1",
        ],
        "prior_ablation.csv": ["setting", "mAP"],
        "target_width_ablation.csv": ["width", "mAP"],
        "paired_fold_consistency.csv": [
            "comparison",
            "paired_records",
            "wins",
            "sign_p",
            "mean_delta",
            "min_delta",
        ],
        "segmentation_reconstruction.csv": [
            "method",
            "decoder",
            "accuracy",
            "balanced_accuracy",
            "f1",
            "iou",
        ],
    }
    for filename, columns in csv_outputs.items():
        pd.DataFrame(columns=columns).to_csv(outdir / filename, index=False)

    latex_outputs = {
        "sleep_main_architecture.tex": [
            "model",
            "BDL-Hard",
            "Cross-entropy",
            "$\\Delta$",
        ],
        "summary_tuned_map.tex": [
            "dataset",
            "model",
            "objective",
            "run_tag",
            "postprocess_objective",
            "mean_std",
        ],
        "sleep_objective_sweep.tex": [
            "family",
            "method",
            "post-processing",
            "mAP",
        ],
        "seizure_replication.tex": [
            "model",
            "BDL-Hard",
            "BDL-Gaussian",
            "Cross-entropy",
        ],
        "online_ablation.tex": [
            "model",
            "BDL-Hard",
            "Cross-entropy",
            "$\\Delta$",
        ],
        "online_smoothing_ablation.tex": [
            "model",
            "BDL-Gaussian",
            "BDL-Tolerance",
            "Cross-entropy",
            "$\\Delta$",
        ],
        "offline_transformer_diagnostic.tex": ["objective", "mAP", "seeds"],
        "point_event_ablation.tex": ["dataset", "objective", "mAP", "mf1"],
        "prior_ablation.tex": ["setting", "mAP"],
        "target_width_ablation.tex": ["width", "mAP"],
        "paired_fold_consistency.tex": [
            "comparison",
            "paired_records",
            "wins",
            "sign_p",
            "mean_delta",
            "min_delta",
        ],
        "segmentation_reconstruction.tex": [
            "method",
            "decoder",
            "accuracy",
            "balanced_accuracy",
            "f1",
            "iou",
        ],
    }
    for filename, columns in latex_outputs.items():
        write_latex_table(pd.DataFrame(columns=columns), outdir / filename, columns)

    (outdir / "generated_tables.txt").write_text(
        "No experiment scores were found. Placeholder tables were written so the manuscript can compile before results are restored.\n",
        encoding="utf-8",
    )


def write_outputs(scores, folds, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    results_root = infer_results_root(scores, folds)

    if scores.empty:
        write_empty_outputs(outdir)
        return

    standard_scores = default_score_rows(scores)
    summary = tuned_summary(standard_scores)
    tolerances = tolerance_summary(standard_scores)
    strict3_scores = suffixed_score_rows(scores, "strict3")
    strict3_summary = tuned_summary(strict3_scores)
    strict3_tolerances = tolerance_summary(strict3_scores)
    fold_table = fold_summary(folds)
    sleep_arch = sleep_main_architecture(summary)
    write_attention_unet_control(summary, outdir)
    seizure_table = seizure_replication(summary)
    seizure_highscore = seizure_highscore_table(summary)
    seizure_highscore_value_columns = [
        "BDL-Hard",
        "BDL-Gaussian",
        "BDL-Tolerance",
        "Cross-entropy",
    ]
    seizure_highscore_completed = complete_rows(
        seizure_highscore,
        seizure_highscore_value_columns,
    )
    seizure_highscore_strict3 = seizure_highscore_table(strict3_summary)
    seizure_highscore_strict3_completed = complete_rows(
        seizure_highscore_strict3,
        seizure_highscore_value_columns,
    )
    online_table = online_ablation(summary)
    online_smoothing_table = online_smoothing_ablation(summary)
    transformer_table = offline_transformer_diagnostic(summary)
    transformer_candidate_table = transformer_main_candidate(summary)
    transformer_candidate_value_columns = [
        "BDL-Gaussian",
        "BDL-Tolerance",
        "Cross-entropy",
    ]
    transformer_strong_table = transformer_strong_candidate(summary)
    transformer_strong_value_columns = [
        "BDL-Hard",
        "BDL-Gaussian",
        "BDL-Tolerance",
        "Cross-entropy",
    ]
    point_table = point_event_ablation(summary)
    prior_table = prior_ablation(summary)
    width_table = target_width_ablation(summary)

    scores.to_csv(outdir / "all_scores.csv", index=False)
    standard_scores.to_csv(outdir / "all_scores_standard.csv", index=False)
    summary.to_csv(outdir / "summary_tuned.csv", index=False)
    tolerances.to_csv(outdir / "tolerance_curves.csv", index=False)
    if not strict3_scores.empty:
        strict3_scores.to_csv(outdir / "chbmit_strict3_scores.csv", index=False)
    if not strict3_summary.empty:
        strict3_summary.to_csv(outdir / "chbmit_strict3_summary.csv", index=False)
    if not strict3_tolerances.empty:
        strict3_tolerances.to_csv(outdir / "chbmit_strict3_tolerance_curves.csv", index=False)
    sleep_arch.to_csv(outdir / "sleep_main_architecture.csv", index=False)
    seizure_table.to_csv(outdir / "seizure_replication.csv", index=False)
    seizure_highscore.to_csv(outdir / "seizure_highscore_tracker.csv", index=False)
    seizure_highscore_completed.to_csv(
        outdir / "seizure_highscore_completed.csv",
        index=False,
    )
    seizure_highscore_strict3.to_csv(
        outdir / "seizure_highscore_strict3_tracker.csv",
        index=False,
    )
    seizure_highscore_strict3_completed.to_csv(
        outdir / "seizure_highscore_strict3_completed.csv",
        index=False,
    )
    online_table.to_csv(outdir / "online_ablation.csv", index=False)
    online_smoothing_table.to_csv(outdir / "online_smoothing_ablation.csv", index=False)
    transformer_table.to_csv(outdir / "offline_transformer_diagnostic.csv", index=False)
    transformer_candidate_table.to_csv(
        outdir / "transformer_main_candidate.csv",
        index=False,
    )
    transformer_strong_table.to_csv(
        outdir / "transformer_strong_candidate.csv",
        index=False,
    )
    point_table.to_csv(outdir / "point_event_ablation.csv", index=False)
    prior_table.to_csv(outdir / "prior_ablation.csv", index=False)
    width_table.to_csv(outdir / "target_width_ablation.csv", index=False)
    sleep_sweep = best_sleep_objective_sweep(summary)
    if not sleep_sweep.empty:
        sleep_sweep.to_csv(outdir / "sleep_objective_sweep.csv", index=False)
    if not fold_table.empty:
        fold_table.to_csv(outdir / "fold_summary.csv", index=False)
    if results_root is not None:
        write_paired_fold_outputs(results_root, outdir)

    write_latex_table(
        sleep_arch,
        outdir / "sleep_main_architecture.tex",
        ["model", "BDL-Hard", "Cross-entropy", "$\\Delta$"],
    )
    write_latex_table(
        summary[summary["metric"] == "mAP"],
        outdir / "summary_tuned_map.tex",
        [
            "dataset",
            "model",
            "objective",
            "run_tag",
            "postprocess_objective",
            "mean_std",
        ],
    )
    write_latex_table(
        sleep_sweep,
        outdir / "sleep_objective_sweep.tex",
        [
            "family",
            "method",
            "post-processing",
            "mAP",
        ],
    )
    write_latex_table(
        seizure_table,
        outdir / "seizure_replication.tex",
        ["model", "BDL-Hard", "BDL-Gaussian", "Cross-entropy"],
    )
    seizure_highscore_ready = write_ready_latex_table(
        seizure_highscore,
        outdir / "seizure_highscore_ready.tex",
        [
            "setting",
            "BDL-Hard",
            "BDL-Gaussian",
            "BDL-Tolerance",
            "Cross-entropy",
        ],
        seizure_highscore_value_columns,
    )
    seizure_highscore_completed_ready = write_ready_latex_table(
        seizure_highscore_completed,
        outdir / "seizure_highscore_completed.tex",
        [
            "setting",
            "BDL-Hard",
            "BDL-Gaussian",
            "BDL-Tolerance",
            "Cross-entropy",
        ],
        seizure_highscore_value_columns,
    )
    seizure_highscore_strict3_ready = write_ready_latex_table(
        seizure_highscore_strict3,
        outdir / "seizure_highscore_strict3_ready.tex",
        [
            "setting",
            "BDL-Hard",
            "BDL-Gaussian",
            "BDL-Tolerance",
            "Cross-entropy",
        ],
        seizure_highscore_value_columns,
    )
    seizure_highscore_strict3_completed_ready = write_ready_latex_table(
        seizure_highscore_strict3_completed,
        outdir / "seizure_highscore_strict3_completed.tex",
        [
            "setting",
            "BDL-Hard",
            "BDL-Gaussian",
            "BDL-Tolerance",
            "Cross-entropy",
        ],
        seizure_highscore_value_columns,
    )
    write_latex_table(
        online_table,
        outdir / "online_ablation.tex",
        ["model", "BDL-Hard", "Cross-entropy", "$\\Delta$"],
    )
    write_latex_table(
        online_smoothing_table,
        outdir / "online_smoothing_ablation.tex",
        ["model", "BDL-Gaussian", "BDL-Tolerance", "Cross-entropy", "$\\Delta$"],
    )
    write_latex_table(
        transformer_table,
        outdir / "offline_transformer_diagnostic.tex",
        ["objective", "mAP", "seeds"],
    )
    transformer_candidate_ready = write_ready_latex_table(
        transformer_candidate_table,
        outdir / "transformer_main_candidate_ready.tex",
        ["model", "BDL-Gaussian", "BDL-Tolerance", "Cross-entropy", "$\\Delta$"],
        transformer_candidate_value_columns,
    )
    transformer_strong_ready = write_ready_latex_table(
        transformer_strong_table,
        outdir / "transformer_strong_candidate_ready.tex",
        [
            "model",
            "BDL-Hard",
            "BDL-Gaussian",
            "BDL-Tolerance",
            "Cross-entropy",
            "$\\Delta$",
        ],
        transformer_strong_value_columns,
    )
    write_latex_table(
        point_table,
        outdir / "point_event_ablation.tex",
        ["dataset", "objective", "mAP", "mf1"],
    )
    point_table_ready = write_ready_latex_table(
        point_table,
        outdir / "point_event_ablation_ready.tex",
        ["dataset", "objective", "mAP", "mf1"],
        ["mAP", "mf1"],
    )
    write_latex_table(
        prior_table,
        outdir / "prior_ablation.tex",
        ["setting", "mAP"],
    )
    write_latex_table(
        width_table,
        outdir / "target_width_ablation.tex",
        ["width", "mAP"],
    )

    (outdir / "generated_tables.txt").write_text(
        "\n".join(
            [
                "Generated Experiment Tables",
                "",
                "Regenerate with `uv run --offline python paper/collect_results.py --results-root experiments`.",
                "Run directories under `experiments/quarantine/` are excluded.",
                "",
                "- `summary_tuned.csv`: tuned summary metrics grouped by run tag.",
                "- `all_scores.csv`: all score files, including suffixed diagnostics such as strict CHB-MIT rescoring.",
                "- `all_scores_standard.csv`: default `scores.csv` rows used by the main generated tables.",
                "- `sleep_main_architecture.csv`: main sleep architecture comparison.",
                "- `attention_unet_control.tex`: not emitted; attention-gated U-Net remains in the main architecture table.",
                "- `sleep_objective_sweep.csv`: compact best-postprocessing table for the sleep GRU objective sweep.",
                "- `seizure_replication.csv`: second-benchmark replication table.",
                "- `seizure_highscore_tracker.csv`: CHB-MIT high-capacity GRU reproduction and stride-ablation tracker.",
                f"- `seizure_highscore_completed.tex`: {'available for completed stride rows' if seizure_highscore_completed_ready else 'not yet available; no stride row has all methods scored'}.",
                f"- `seizure_highscore_ready.tex`: {'available' if seizure_highscore_ready else 'not yet available; P12/P12b still need complete scores'}.",
                "- `seizure_highscore_strict3_tracker.csv`: compact strict 1--3 second CHB-MIT tracker matching the high-capacity table schema.",
                f"- `seizure_highscore_strict3_completed.tex`: {'available for completed strict stride rows' if seizure_highscore_strict3_completed_ready else 'not yet available; no strict stride row has all methods scored'}.",
                f"- `seizure_highscore_strict3_ready.tex`: {'available' if seizure_highscore_strict3_ready else 'not yet available; strict P12/P12b still need complete scores'}.",
                "- `online_ablation.csv`: causal/online context ablation table.",
                "- `online_smoothing_ablation.csv`: smoothed-target streaming-context diagnostic table.",
                "- `offline_transformer_diagnostic.csv`: optional sleep offline-Transformer diagnostic.",
                f"- `transformer_main_candidate_ready.tex`: {'available' if transformer_candidate_ready else 'not yet available; P15 Transformer candidate still needs complete scores'}.",
                f"- `transformer_strong_candidate_ready.tex`: {'available' if transformer_strong_ready else 'not yet available; P16 strong Transformer candidate still needs complete scores'}.",
                "- `point_event_ablation.csv`: Bowshock/Fraud point-event ablation tracker.",
                f"- `point_event_ablation_ready.tex`: {'available' if point_table_ready else 'not yet available; P13/P14 still need complete scores'}.",
                "- `paired_fold_consistency.csv`: paired validation-fold consistency checks for completed main comparisons.",
                "- `prior_ablation.csv`: sparse-output-prior ablation table.",
                "- `target_width_ablation.csv`: tolerance-kernel width ablation table.",
                "- `tolerance_curves.csv`: tuned mAP by tolerance.",
                "- `chbmit_strict3_*.csv`: optional CHB-MIT strict 1--3 second boundary-rescoring outputs when `scores_strict3.csv` files exist.",
                "- `fold_summary.csv`: fold-level validation diagnostics.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def infer_results_root(*frames):
    for frame in frames:
        if frame is None or frame.empty or "source_path" not in frame:
            continue
        for value in frame["source_path"].dropna().astype(str):
            path = Path(value)
            parts = path.parts
            if "experiments" in parts:
                return Path(*parts[: parts.index("experiments") + 1])
    return None


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default="experiments")
    parser.add_argument("--outdir", default="paper/results/generated")
    return parser.parse_args()


def main():
    args = parse_args()
    scores = load_scores(args.results_root)
    folds = load_fold_results(args.results_root)
    write_outputs(scores, folds, args.outdir)
    print(f"Found {len(scores)} score rows and {len(folds)} fold rows.")
    print(f"Wrote generated result tables to {args.outdir}.")


if __name__ == "__main__":
    main()
