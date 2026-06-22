"""Small optional Weights & Biases helpers for experiment tracking."""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np


def parse_wandb_tags(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        tags = value
    else:
        tags = str(value).split(",")
    tags = [str(tag).strip() for tag in tags if str(tag).strip()]
    return tags or None


def safe_wandb_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-") or "run"


def default_wandb_name(config, job_type="train"):
    parts = [
        config.get("dataset", "dataset"),
        config.get("model", "model"),
        config.get("objective", "objective"),
        f"seed{config.get('seed', 0)}",
    ]
    run_tag = config.get("run_tag")
    if run_tag:
        parts.append(run_tag)
    if job_type and job_type != "train":
        parts.append(job_type)
    return safe_wandb_name("-".join(str(part) for part in parts))


def init_wandb_run(
    enabled,
    config,
    project,
    entity=None,
    group=None,
    name=None,
    tags=None,
    mode=None,
    job_type="train",
):
    if not enabled:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "W&B logging was requested, but wandb is not installed. "
            "Install it with `uv pip install --python .venv/bin/python wandb`."
        ) from exc

    init_kwargs = {
        "project": project,
        "config": config,
        "name": name or default_wandb_name(config, job_type=job_type),
        "group": group
        or safe_wandb_name(
            f"{config.get('dataset')}-{config.get('model')}-{config.get('objective')}"
        ),
        "job_type": job_type,
        "tags": parse_wandb_tags(tags),
    }
    if entity:
        init_kwargs["entity"] = entity
    if mode:
        init_kwargs["mode"] = mode
    return wandb.init(**init_kwargs)


def finite_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def log_wandb(run, metrics, step=None):
    if run is None:
        return
    clean = {}
    for key, value in metrics.items():
        if isinstance(value, (float, int, np.floating, np.integer)):
            value = finite_float(value)
            if value is None:
                continue
        clean[key] = value
    if clean:
        run.log(clean, step=step)


def update_wandb_summary(run, metrics):
    if run is None:
        return
    for key, value in metrics.items():
        if isinstance(value, (float, int, np.floating, np.integer)):
            value = finite_float(value)
            if value is None:
                continue
        run.summary[key] = value


def log_wandb_artifact(run, name, artifact_type, paths):
    if run is None:
        return
    try:
        import wandb
    except ImportError:
        return

    artifact = wandb.Artifact(safe_wandb_name(name), type=artifact_type)
    added = False
    for path in paths:
        path = Path(path)
        if path.is_file():
            artifact.add_file(str(path), name=path.name)
            added = True
    if added:
        run.log_artifact(artifact)


def log_score_payload_to_wandb(run, payload, prefix="score"):
    if run is None or not payload:
        return
    summary = {}
    for result in payload.get("results", []):
        postprocess_objective = result.get("postprocess_objective", "postprocess")
        for metric, optimized in result.get("optimized", {}).items():
            scores = optimized.get("scores", {})
            for score_name, score in scores.items():
                score = finite_float(score)
                if score is None:
                    continue
                summary[
                    f"{prefix}/{postprocess_objective}/optimized_for_{metric}/{score_name}"
                ] = score
            best_score = finite_float(optimized.get("best_score"))
            if best_score is not None:
                summary[
                    f"{prefix}/{postprocess_objective}/optimized_for_{metric}/best_score"
                ] = best_score
    update_wandb_summary(run, summary)
