"""Backfill epoch progress CSVs from legacy train logs.

This is for long-running paper jobs that started before ``train.py`` wrote
``results/epoch_results.csv`` after every epoch. It parses the compact epoch
summary lines emitted by training logs and writes the same monitoring CSV
schema used by newer runs.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


EPOCH_RE = re.compile(
    r"fold (?P<fold>\d+), epoch (?P<epoch>\d+)/(?P<epochs>\d+): "
    r"train loss: (?P<train_loss>[0-9.eE+-]+), "
    r"valid loss: (?P<valid_loss>[0-9.eE+-]+), "
    r"valid mAP: (?P<valid_mAP>[0-9.eE+-]+)"
)


def parse_epoch_segments(log_text: str) -> list[list[dict[str, float | int]]]:
    """Return sequential objective segments from train log text."""

    segments: list[list[dict[str, float | int]]] = []
    current: list[dict[str, float | int]] = []
    for match in EPOCH_RE.finditer(log_text):
        row = {
            "fold": int(match.group("fold")),
            "epoch": int(match.group("epoch")),
            "epochs": int(match.group("epochs")),
            "train_loss": float(match.group("train_loss")),
            "valid_loss": float(match.group("valid_loss")),
            "valid_mAP": float(match.group("valid_mAP")),
        }
        if row["fold"] == 0 and row["epoch"] == 1 and current:
            segments.append(current)
            current = []
        current.append(row)
    if current:
        segments.append(current)
    return segments


def result_dir(results_root: Path, dataset: str, model: str, objective: str, seed: int, run_tag: str) -> Path:
    return results_root / dataset / model / objective / f"seed_{seed}_{run_tag}" / "results"


def write_epoch_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["fold", "epoch", "epochs", "train_loss", "valid_loss", "valid_mAP"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--results-root", default=Path("experiments"), type=Path)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument(
        "--objectives",
        required=True,
        help="Comma-separated objective order in the sequential training log.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing epoch_results.csv files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    objectives = [item.strip() for item in args.objectives.split(",") if item.strip()]
    text = args.log.read_text(encoding="utf-8", errors="replace")
    segments = parse_epoch_segments(text)
    if len(segments) > len(objectives):
        raise SystemExit(
            f"Found {len(segments)} log segments but only {len(objectives)} objectives were provided."
        )

    wrote = 0
    skipped = 0
    for objective, rows in zip(objectives, segments):
        outdir = result_dir(args.results_root, args.dataset, args.model, objective, args.seed, args.run_tag)
        config_path = outdir / "run_config.json"
        if not config_path.exists():
            skipped += 1
            print(f"skip {objective}: missing {config_path}")
            continue
        output_path = outdir / "epoch_results.csv"
        if output_path.exists() and not args.overwrite:
            skipped += 1
            print(f"skip {objective}: found {output_path}")
            continue
        write_epoch_csv(output_path, rows)
        wrote += 1
        latest = rows[-1]
        print(
            f"wrote {output_path}: {len(rows)} epochs, "
            f"latest fold {latest['fold']} epoch {latest['epoch']}/{latest['epochs']} "
            f"mAP {latest['valid_mAP']:.3f}"
        )

    print(f"segments={len(segments)} wrote={wrote} skipped={skipped}")


if __name__ == "__main__":
    main()
