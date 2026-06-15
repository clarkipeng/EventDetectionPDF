"""Run or print the experiment matrix for the density-likelihood paper.

By default this script prints commands without executing them. Pass --execute
when the data directory is ready and you want to launch the runs.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from pathlib import Path


CORE_MODELS = ["gru", "unet", "unet_t", "prectime"]
ONLINE_MODELS = ["fgru", "flstm", "causal_transformer"]
DEFAULT_MODELS = CORE_MODELS
DEFAULT_OBJECTIVES = [
    "density_hard",
    "density_gau",
    "density_custom",
    "hard",
    "gau",
    "custom",
    "seg",
    "seg_weighted",
    "seg_focal",
]


def expand_model_aliases(models):
    expanded = []
    for model in models:
        if model == "core":
            expanded.extend(CORE_MODELS)
        elif model == "online":
            expanded.extend(ONLINE_MODELS)
        elif model == "all":
            expanded.extend(CORE_MODELS + ONLINE_MODELS)
        else:
            expanded.append(model)
    return list(dict.fromkeys(expanded))


def command_to_string(command):
    return shlex.join(command)


def expected_scores_path(args, dataset, model, objective, seed):
    return (
        Path(args.experiments_root)
        / dataset
        / model
        / objective
        / f"seed_{seed}"
        / "results"
        / "scores.csv"
    )


def build_command(args, dataset, model, objective, seed):
    command = [
        args.python,
        "train.py",
        "--dataset",
        dataset,
        "--model",
        model,
        "--objective",
        objective,
        "--datadir",
        args.datadir,
        "--epochs",
        str(args.epochs),
        "--folds",
        str(args.folds),
        "--bs",
        str(args.bs),
        "--downsample",
        str(args.downsample),
        "--agg_feats",
        args.agg_feats,
        "--use_cat",
        str(args.use_cat),
        "--normalize",
        str(args.normalize),
        "--workers",
        str(args.workers),
        "--seed",
        str(seed),
        "--density_prior",
        args.density_prior,
        "--best_metric",
        args.best_metric,
        "--save_all_epochs",
        str(args.save_all_epochs),
        "--tolerance_scale",
        str(args.tolerance_scale),
    ]
    if args.sequence_length is not None:
        command.extend(["--sequence_length", str(args.sequence_length)])
    if args.gaussian_sigma is not None:
        command.extend(["--gaussian_sigma", str(args.gaussian_sigma)])
    if args.device is not None:
        command.extend(["--device", args.device])
    return command


def iter_runs(args):
    for dataset in args.datasets:
        for seed in args.seeds:
            for model in args.models:
                for objective in args.objectives:
                    if args.skip_existing and expected_scores_path(
                        args, dataset, model, objective, seed
                    ).exists():
                        continue
                    yield {
                        "dataset": dataset,
                        "seed": seed,
                        "model": model,
                        "objective": objective,
                        "command": build_command(args, dataset, model, objective, seed),
                    }


def write_shell_script(path, runs, repo_root):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rel_repo = os.path.relpath(repo_root, path.parent.resolve())
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f'REPO_ROOT="$(cd "$(dirname "${{BASH_SOURCE[0]}}")/{rel_repo}" && pwd)"',
        'cd "$REPO_ROOT"',
        "",
    ]
    for run in runs:
        lines.append(
            "# "
            f"{run['dataset']} {run['model']} {run['objective']} seed={run['seed']}"
        )
        lines.append(command_to_string(run["command"]))
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    path.chmod(path.stat().st_mode | 0o111)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datadir", default="data")
    parser.add_argument(
        "--python",
        default="python",
        help="Python executable to use in printed, written, and executed commands.",
    )
    parser.add_argument("--datasets", nargs="+", default=["sleep"])
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Model names, or aliases: core, online, all.",
    )
    parser.add_argument("--objectives", nargs="+", default=DEFAULT_OBJECTIVES)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--bs", type=int, default=10)
    parser.add_argument("--downsample", type=int, default=10)
    parser.add_argument("--sequence_length", type=int, default=None)
    parser.add_argument("--agg_feats", default="stat", choices=["stat", "none", "all"])
    parser.add_argument("--use_cat", default=True)
    parser.add_argument("--normalize", default=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--density_prior",
        default="sparse",
        choices=["sparse", "none"],
    )
    parser.add_argument("--best_metric", default="mAP", choices=["mAP", "loss"])
    parser.add_argument("--save_all_epochs", default=False)
    parser.add_argument("--gaussian_sigma", type=float, default=None)
    parser.add_argument("--tolerance_scale", type=float, default=1.0)
    parser.add_argument(
        "--experiments-root",
        default="experiments",
        help="Root used only for --skip-existing checks; train.py writes to experiments/.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip runs whose results/scores.csv already exists.",
    )
    parser.add_argument(
        "--write-script",
        default=None,
        help="Write the planned commands to an executable shell script.",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop after the first failed subprocess when executing.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.models = expand_model_aliases(args.models)
    repo_root = Path(__file__).resolve().parents[1]
    runs = list(iter_runs(args))
    failures = []

    if args.write_script:
        write_shell_script(args.write_script, runs, repo_root)
        print(f"Wrote {len(runs)} commands to {args.write_script}")

    for run in runs:
        command = run["command"]
        print(command_to_string(command))
        if not args.execute:
            continue
        result = subprocess.run(command, cwd=repo_root, check=False)
        if result.returncode != 0:
            failures.append((command, result.returncode))
            if args.stop_on_error:
                raise SystemExit(result.returncode)

    if failures:
        print("\nFailed commands:")
        for command, returncode in failures:
            print(f"{returncode}: {' '.join(command)}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
