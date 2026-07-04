"""Run or print the experiment matrix for the density-likelihood paper.

By default this script prints Trace-wrapped commands without executing them.
Pass --execute when the data directory is ready and you want to launch the
runs. Pass --no-trace only for explicit local debugging.
"""

from __future__ import annotations

import argparse
import csv
import os
import shlex
import subprocess
from pathlib import Path


CORE_MODELS = ["gru", "unet", "unet_t"]
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
DEFAULT_TUNE_SMOOTH_VALUES = "none,1,10,100,1000"
SEIZURE_TUNE_SMOOTH_VALUES = "none,256,512"


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


def truthy(value):
    return str(value).lower() in {"1", "true", "t", "yes", "y"}


def expected_results_path(args, dataset, model, objective, seed):
    seed_dir = f"seed_{seed}"
    if args.run_tag:
        safe_tag = str(args.run_tag).replace(os.sep, "_").replace(" ", "_")
        seed_dir = f"{seed_dir}_{safe_tag}"
    result_name = "scores.csv" if truthy(args.score_after_train) else "fold_results.csv"
    return (
        Path(args.experiments_root)
        / dataset
        / model
        / objective
        / seed_dir
        / "results"
        / result_name
    )


def expected_results_complete(args, dataset, model, objective, seed):
    path = expected_results_path(args, dataset, model, objective, seed)
    if not path.exists() or path.stat().st_size == 0:
        return False

    if truthy(args.score_after_train):
        return True

    try:
        with path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except (OSError, csv.Error):
        return False

    folds = {row.get("fold") for row in rows if row.get("fold") not in {None, ""}}
    return len(folds) >= int(args.folds)


def tune_smooth_values_for_dataset(args, dataset):
    if str(args.tune_smooth_values).lower() != "auto":
        return args.tune_smooth_values
    if dataset == "seizure":
        return SEIZURE_TUNE_SMOOTH_VALUES
    return DEFAULT_TUNE_SMOOTH_VALUES


def build_command(args, dataset, model, objective, seed):
    command = shlex.split(args.python) + [
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
        "--lr",
        str(args.lr),
        "--weight_decay",
        str(args.weight_decay),
        "--clip_grad_norm",
        str(args.clip_grad_norm),
        "--tune_cutoff_steps",
        str(args.tune_cutoff_steps),
        "--tune_smooth_values",
        tune_smooth_values_for_dataset(args, dataset),
        "--tune_alternating",
        str(args.tune_alternating),
        "--score_after_train",
        str(args.score_after_train),
        "--eval_every",
        str(args.eval_every),
        "--wandb",
        str(args.wandb),
        "--wandb_project",
        args.wandb_project,
        "--wandb_log_artifacts",
        str(args.wandb_log_artifacts),
        "--tolerance_scale",
        str(args.tolerance_scale),
    ]
    if args.wandb_entity:
        command.extend(["--wandb_entity", args.wandb_entity])
    if args.wandb_group:
        command.extend(["--wandb_group", args.wandb_group])
    if args.wandb_name:
        command.extend(["--wandb_name", args.wandb_name])
    if args.wandb_mode:
        command.extend(["--wandb_mode", args.wandb_mode])
    if args.wandb_tags:
        command.extend(["--wandb_tags", args.wandb_tags])
    if args.run_tag:
        command.extend(["--run_tag", args.run_tag])
    if args.tune_cutoff_values is not None:
        command.extend(["--tune_cutoff_values", str(args.tune_cutoff_values)])
    if args.sequence_length is not None:
        command.extend(["--sequence_length", str(args.sequence_length)])
    if args.gaussian_sigma is not None:
        command.extend(["--gaussian_sigma", str(args.gaussian_sigma)])
    if args.device is not None:
        command.extend(["--device", args.device])
    return command


def trace_setup(args, dataset, model, objective, seed):
    fields = {
        "dataset": dataset,
        "model": model,
        "objective": objective,
        "seed": seed,
        "epochs": args.epochs,
        "folds": args.folds,
        "batch_size": args.bs,
        "downsample": args.downsample,
        "learning_rate": args.lr,
        "tune_smooth_values": tune_smooth_values_for_dataset(args, dataset),
        "run_tag": args.run_tag or "",
        "score_after_train": args.score_after_train,
    }
    return ", ".join(f"{key}={value}" for key, value in fields.items())


def trace_intent(dataset, model, objective, seed):
    return (
        "Run BDL paper experiment "
        f"{dataset}/{model}/{objective} with seed {seed}."
    )


def wrap_with_trace(args, run):
    if not args.trace:
        return run["command"]
    return [
        "paper/trace_run.sh",
        "--intent",
        trace_intent(
            run["dataset"],
            run["model"],
            run["objective"],
            run["seed"],
        ),
        "--setup",
        trace_setup(
            args,
            run["dataset"],
            run["model"],
            run["objective"],
            run["seed"],
        ),
        "--expect-artifact",
        str(
            expected_results_path(
                args,
                run["dataset"],
                run["model"],
                run["objective"],
                run["seed"],
            )
        ),
        "--",
        *run["command"],
    ]


def iter_runs(args):
    for dataset in args.datasets:
        for seed in args.seeds:
            for model in args.models:
                for objective in args.objectives:
                    if args.skip_existing and expected_results_complete(
                        args, dataset, model, objective, seed
                    ):
                        continue
                    yield {
                        "dataset": dataset,
                        "seed": seed,
                        "model": model,
                        "objective": objective,
                        "command": build_command(args, dataset, model, objective, seed),
                    }


def write_shell_script(path, runs, repo_root, args):
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
        command = wrap_with_trace(args, run)
        lines.append(
            "# "
            f"{run['dataset']} {run['model']} {run['objective']} seed={run['seed']}"
        )
        lines.append(command_to_string(command))
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
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--clip_grad_norm", type=float, default=1e-1)
    parser.add_argument("--run_tag", default=None)
    parser.add_argument("--tune_cutoff_steps", type=int, default=11)
    parser.add_argument("--tune_cutoff_values", default=None)
    parser.add_argument(
        "--tune_smooth_values",
        default="auto",
        help=(
            "Comma-separated smoothing values, or 'auto'. Auto keeps the broad "
            "default grid for sleep/point-event runs and uses none,256,512 for "
            "CHB-MIT seizure scoring."
        ),
    )
    parser.add_argument("--tune_alternating", default=True)
    parser.add_argument("--score_after_train", default=True)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--wandb", default=False)
    parser.add_argument("--wandb_project", default="event-detection-pdf")
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--wandb_group", default=None)
    parser.add_argument("--wandb_name", default=None)
    parser.add_argument("--wandb_mode", default=None)
    parser.add_argument("--wandb_tags", default=None)
    parser.add_argument("--wandb_log_artifacts", default=True)
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
    parser.add_argument(
        "--trace",
        dest="trace",
        action="store_true",
        default=True,
        help="Wrap runs with paper/trace_run.sh. This is the default.",
    )
    parser.add_argument(
        "--no-trace",
        dest="trace",
        action="store_false",
        help=(
            "Local-debug only: print, write, or execute raw train.py commands "
            "without Trace. Requires EVENTPDF_ALLOW_UNTRACED=1."
        ),
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
    if not args.trace and not truthy(os.environ.get("EVENTPDF_ALLOW_UNTRACED", "0")):
        raise SystemExit(
            "Refusing to produce untraced experiment commands. "
            "Future paper experiments must run through Trace/otrace. "
            "Set EVENTPDF_ALLOW_UNTRACED=1 only for explicit local debugging."
        )
    args.models = expand_model_aliases(args.models)
    repo_root = Path(__file__).resolve().parents[1]
    runs = list(iter_runs(args))
    failures = []

    if args.write_script:
        write_shell_script(args.write_script, runs, repo_root, args)
        print(f"Wrote {len(runs)} commands to {args.write_script}")

    for run in runs:
        command = wrap_with_trace(args, run)
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
