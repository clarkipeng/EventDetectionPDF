"""Generate paired-fold consistency summaries for the BDL paper."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


def load_fold(
    results_root: Path,
    dataset: str,
    model: str,
    objective: str,
    seed: int,
    run_tag: str,
) -> pd.DataFrame:
    path = (
        results_root
        / dataset
        / model
        / objective
        / f"seed_{seed}_{run_tag}"
        / "results"
        / "fold_results.csv"
    )
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df["dataset"] = dataset
    df["model"] = model
    df["objective"] = objective
    df["seed"] = seed
    df["run_tag"] = run_tag
    return df


def summarize_pair(
    results_root: Path,
    label: str,
    dataset: str,
    model: str,
    objective_a: str,
    objective_b: str,
    run_tag: str,
    seeds: list[int],
) -> dict[str, object]:
    rows = []
    for seed in seeds:
        a = load_fold(results_root, dataset, model, objective_a, seed, run_tag)
        b = load_fold(results_root, dataset, model, objective_b, seed, run_tag)
        merged = a[["fold", "best_valid_mAP"]].merge(
            b[["fold", "best_valid_mAP"]],
            on="fold",
            suffixes=("_a", "_b"),
        )
        merged["delta"] = merged["best_valid_mAP_a"] - merged["best_valid_mAP_b"]
        merged["seed"] = seed
        rows.append(merged)
    paired = pd.concat(rows, ignore_index=True)
    wins = int((paired["delta"] > 0).sum())
    non_ties = int((paired["delta"] != 0).sum())
    sign_p = sum(math.comb(non_ties, k) for k in range(wins, non_ties + 1)) / (
        2**non_ties
    )
    return {
        "comparison": label,
        "paired_records": int(len(paired)),
        "wins": wins,
        "sign_p": float(sign_p),
        "mean_delta": float(paired["delta"].mean()),
        "min_delta": float(paired["delta"].min()),
    }


def format_tex(df: pd.DataFrame) -> str:
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Comparison & Pairs & Wins & $p_{\mathrm{sign}}$ & Mean $\Delta$ & Min. $\Delta$ \\",
        r"\midrule",
    ]
    for row in df.itertuples(index=False):
        sign_p = "<0.001" if row.sign_p < 0.001 else f"{row.sign_p:.3f}"
        lines.append(
            f"{row.comparison} & {row.paired_records:d} & {row.wins:d} & "
            f"{sign_p} & {row.mean_delta:+.3f} & {row.min_delta:+.3f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def build_paired_fold_table(results_root: Path) -> pd.DataFrame:
    rows = [
        summarize_pair(
            results_root,
            "Sleep GRU: BDL-Hard vs cross-entropy",
            "sleep",
            "gru",
            "density_hard",
            "seg",
            "objective_e20_bs32_eval5",
            [0, 1, 2],
        ),
        summarize_pair(
            results_root,
            "Sleep U-Net: BDL-Hard vs cross-entropy",
            "sleep",
            "unet",
            "density_hard",
            "seg",
            "objective_e20_bs32_eval5",
            [0, 1, 2],
        ),
        summarize_pair(
            results_root,
            "Sleep attention-gated U-Net: BDL-Hard vs cross-entropy",
            "sleep",
            "unet_t",
            "density_hard",
            "seg",
            "objective_e20_bs32_eval5",
            [0, 1, 2],
        ),
        summarize_pair(
            results_root,
            "Sleep GRU: BDL-Hard vs weighted cross-entropy",
            "sleep",
            "gru",
            "density_hard",
            "seg_weighted",
            "objective_e20_bs32_eval5",
            [0, 1, 2],
        ),
        summarize_pair(
            results_root,
            "Sleep GRU: BDL-Hard vs focal",
            "sleep",
            "gru",
            "density_hard",
            "seg_focal",
            "objective_e20_bs32_eval5",
            [0, 1, 2],
        ),
        summarize_pair(
            results_root,
            "CHB-MIT 1 s: BDL-Hard vs cross-entropy",
            "seizure",
            "gru_3l_128h",
            "density_hard",
            "seg",
            "seizure_highscore_gru3l128h_ds256_bs8_e20",
            [0],
        ),
        summarize_pair(
            results_root,
            "CHB-MIT 2 s: BDL-Gaussian vs cross-entropy",
            "seizure",
            "gru_3l_128h",
            "density_gau",
            "seg",
            "seizure_highscore_gru3l128h_ds512_bs8_e20",
            [0],
        ),
    ]
    return pd.DataFrame(rows)


def write_paired_fold_outputs(results_root: Path, outdir: Path) -> pd.DataFrame:
    df = build_paired_fold_table(results_root)
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / "paired_fold_consistency.csv", index=False)
    (outdir / "paired_fold_consistency.tex").write_text(
        format_tex(df),
        encoding="utf-8",
    )
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="experiments", type=Path)
    parser.add_argument("--outdir", default="paper/results/generated", type=Path)
    args = parser.parse_args()

    write_paired_fold_outputs(args.results_root, args.outdir)
    print(f"Wrote {args.outdir / 'paired_fold_consistency.csv'}")
    print(f"Wrote {args.outdir / 'paired_fold_consistency.tex'}")


if __name__ == "__main__":
    main()
