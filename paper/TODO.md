# Density-Likelihood Rewrite TODO

## Submission blockers

- Replace legacy MSE-centered claims with a maximum-likelihood boundary-density framing.
- Rerun experiments from scratch; current tables and plots are placeholders only.
- Compile the rewritten paper with a clean LaTeX preamble and no generated artifacts committed.
- Add prior-work coverage for heatmap regression, temporal boundary localization, event metrics, and temporal point processes.

## Core experiments

- Sleep dataset first: compare `density_hard`, `density_gau`, and `density_custom` against legacy `hard`, `gau`, `custom`, and segmentation `seg1`/`seg2`.
- Seizure dataset second if compute allows, using the same objective matrix.
- Report fold means and standard deviations, plus pooled EDAP only as a secondary view.
- Add likelihood-vs-MSE, kernel-type, and prior-rate ablations.
- Add an online-model ablation: compare bidirectional `gru` against `fgru`, `flstm`, and FlashAttention-backed `causal_transformer`.
- Include alternating onset/offset post-processing in the post-processing ablation, matching the competition-style sleep/wake cleanup.
- Use a fixed seed list for reruns; start with `--seed 0` to match the legacy split, then add two more seeds if runtime is acceptable.
- Keep raw experiment outputs under `experiments/`; do not commit model checkpoints, cached predictions, or generated plots.
- Prefer `paper/run_experiments.py` for reruns so command lines and matrix membership stay consistent.
- Use `paper/EXPERIMENTS.md` as the GPU-later runbook; generate shell scripts with `--write-script` when moving to Kaggle or a GPU box.
- Use the structured files in `experiments/*/*/*/seed_*/results/` for tables and plots.

## Suggested rerun commands

Run the full objective matrix on sleep first:

```bash
python train_all.py --dataset sleep --datadir data --epochs 20 --folds 4 --normalize True
```

If runtime is too high, run a focused model/objective matrix first:

```bash
python train.py --dataset sleep --model gru --objective density_custom --datadir data --epochs 20 --folds 4 --normalize True
python train.py --dataset sleep --model gru --objective custom --datadir data --epochs 20 --folds 4 --normalize True
python train.py --dataset sleep --model gru --objective seg --datadir data --epochs 20 --folds 4 --normalize True
python train.py --dataset sleep --model gru --objective seg_weighted --datadir data --epochs 20 --folds 4 --normalize True
python train.py --dataset sleep --model gru --objective seg_focal --datadir data --epochs 20 --folds 4 --normalize True
```

Then repeat the strongest density objective and strongest baseline on seizure:

```bash
python train.py --dataset seizure --model unet --objective density_custom --datadir data --epochs 20 --folds 4 --normalize True
python train.py --dataset seizure --model unet --objective seg --datadir data --epochs 20 --folds 4 --normalize True
```

These are starting commands, not final hyperparameter claims. Record the exact command, commit hash, data source, and runtime for every table.

## Plot regeneration

- Treat all current paper figures as disposable.
- Regenerate final plots from the Kaggle `sleep-plots.ipynb` workflow after the new runs finish.
- Prefer plots that explain the new likelihood framing: boundary targets, learned rates, peak extraction, and EDAP-by-tolerance curves.
- Before committing regenerated plots, strip notebook outputs or export static figures into `paper/figures/` with descriptive filenames.
- Prefer `python paper/make_plots.py --dataset sleep --results-root experiments` for final paper figures.
- Replace any legacy plot that implies MSE/PDF regression is the main method.
- Required plot set:
  - target kernels for `density_hard`, `density_gau`, and `density_custom`;
  - predicted onset/offset rates with detected peaks;
  - EDAP by tolerance for density, MSE, and segmentation objectives;
  - ablation plot for likelihood vs MSE under the same kernel.

## Writing and framing

- Rename the method around boundary density/intensity, not generic PDF regression.
- State that temporal point processes are related but heavier than needed for this supervised boundary-localization setting.
- Keep claims dataset-specific and metric-specific.
- Make limitations explicit: segmentation can still win when state occupancy is the desired output.
- Do not claim density likelihood wins until reruns support it.
- Use "legacy MSE heatmap" for the old objective family.
- Use "tolerance-aligned kernel" for `density_custom`; avoid saying it directly optimizes AP.
- Describe online ablations as streaming-context stress tests, not as the main architecture contribution.
- State clearly that the likelihood emits independent onset/offset rates; legal interval alternation is a decoder/post-processing constraint.

## Acceptance checklist

- `python3 -m py_compile src/utils.py train.py train_all.py eval.py models/load_model.py` passes.
- `python paper/run_experiments.py --datasets sleep --models gru --objectives density_custom --seeds 0 --epochs 1 --folds 2 --datadir data` prints a valid dry-run command.
- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` passes from `paper/`.
- No LaTeX build artifacts, model checkpoints, cached predictions, or experiment outputs are staged.
- The paper has no tables or plots from old runs unless they are explicitly labeled placeholders.
