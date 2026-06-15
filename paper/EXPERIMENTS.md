# Experiment Runbook

This is the GPU-later checklist for the density-likelihood rewrite. The current paper treats all legacy tables and plots as placeholders, so final claims should only use runs produced from this branch.

## Environment

Install the pinned Python dependencies:

```bash
python -m pip install -r requirements.txt
```

Expected data layout:

```text
data/
  sleep/
  seizure/
```

Record the branch commit hash in every experiment note:

```bash
git rev-parse --short HEAD
```

## Dry Run Without GPU

Print the planned core command before launching anything:

```bash
python paper/run_experiments.py \
  --datasets sleep \
  --models gru \
  --objectives density_custom custom seg seg_weighted seg_focal \
  --seeds 0 \
  --epochs 1 \
  --folds 2 \
  --datadir data
```

Generate an executable script for later GPU execution:

```bash
python paper/run_experiments.py \
  --datasets sleep \
  --models gru \
  --objectives density_custom custom seg seg_weighted seg_focal \
  --seeds 0 \
  --epochs 20 \
  --folds 4 \
  --datadir data \
  --write-script paper/generated_runs/sleep_core.sh
```

`paper/generated_runs/` is intentionally ignored; regenerate these scripts instead of committing them.

## Core GPU Runs

Start with sleep and one seed:

```bash
python paper/run_experiments.py \
  --datasets sleep \
  --models gru unet unet_t prectime \
  --objectives density_hard density_gau density_custom hard gau custom seg seg_weighted seg_focal \
  --seeds 0 \
  --epochs 20 \
  --folds 4 \
  --datadir data \
  --execute \
  --stop-on-error
```

If runtime is acceptable, repeat sleep with three seeds:

```bash
python paper/run_experiments.py \
  --datasets sleep \
  --models gru unet unet_t prectime \
  --objectives density_hard density_gau density_custom hard gau custom seg seg_weighted seg_focal \
  --seeds 0 1 2 \
  --epochs 20 \
  --folds 4 \
  --datadir data \
  --skip-existing \
  --execute \
  --stop-on-error
```

Run seizure second, after the sleep matrix is stable:

```bash
python paper/run_experiments.py \
  --datasets seizure \
  --models gru unet unet_t prectime \
  --objectives density_hard density_gau density_custom hard gau custom seg seg_weighted seg_focal \
  --seeds 0 \
  --epochs 20 \
  --folds 4 \
  --datadir data \
  --execute \
  --stop-on-error
```

## Ablations

Likelihood versus MSE under the same kernel:

```bash
python paper/run_experiments.py \
  --datasets sleep \
  --models gru \
  --objectives density_hard hard density_gau gau density_custom custom \
  --seeds 0 1 2 \
  --epochs 20 \
  --folds 4 \
  --datadir data \
  --skip-existing \
  --execute
```

Sparse prior versus no prior-rate bias:

```bash
python paper/run_experiments.py \
  --datasets sleep \
  --models gru \
  --objectives density_custom \
  --seeds 0 1 2 \
  --epochs 20 \
  --folds 4 \
  --density_prior none \
  --datadir data \
  --execute
```

Tolerance-kernel width sensitivity:

```bash
python paper/run_experiments.py \
  --datasets sleep \
  --models gru \
  --objectives density_custom \
  --seeds 0 \
  --epochs 20 \
  --folds 4 \
  --tolerance_scale 0.5 \
  --datadir data \
  --execute
```

Repeat the tolerance run with `--tolerance_scale 1.5` if the first ablation is informative.

## Outputs

Each run writes:

```text
experiments/[dataset]/[model]/[objective]/seed_[seed]/
  models/
  predictions/
  results/run_config.json
  results/fold_results.csv
  results/scores.csv
  results/scores.json
```

Do not commit `experiments/`, model checkpoints, cached predictions, generated shell scripts, or generated paper figures until the final figure set is intentionally selected.

## Plot Regeneration

After runs finish:

```bash
python paper/make_plots.py --dataset sleep --results-root experiments
python paper/make_plots.py --dataset seizure --results-root experiments
```

The required final plot set is:

- target kernels for `density_hard`, `density_gau`, and `density_custom`;
- learned onset/offset rates with detected peaks;
- event AP by tolerance for density, MSE, and segmentation objectives;
- likelihood-vs-MSE ablation under the same smoothing kernel;
- prior-rate ablation for `density_custom`.

## Paper Compile

Compile from `paper/`:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
latexmk -c main.tex
rm -f main.bbl main.pdf
```

The committed tree should contain no LaTeX build products or experiment outputs.
