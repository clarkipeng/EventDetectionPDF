#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PY=(uv run --offline python)
COMMON=(
  eval.py
  --dataset sleep
  --datadir data/sleep
  --epochs 20
  --folds 4
  --bs 32
  --downsample 10
  --agg_feats stat
  --use_cat True
  --normalize True
  --workers 2
  --seed 0
  --run_tag online_smoothing_e20_bs32_eval5
  --wandb False
  --wandb_log_artifacts False
  --device cpu
)

for model in fgru flstm causal_transformer; do
  for objective in density_gau density_custom; do
    "${PY[@]}" "${COMMON[@]}" \
      --model "$model" \
      --objective "$objective" \
      --tune_cutoff_steps 3 \
      --tune_smooth_values none,10 \
      --tune_alternating True
  done

  "${PY[@]}" "${COMMON[@]}" \
    --model "$model" \
    --objective seg \
    --tune_cutoff_values 0.5,0.7 \
    --tune_smooth_values none \
    --tune_alternating True
done
