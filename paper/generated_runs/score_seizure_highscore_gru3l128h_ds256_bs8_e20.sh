#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PY=(uv run --offline python)
RUN_TAG="seizure_highscore_gru3l128h_ds256_bs8_e20"
COMMON=(
  eval.py
  --dataset seizure
  --datadir data/seizure
  --epochs 20
  --folds 4
  --bs 8
  --downsample 256
  --agg_feats stat
  --use_cat True
  --normalize True
  --workers 2
  --seed 0
  --run_tag "$RUN_TAG"
  --wandb False
  --wandb_log_artifacts False
  --device cpu
  --tune_cutoff_steps 11
  --tune_smooth_values none,256,512
  --tune_alternating True
)

run_score_if_missing() {
  local objective="$1"
  local scores_path="experiments/seizure/gru_3l_128h/${objective}/seed_0_${RUN_TAG}/results/scores.csv"
  if [[ -s "$scores_path" ]]; then
    echo "Skipping ${objective}; ${scores_path} already exists."
    return 0
  fi
  "${PY[@]}" "${COMMON[@]}" --model gru_3l_128h --objective "$objective"
}

for objective in seg density_hard density_gau density_custom; do
  run_score_if_missing "$objective"
done
