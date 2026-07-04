#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PY=(uv run --offline python)
RUN_TAG="seizure_highscore_gru3l128h_ds512_bs8_e20"
COMMON=(
  eval.py
  --dataset seizure
  --datadir data/seizure
  --epochs 20
  --folds 4
  --bs 8
  --downsample 512
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
  --score_tolerances 256,512,768
  --score_suffix strict3
)

run_strict_score_if_missing() {
  local objective="$1"
  local pred_dir="experiments/seizure/gru_3l_128h/${objective}/seed_0_${RUN_TAG}/predictions"
  local scores_path="experiments/seizure/gru_3l_128h/${objective}/seed_0_${RUN_TAG}/results/scores_strict3.csv"
  if [[ -s "$scores_path" ]]; then
    echo "Skipping ${objective} ds512 strict3; ${scores_path} already exists."
    return 0
  fi
  if [[ -d "$pred_dir" ]] && compgen -G "$pred_dir/*.npy" > /dev/null; then
    "${PY[@]}" "${COMMON[@]}" --model gru_3l_128h --objective "$objective"
  else
    echo "Skipping ${objective} ds512 strict3: no predictions in ${pred_dir}"
  fi
}

for objective in seg density_hard density_gau density_custom; do
  run_strict_score_if_missing "$objective"
done
