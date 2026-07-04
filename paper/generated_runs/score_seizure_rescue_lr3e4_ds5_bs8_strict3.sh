#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PY=(uv run --offline python)
RUN_TAG="seizure_rescue_lr3e4_ds5_bs8_e20_eval5"
COMMON=(
  eval.py
  --dataset seizure
  --datadir data/seizure
  --epochs 20
  --folds 4
  --bs 8
  --downsample 5
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

run_strict_score_if_ready() {
  local model="$1"
  local objective="$2"
  local pred_dir="experiments/seizure/${model}/${objective}/seed_0_${RUN_TAG}/predictions"
  local scores_path="experiments/seizure/${model}/${objective}/seed_0_${RUN_TAG}/results/scores_strict3.csv"
  if [[ -s "$scores_path" ]]; then
    echo "Skipping ${model}/${objective} strict3; ${scores_path} already exists."
    return 0
  fi
  if [[ -d "$pred_dir" ]] && compgen -G "$pred_dir/*.npy" > /dev/null; then
    "${PY[@]}" "${COMMON[@]}" --model "$model" --objective "$objective"
  else
    echo "Skipping ${model}/${objective} strict3: no predictions in ${pred_dir}"
  fi
}

for model in gru unet; do
  for objective in density_hard density_gau seg; do
    run_strict_score_if_ready "$model" "$objective"
  done
done
