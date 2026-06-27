#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 <model> <objective> [normal|strict3]" >&2
  exit 2
fi

MODEL="$1"
OBJECTIVE="$2"
MODE="${3:-normal}"

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
  --model "$MODEL"
  --objective "$OBJECTIVE"
)

case "$MODE" in
  normal)
    scores_path="experiments/seizure/${MODEL}/${OBJECTIVE}/seed_0_${RUN_TAG}/results/scores.csv"
    ;;
  strict3)
    scores_path="experiments/seizure/${MODEL}/${OBJECTIVE}/seed_0_${RUN_TAG}/results/scores_strict3.csv"
    COMMON+=(--score_tolerances 256,512,768 --score_suffix strict3)
    ;;
  *)
    echo "Unknown score mode: ${MODE}" >&2
    exit 2
    ;;
esac

pred_dir="experiments/seizure/${MODEL}/${OBJECTIVE}/seed_0_${RUN_TAG}/predictions"
if [[ -s "$scores_path" ]]; then
  echo "Skipping ${MODEL}/${OBJECTIVE} ${MODE}; ${scores_path} already exists."
  exit 0
fi
if [[ ! -d "$pred_dir" ]] || ! compgen -G "$pred_dir/*.npy" > /dev/null; then
  echo "Cannot score ${MODEL}/${OBJECTIVE} ${MODE}: no predictions in ${pred_dir}" >&2
  exit 1
fi

"${PY[@]}" "${COMMON[@]}"
