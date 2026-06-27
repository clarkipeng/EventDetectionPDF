#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

WAIT_PID="${1:-3992966}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$REPO_ROOT/.uv-cache}"
export WANDB_MODE="${WANDB_MODE:-offline}"

echo "Queueing P9 Transformer ablation after PID ${WAIT_PID}."
echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

while kill -0 "$WAIT_PID" >/dev/null 2>&1; do
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) waiting for PID ${WAIT_PID}"
  sleep 300
done

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) starting Trace-wrapped P9 Transformer ablation"

uv run --offline python paper/run_experiments.py \
  --python "uv run --offline python" \
  --datasets sleep \
  --datadir data/sleep \
  --models transformer_4l_64h_4a \
  --objectives density_hard seg \
  --seeds 0 1 2 \
  --epochs 20 \
  --folds 4 \
  --bs 32 \
  --downsample 10 \
  --workers 2 \
  --device cuda \
  --lr 0.003 \
  --score_after_train True \
  --eval_every 5 \
  --tune_cutoff_steps 11 \
  --tune_smooth_values auto \
  --tune_alternating True \
  --wandb True \
  --wandb_project event-detection-pdf \
  --wandb_mode offline \
  --wandb_group P9-transformer-offline-trace \
  --wandb_tags paper-density-likelihood-rewrite,sleep,P9,transformer-offline,trace \
  --wandb_log_artifacts True \
  --run_tag transformer_offline_e20_bs32_eval5 \
  --skip-existing \
  --execute \
  --stop-on-error

echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
