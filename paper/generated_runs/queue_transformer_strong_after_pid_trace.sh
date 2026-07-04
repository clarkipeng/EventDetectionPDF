#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 1 ]]; then
  echo "Usage: $0 [pid-to-watch]" >&2
  exit 2
fi

WAIT_PID="${1:-}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export UV_CACHE_DIR="${UV_CACHE_DIR:-$REPO_ROOT/.uv-cache}"
export WANDB_MODE="${WANDB_MODE:-offline}"

if [[ -n "$WAIT_PID" ]]; then
  echo "Queueing P16 strong Transformer candidate after PID ${WAIT_PID}."
  while kill -0 "$WAIT_PID" >/dev/null 2>&1; do
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) waiting for PID ${WAIT_PID}"
    sleep 300
  done
fi

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) starting Trace-wrapped P16 strong Transformer candidate"

uv run --offline python paper/run_experiments.py \
  --python "uv run --offline python" \
  --datasets sleep \
  --datadir data/sleep \
  --models transformer_6l_128h_8a_10d \
  --objectives density_hard density_gau density_custom seg \
  --seeds 0 \
  --epochs 20 \
  --folds 4 \
  --bs 8 \
  --downsample 10 \
  --workers 2 \
  --device cuda \
  --lr 0.0005 \
  --weight_decay 0.01 \
  --clip_grad_norm 0.1 \
  --score_after_train False \
  --eval_every 5 \
  --tune_cutoff_steps 11 \
  --tune_smooth_values auto \
  --tune_alternating True \
  --wandb True \
  --wandb_project event-detection-pdf \
  --wandb_mode offline \
  --wandb_group P16-transformer-strong \
  --wandb_tags paper-density-likelihood-rewrite,sleep,P16,transformer-strong,trace \
  --wandb_log_artifacts True \
  --run_tag transformer_strong_lr5e4_wd1e2_e20_bs8_eval5 \
  --skip-existing \
  --execute \
  --stop-on-error

echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
