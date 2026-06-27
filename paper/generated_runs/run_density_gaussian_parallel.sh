#!/usr/bin/env bash
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export UV_CACHE_DIR="${UV_CACHE_DIR:-$REPO_ROOT/.uv-cache}"
export WANDB_MODE="${WANDB_MODE:-offline}"
EVENTPDF_PY="${EVENTPDF_PY:-$REPO_ROOT/.venv/bin/python}"

if [[ ! -x "$EVENTPDF_PY" ]]; then
  echo "Python executable not found: $EVENTPDF_PY" >&2
  exit 1
fi

RUN_ID="${RUN_ID:-density_gau_parallel_$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="$REPO_ROOT/paper/generated_runs/$RUN_ID"
LOG_FILE="$RUN_DIR/experiment_output.log"
STATUS_FILE="$RUN_DIR/status.txt"
mkdir -p "$RUN_DIR"

{
  echo "run_id=$RUN_ID"
  echo "repo_root=$REPO_ROOT"
  echo "python=$EVENTPDF_PY"
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "commit=$(git rev-parse --short HEAD 2>/dev/null || true)"
} | tee "$RUN_DIR/manifest.txt" | tee -a "$LOG_FILE"

echo "running sleep density_gau architecture lane" > "$STATUS_FILE"
"$EVENTPDF_PY" paper/run_experiments.py \
  --python "$EVENTPDF_PY" \
  --datasets sleep \
  --datadir data/sleep \
  --models unet unet_t prectime \
  --objectives density_gau \
  --seeds 0 1 2 \
  --epochs 20 \
  --folds 4 \
  --bs 32 \
  --workers 1 \
  --device cuda \
  --lr 0.003 \
  --score_after_train True \
  --eval_every 5 \
  --tune_cutoff_steps 3 \
  --tune_smooth_values none,10 \
  --tune_alternating True \
  --wandb True \
  --wandb_project event-detection-pdf \
  --wandb_mode "$WANDB_MODE" \
  --wandb_group P4G-bdl-gaussian-architecture \
  --wandb_tags paper-density-likelihood-rewrite,sleep,P4G,parallel-lane \
  --wandb_log_artifacts True \
  --skip-existing \
  --execute \
  --run_tag objective_e20_bs32_eval5 2>&1 | tee -a "$LOG_FILE"
sleep_rc="${PIPESTATUS[0]}"

seizure_rc=0
if "$EVENTPDF_PY" paper/check_data.py --datasets seizure --seizure-dir data/seizure >/dev/null 2>&1; then
  echo "running seizure density_gau lane" > "$STATUS_FILE"
  "$EVENTPDF_PY" paper/run_experiments.py \
    --python "$EVENTPDF_PY" \
    --datasets seizure \
    --datadir data/seizure \
    --models gru unet \
    --objectives density_gau \
    --seeds 0 \
    --epochs 20 \
    --folds 4 \
    --bs 32 \
    --workers 1 \
    --device cuda \
    --lr 0.003 \
    --score_after_train True \
    --eval_every 5 \
    --tune_cutoff_steps 3 \
    --tune_smooth_values none,10 \
    --tune_alternating True \
    --wandb True \
    --wandb_project event-detection-pdf \
    --wandb_mode "$WANDB_MODE" \
    --wandb_group P8G-seizure-bdl-gaussian \
    --wandb_tags paper-density-likelihood-rewrite,seizure,P8G,parallel-lane \
    --wandb_log_artifacts True \
    --skip-existing \
    --execute \
    --run_tag seizure_main_e20_bs32_eval5 2>&1 | tee -a "$LOG_FILE"
  seizure_rc="${PIPESTATUS[0]}"
fi

if [[ "$sleep_rc" -eq 0 && "$seizure_rc" -eq 0 ]]; then
  echo "complete" > "$STATUS_FILE"
else
  echo "failed sleep_rc=$sleep_rc seizure_rc=$seizure_rc" > "$STATUS_FILE"
fi

echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$RUN_DIR/manifest.txt" | tee -a "$LOG_FILE"
exit "$(( sleep_rc != 0 ? sleep_rc : seizure_rc ))"
