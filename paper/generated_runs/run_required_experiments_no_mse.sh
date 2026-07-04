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

RUN_ID="${RUN_ID:-required_no_mse_$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="$REPO_ROOT/paper/generated_runs/$RUN_ID"
STATUS_JSONL="$RUN_DIR/experiment_events.jsonl"
LOG_FILE="$RUN_DIR/experiment_output.log"
mkdir -p "$RUN_DIR"

json_log() {
  local event="$1"
  local phase="${2:-}"
  local return_code="${3:-}"
  local command="${4:-}"
  EVENT="$event" PHASE="$phase" RETURN_CODE="$return_code" COMMAND="$command" \
    "$EVENTPDF_PY" - "$STATUS_JSONL" <<'PY'
import datetime as _dt
import json
import os
import sys

row = {
    "time_utc": _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
    "event": os.environ.get("EVENT", ""),
    "phase": os.environ.get("PHASE", ""),
}
if os.environ.get("RETURN_CODE") not in {None, ""}:
    row["return_code"] = int(os.environ["RETURN_CODE"])
if os.environ.get("COMMAND"):
    row["command"] = os.environ["COMMAND"]
with open(sys.argv[1], "a", encoding="utf-8") as handle:
    handle.write(json.dumps(row, sort_keys=True) + "\n")
PY
}

run_phase() {
  local phase="$1"
  shift
  local command=("$@")
  local command_string
  command_string="$(printf '%q ' "${command[@]}")"

  echo
  echo "===== $phase ====="
  echo "$command_string"
  json_log "phase_start" "$phase" "" "$command_string"

  "${command[@]}" 2>&1 | tee -a "$LOG_FILE"
  local rc="${PIPESTATUS[0]}"

  json_log "phase_end" "$phase" "$rc" "$command_string"
  if [[ "$rc" -ne 0 ]]; then
    echo "Phase failed with return code $rc: $phase" | tee -a "$LOG_FILE"
  fi
  return 0
}

BASE_ARGS=(
  --python "$EVENTPDF_PY"
  --datasets sleep
  --datadir data/sleep
  --epochs 20
  --folds 4
  --bs 32
  --workers 1
  --device cuda
  --lr 0.003
  --score_after_train True
  --eval_every 5
  --tune_cutoff_steps 3
  --tune_smooth_values none,10
  --tune_alternating True
  --wandb True
  --wandb_project event-detection-pdf
  --wandb_mode "$WANDB_MODE"
  --wandb_log_artifacts True
  --skip-existing
  --execute
)

json_log "run_start" "all" "" "RUN_ID=$RUN_ID"
{
  echo "run_id=$RUN_ID"
  echo "repo_root=$REPO_ROOT"
  echo "python=$EVENTPDF_PY"
  echo "status_jsonl=$STATUS_JSONL"
  echo "log_file=$LOG_FILE"
  echo "commit=$(git rev-parse --short HEAD 2>/dev/null || true)"
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} | tee "$RUN_DIR/manifest.txt" | tee -a "$LOG_FILE"

run_phase "sleep_core_architectures_no_mse" \
  "$EVENTPDF_PY" paper/run_experiments.py \
  "${BASE_ARGS[@]}" \
  --models gru unet unet_t prectime \
  --objectives density_hard seg \
  --seeds 0 1 2 \
  --run_tag objective_e20_bs32_eval5 \
  --wandb_group P3-main-table-no-mse \
  --wandb_tags paper-density-likelihood-rewrite,sleep,P3,no-mse

run_phase "sleep_gru_density_and_segmentation_ablations" \
  "$EVENTPDF_PY" paper/run_experiments.py \
  "${BASE_ARGS[@]}" \
  --models gru \
  --objectives density_gau density_custom seg_weighted seg_focal \
  --seeds 0 1 2 \
  --run_tag objective_e20_bs32_eval5 \
  --wandb_group P2-P4-gru-ablation \
  --wandb_tags paper-density-likelihood-rewrite,sleep,kernel-seg-ablation

run_phase "sleep_prior_sparse_reference" \
  "$EVENTPDF_PY" paper/run_experiments.py \
  "${BASE_ARGS[@]}" \
  --models gru \
  --objectives density_custom \
  --seeds 0 1 2 \
  --density_prior sparse \
  --run_tag prior_sparse_e20_bs32_eval5 \
  --wandb_group P5-prior-ablation \
  --wandb_tags paper-density-likelihood-rewrite,sleep,P5,prior-sparse

run_phase "sleep_prior_none" \
  "$EVENTPDF_PY" paper/run_experiments.py \
  "${BASE_ARGS[@]}" \
  --models gru \
  --objectives density_custom \
  --seeds 0 1 2 \
  --density_prior none \
  --run_tag prior_none_e20_bs32_eval5 \
  --wandb_group P5-prior-ablation \
  --wandb_tags paper-density-likelihood-rewrite,sleep,P5,prior-none

run_phase "sleep_tolerance_width_0p5" \
  "$EVENTPDF_PY" paper/run_experiments.py \
  "${BASE_ARGS[@]}" \
  --models gru \
  --objectives density_custom \
  --seeds 0 \
  --tolerance_scale 0.5 \
  --run_tag tol05_e20_bs32_eval5 \
  --wandb_group P6-target-width-ablation \
  --wandb_tags paper-density-likelihood-rewrite,sleep,P6,tol05

run_phase "sleep_tolerance_width_1p5" \
  "$EVENTPDF_PY" paper/run_experiments.py \
  "${BASE_ARGS[@]}" \
  --models gru \
  --objectives density_custom \
  --seeds 0 \
  --tolerance_scale 1.5 \
  --run_tag tol15_e20_bs32_eval5 \
  --wandb_group P6-target-width-ablation \
  --wandb_tags paper-density-likelihood-rewrite,sleep,P6,tol15

run_phase "sleep_online_models_no_mse" \
  "$EVENTPDF_PY" paper/run_experiments.py \
  "${BASE_ARGS[@]}" \
  --models fgru flstm causal_transformer \
  --objectives density_hard seg \
  --seeds 0 1 2 \
  --run_tag online_e20_bs32_eval5 \
  --wandb_group P7-online-ablation-no-mse \
  --wandb_tags paper-density-likelihood-rewrite,sleep,P7,no-mse

if "$EVENTPDF_PY" paper/check_data.py --datasets seizure --seizure-dir data/seizure >/dev/null 2>&1; then
  run_phase "seizure_replication_no_mse" \
    "$EVENTPDF_PY" paper/run_experiments.py \
    --python "$EVENTPDF_PY" \
    --datasets seizure \
    --datadir data/seizure \
    --models gru unet \
    --objectives density_hard seg \
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
    --wandb_group P8-seizure-replication-no-mse \
    --wandb_tags paper-density-likelihood-rewrite,seizure,P8,no-mse \
    --wandb_log_artifacts True \
    --skip-existing \
    --execute \
    --run_tag seizure_main_e20_bs32_eval5
else
  echo "Skipping seizure_replication_no_mse: data/seizure is missing required files." | tee -a "$LOG_FILE"
  json_log "phase_skipped" "seizure_replication_no_mse" "" "missing data/seizure"
fi

json_log "run_end" "all" "" "RUN_ID=$RUN_ID"
echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$RUN_DIR/manifest.txt" | tee -a "$LOG_FILE"
