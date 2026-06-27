#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 <model> <objective> [pid-to-watch]" >&2
  exit 2
fi

MODEL="$1"
OBJECTIVE="$2"
WAIT_PID="${3:-}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

RUN_TAG="seizure_rescue_lr3e4_ds5_bs8_e20_eval5"
ROW_DIR="experiments/seizure/${MODEL}/${OBJECTIVE}/seed_0_${RUN_TAG}"
FOLD_RESULTS="${ROW_DIR}/results/fold_results.csv"
RUN_LOG="${EVENTPDF_P11_ROW_SCORE_LOG:-paper/run_logs/p11_${MODEL}_${OBJECTIVE}_score_after_ready_trace.log}"

mkdir -p paper/run_logs

echo "Waiting for P11 seizure/${MODEL}/${OBJECTIVE} to finish four folds before row scoring."
while true; do
  if [[ -s "$FOLD_RESULTS" ]]; then
    completed_rows=$(( $(wc -l < "$FOLD_RESULTS") - 1 ))
    if (( completed_rows >= 4 )); then
      break
    fi
    echo "Current completed folds for seizure/${MODEL}/${OBJECTIVE}: ${completed_rows}/4."
  else
    echo "Waiting for ${FOLD_RESULTS}."
  fi
  if [[ -n "$WAIT_PID" ]] && ! kill -0 "$WAIT_PID" 2>/dev/null; then
    echo "Training PID ${WAIT_PID} exited before seizure/${MODEL}/${OBJECTIVE} reached four folds." >&2
    exit 1
  fi
  sleep 60
done

export WANDB_MODE="${WANDB_MODE:-offline}"

paper/trace_run.sh \
  --intent "Score completed P11 CHB-MIT fine-stride seizure/${MODEL}/${OBJECTIVE} row." \
  --setup "dataset=seizure, model=${MODEL}, objective=${OBJECTIVE}, epochs=20, folds=4, downsample=5, smoothing=none/256/512, run_tag=${RUN_TAG}" \
  --expect-log "$RUN_LOG" \
  --expect-artifact "${ROW_DIR}/results/scores.csv" \
  -- bash paper/generated_runs/score_p11_seizure_fine_stride_single_row.sh "$MODEL" "$OBJECTIVE" normal

paper/trace_run.sh \
  --intent "Score strict completed P11 CHB-MIT fine-stride seizure/${MODEL}/${OBJECTIVE} row." \
  --setup "dataset=seizure, model=${MODEL}, objective=${OBJECTIVE}, score_tolerances=256/512/768, downsample=5, smoothing=none/256/512, score_suffix=strict3" \
  --expect-log "$RUN_LOG" \
  --expect-artifact "${ROW_DIR}/results/scores_strict3.csv" \
  -- bash paper/generated_runs/score_p11_seizure_fine_stride_single_row.sh "$MODEL" "$OBJECTIVE" strict3

uv run --offline python paper/collect_results.py --results-root experiments --outdir paper/results/generated
uv run --offline python paper/experiment_status.py --results-root experiments --outdir paper/results/generated

echo "Completed Trace scoring for P11 seizure/${MODEL}/${OBJECTIVE}."
