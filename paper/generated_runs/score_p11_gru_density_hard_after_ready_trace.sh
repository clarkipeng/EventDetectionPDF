#!/usr/bin/env bash
set -euo pipefail

WAIT_PID="${1:-}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

RUN_TAG="seizure_rescue_lr3e4_ds5_bs8_e20_eval5"
FOLD_RESULTS="experiments/seizure/gru/density_hard/seed_0_${RUN_TAG}/results/fold_results.csv"
RUN_LOG="${EVENTPDF_P11_ROW_SCORE_LOG:-paper/run_logs/p11_gru_density_hard_score_after_ready_trace.log}"

mkdir -p paper/run_logs

echo "Waiting for P11 seizure/gru/density_hard to finish four folds before row scoring."
while true; do
  if [[ -s "$FOLD_RESULTS" ]]; then
    completed_rows=$(( $(wc -l < "$FOLD_RESULTS") - 1 ))
    if (( completed_rows >= 4 )); then
      break
    fi
    echo "Current completed folds for seizure/gru/density_hard: ${completed_rows}/4."
  else
    echo "Waiting for ${FOLD_RESULTS}."
  fi
  if [[ -n "$WAIT_PID" ]] && ! kill -0 "$WAIT_PID" 2>/dev/null; then
    echo "Training PID ${WAIT_PID} exited before seizure/gru/density_hard reached four folds." >&2
    exit 1
  fi
  sleep 60
done

export WANDB_MODE="${WANDB_MODE:-offline}"

paper/trace_run.sh \
  --intent "Score completed P11 CHB-MIT fine-stride seizure/gru/density_hard row." \
  --setup "dataset=seizure, model=gru, objective=density_hard, epochs=20, folds=4, downsample=5, smoothing=none/256/512, run_tag=${RUN_TAG}" \
  --expect-log "$RUN_LOG" \
  --expect-artifact experiments/seizure/gru/density_hard/seed_0_${RUN_TAG}/results/scores.csv \
  -- bash paper/generated_runs/score_p11_seizure_fine_stride_single_row.sh gru density_hard normal

paper/trace_run.sh \
  --intent "Score strict completed P11 CHB-MIT fine-stride seizure/gru/density_hard row." \
  --setup "dataset=seizure, model=gru, objective=density_hard, score_tolerances=256/512/768, downsample=5, smoothing=none/256/512, score_suffix=strict3" \
  --expect-log "$RUN_LOG" \
  --expect-artifact experiments/seizure/gru/density_hard/seed_0_${RUN_TAG}/results/scores_strict3.csv \
  -- bash paper/generated_runs/score_p11_seizure_fine_stride_single_row.sh gru density_hard strict3

uv run --offline python paper/collect_results.py --results-root experiments --outdir paper/results/generated
uv run --offline python paper/experiment_status.py --results-root experiments --outdir paper/results/generated

echo "Completed Trace scoring for P11 seizure/gru/density_hard."
