#!/usr/bin/env bash
set -euo pipefail

WAIT_PID="${1:-}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

mkdir -p paper/run_logs
RUN_LOG="${EVENTPDF_DS512_SCORE_LOG:-paper/run_logs/chbmit_ds512_corrected_score_after_wait_trace.log}"

if [[ -n "$WAIT_PID" ]]; then
  echo "Waiting for PID ${WAIT_PID} before starting traced CHB-MIT ds512 scoring."
  while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep 30
  done
fi

export WANDB_MODE="${WANDB_MODE:-offline}"
export EVENTPDF_FORCE_SCORE=1

echo "Starting traced CHB-MIT ds512 normal-tolerance scoring with corrected smoothing grid."
paper/trace_run.sh \
  --intent "Score CHB-MIT ds512 high-capacity GRU rows with the corrected seizure smoothing grid." \
  --setup "dataset=seizure, model=gru_3l_128h, objectives=seg/density_hard/density_gau/density_custom, epochs=20, folds=4, downsample=512, smoothing=none/256/512, force_rescore_existing_scores=true" \
  --expect-log "$RUN_LOG" \
  --expect-artifact experiments/seizure/gru_3l_128h/seg/seed_0_seizure_highscore_gru3l128h_ds512_bs8_e20/results/scores.csv \
  --expect-artifact experiments/seizure/gru_3l_128h/density_hard/seed_0_seizure_highscore_gru3l128h_ds512_bs8_e20/results/scores.csv \
  --expect-artifact experiments/seizure/gru_3l_128h/density_gau/seed_0_seizure_highscore_gru3l128h_ds512_bs8_e20/results/scores.csv \
  --expect-artifact experiments/seizure/gru_3l_128h/density_custom/seed_0_seizure_highscore_gru3l128h_ds512_bs8_e20/results/scores.csv \
  -- bash paper/generated_runs/score_seizure_highscore_gru3l128h_ds512_bs8_e20.sh

echo "Starting traced CHB-MIT ds512 strict one-to-three-second scoring with corrected smoothing grid."
paper/trace_run.sh \
  --intent "Score strict CHB-MIT ds512 high-capacity GRU rows for the BDL paper." \
  --setup "dataset=seizure, model=gru_3l_128h, objectives=seg/density_hard/density_gau/density_custom, score_tolerances=256/512/768, downsample=512, smoothing=none/256/512, score_suffix=strict3" \
  --expect-log "$RUN_LOG" \
  --expect-artifact experiments/seizure/gru_3l_128h/seg/seed_0_seizure_highscore_gru3l128h_ds512_bs8_e20/results/scores_strict3.csv \
  --expect-artifact experiments/seizure/gru_3l_128h/density_hard/seed_0_seizure_highscore_gru3l128h_ds512_bs8_e20/results/scores_strict3.csv \
  --expect-artifact experiments/seizure/gru_3l_128h/density_gau/seed_0_seizure_highscore_gru3l128h_ds512_bs8_e20/results/scores_strict3.csv \
  --expect-artifact experiments/seizure/gru_3l_128h/density_custom/seed_0_seizure_highscore_gru3l128h_ds512_bs8_e20/results/scores_strict3.csv \
  -- bash paper/generated_runs/score_seizure_highscore_gru3l128h_ds512_bs8_e20_strict3.sh

uv run --offline python paper/collect_results.py --results-root experiments
uv run --offline python paper/experiment_status.py --results-root experiments

echo "Traced CHB-MIT ds512 corrected scoring complete."
