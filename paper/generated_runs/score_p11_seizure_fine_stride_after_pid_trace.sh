#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <pid-to-wait-for>" >&2
  exit 2
fi

WAIT_PID="$1"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

mkdir -p paper/run_logs
RUN_LOG="${EVENTPDF_P11_SCORE_LOG:-paper/run_logs/p11_seizure_fine_stride_score_after_wait_trace.log}"

echo "Waiting for PID ${WAIT_PID} before starting traced P11 CHB-MIT fine-stride scoring."
while kill -0 "$WAIT_PID" 2>/dev/null; do
  sleep 60
done

export WANDB_MODE="${WANDB_MODE:-offline}"

echo "Starting traced P11 CHB-MIT fine-stride normal-tolerance scoring."
paper/trace_run.sh \
  --intent "Score P11 CHB-MIT fine-stride sensitivity rows for the BDL paper." \
  --setup "dataset=seizure, models=gru/unet, objectives=seg/density_hard/density_gau, epochs=20, folds=4, downsample=5, smoothing=none/256/512, run_tag=seizure_rescue_lr3e4_ds5_bs8_e20_eval5" \
  --expect-log "$RUN_LOG" \
  --expect-artifact experiments/seizure/gru/density_hard/seed_0_seizure_rescue_lr3e4_ds5_bs8_e20_eval5/results/scores.csv \
  --expect-artifact experiments/seizure/gru/density_gau/seed_0_seizure_rescue_lr3e4_ds5_bs8_e20_eval5/results/scores.csv \
  --expect-artifact experiments/seizure/gru/seg/seed_0_seizure_rescue_lr3e4_ds5_bs8_e20_eval5/results/scores.csv \
  --expect-artifact experiments/seizure/unet/density_hard/seed_0_seizure_rescue_lr3e4_ds5_bs8_e20_eval5/results/scores.csv \
  --expect-artifact experiments/seizure/unet/density_gau/seed_0_seizure_rescue_lr3e4_ds5_bs8_e20_eval5/results/scores.csv \
  --expect-artifact experiments/seizure/unet/seg/seed_0_seizure_rescue_lr3e4_ds5_bs8_e20_eval5/results/scores.csv \
  -- bash paper/generated_runs/score_seizure_rescue_lr3e4_ds5_bs8.sh

echo "Starting traced P11 CHB-MIT fine-stride strict one-to-three-second scoring."
paper/trace_run.sh \
  --intent "Score strict P11 CHB-MIT fine-stride sensitivity rows for the BDL paper." \
  --setup "dataset=seizure, models=gru/unet, objectives=seg/density_hard/density_gau, score_tolerances=256/512/768, downsample=5, smoothing=none/256/512, score_suffix=strict3" \
  --expect-log "$RUN_LOG" \
  --expect-artifact experiments/seizure/gru/density_hard/seed_0_seizure_rescue_lr3e4_ds5_bs8_e20_eval5/results/scores_strict3.csv \
  --expect-artifact experiments/seizure/gru/density_gau/seed_0_seizure_rescue_lr3e4_ds5_bs8_e20_eval5/results/scores_strict3.csv \
  --expect-artifact experiments/seizure/gru/seg/seed_0_seizure_rescue_lr3e4_ds5_bs8_e20_eval5/results/scores_strict3.csv \
  --expect-artifact experiments/seizure/unet/density_hard/seed_0_seizure_rescue_lr3e4_ds5_bs8_e20_eval5/results/scores_strict3.csv \
  --expect-artifact experiments/seizure/unet/density_gau/seed_0_seizure_rescue_lr3e4_ds5_bs8_e20_eval5/results/scores_strict3.csv \
  --expect-artifact experiments/seizure/unet/seg/seed_0_seizure_rescue_lr3e4_ds5_bs8_e20_eval5/results/scores_strict3.csv \
  -- bash paper/generated_runs/score_seizure_rescue_lr3e4_ds5_bs8_strict3.sh

uv run --offline python paper/collect_results.py --results-root experiments --outdir paper/results/generated
uv run --offline python paper/experiment_status.py --results-root experiments --outdir paper/results/generated

echo "Traced P11 CHB-MIT fine-stride scoring complete."
