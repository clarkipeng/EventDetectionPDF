#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <pid-to-wait-for>" >&2
  exit 2
fi

WAIT_PID="$1"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

echo "Waiting for PID ${WAIT_PID} before starting traced CHB-MIT ds256 scoring."
while kill -0 "$WAIT_PID" 2>/dev/null; do
  sleep 60
done

export WANDB_MODE="${WANDB_MODE:-offline}"

echo "Starting traced CHB-MIT ds256 normal-tolerance scoring."
paper/trace_run.sh \
  --intent "Score CHB-MIT ds256 high-capacity GRU rows for the BDL paper." \
  --setup "dataset=seizure, model=gru_3l_128h, objectives=seg/density_hard/density_gau/density_custom, epochs=20, folds=4, downsample=256, smoothing=none/256/512, run_tag=seizure_highscore_gru3l128h_ds256_bs8_e20" \
  --expect-artifact experiments/seizure/gru_3l_128h/seg/seed_0_seizure_highscore_gru3l128h_ds256_bs8_e20/results/scores.csv \
  --expect-artifact experiments/seizure/gru_3l_128h/density_hard/seed_0_seizure_highscore_gru3l128h_ds256_bs8_e20/results/scores.csv \
  --expect-artifact experiments/seizure/gru_3l_128h/density_gau/seed_0_seizure_highscore_gru3l128h_ds256_bs8_e20/results/scores.csv \
  --expect-artifact experiments/seizure/gru_3l_128h/density_custom/seed_0_seizure_highscore_gru3l128h_ds256_bs8_e20/results/scores.csv \
  -- bash paper/generated_runs/score_seizure_highscore_gru3l128h_ds256_bs8_e20.sh

echo "Starting traced CHB-MIT ds256 strict one-to-three-second scoring."
paper/trace_run.sh \
  --intent "Score strict CHB-MIT ds256 high-capacity GRU rows for the BDL paper." \
  --setup "dataset=seizure, model=gru_3l_128h, objectives=seg/density_hard/density_gau/density_custom, score_tolerances=256/512/768, downsample=256, smoothing=none/256/512, score_suffix=strict3" \
  --expect-artifact experiments/seizure/gru_3l_128h/seg/seed_0_seizure_highscore_gru3l128h_ds256_bs8_e20/results/scores_strict3.csv \
  --expect-artifact experiments/seizure/gru_3l_128h/density_hard/seed_0_seizure_highscore_gru3l128h_ds256_bs8_e20/results/scores_strict3.csv \
  --expect-artifact experiments/seizure/gru_3l_128h/density_gau/seed_0_seizure_highscore_gru3l128h_ds256_bs8_e20/results/scores_strict3.csv \
  --expect-artifact experiments/seizure/gru_3l_128h/density_custom/seed_0_seizure_highscore_gru3l128h_ds256_bs8_e20/results/scores_strict3.csv \
  -- bash paper/generated_runs/score_seizure_highscore_gru3l128h_ds256_bs8_e20_strict3.sh

uv run --offline python paper/collect_results.py --results-root experiments
uv run --offline python paper/experiment_status.py --results-root experiments

echo "Traced CHB-MIT ds256 scoring complete."
