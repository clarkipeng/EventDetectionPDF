#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <pid-to-wait-for>" >&2
  exit 2
fi

WAIT_PID="$1"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

echo "Waiting for PID ${WAIT_PID} before starting traced Bowshock/Fraud point-event ablations."
while kill -0 "$WAIT_PID" 2>/dev/null; do
  sleep 60
done

export WANDB_MODE="${WANDB_MODE:-offline}"

echo "Starting traced Bowshock point-event ablation."
paper/trace_run.sh \
  --intent "Run Bowshock point-event ablation for the BDL paper." \
  --setup "dataset=bowshock, model=gru, objectives=density_hard/density_gau/density_custom/seg, epochs=20, folds=4, downsample=10, run_tag=point_event_ablation_e20" \
  --expect-artifact experiments/bowshock/gru/density_hard/seed_0_point_event_ablation_e20/results/scores.csv \
  --expect-artifact experiments/bowshock/gru/density_gau/seed_0_point_event_ablation_e20/results/scores.csv \
  --expect-artifact experiments/bowshock/gru/density_custom/seed_0_point_event_ablation_e20/results/scores.csv \
  --expect-artifact experiments/bowshock/gru/seg/seed_0_point_event_ablation_e20/results/scores.csv \
  -- bash paper/generated_runs/bowshock_point_event_ablation_e20.sh

echo "Starting traced Fraud point-event ablation."
paper/trace_run.sh \
  --intent "Run Fraud point-event ablation for the BDL paper." \
  --setup "dataset=fraud, model=gru, objectives=density_hard/density_gau/density_custom/seg, epochs=20, folds=4, downsample=1, run_tag=point_event_ablation_e20" \
  --expect-artifact experiments/fraud/gru/density_hard/seed_0_point_event_ablation_e20/results/scores.csv \
  --expect-artifact experiments/fraud/gru/density_gau/seed_0_point_event_ablation_e20/results/scores.csv \
  --expect-artifact experiments/fraud/gru/density_custom/seed_0_point_event_ablation_e20/results/scores.csv \
  --expect-artifact experiments/fraud/gru/seg/seed_0_point_event_ablation_e20/results/scores.csv \
  -- bash paper/generated_runs/fraud_point_event_ablation_e20.sh

uv run --offline python paper/collect_results.py --results-root experiments
uv run --offline python paper/experiment_status.py --results-root experiments

echo "Traced point-event ablations complete."
