#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <pid-to-wait-for>" >&2
  exit 2
fi

WAIT_PID="$1"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

echo "Waiting for PID ${WAIT_PID} before starting Bowshock/Fraud point-event ablations."
while kill -0 "$WAIT_PID" 2>/dev/null; do
  sleep 60
done

echo "Starting Bowshock point-event ablation."
bash paper/generated_runs/bowshock_point_event_ablation_e20.sh

echo "Starting Fraud point-event ablation."
bash paper/generated_runs/fraud_point_event_ablation_e20.sh

uv run --offline python paper/collect_results.py --results-root experiments
uv run --offline python paper/experiment_status.py --results-root experiments

echo "Point-event ablations complete."
