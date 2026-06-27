#!/usr/bin/env bash
set -euo pipefail

WAIT_PID="${1:-}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

mkdir -p paper/run_logs

if [[ -n "$WAIT_PID" ]]; then
  echo "Waiting for PID ${WAIT_PID} before scoring P10 online smoothing rows."
  while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep 60
  done
fi

export WANDB_MODE="${WANDB_MODE:-offline}"

echo "Scoring P10 online smoothing rows."
bash paper/generated_runs/score_online_smoothing_e20_bs32_eval5.sh

echo "Refreshing collected results and experiment status."
uv run --offline python paper/collect_results.py --results-root experiments
uv run --offline python paper/experiment_status.py --results-root experiments

echo "P10 online smoothing scoring complete."
