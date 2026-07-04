#!/usr/bin/env bash
set -euo pipefail

WAIT_PID="${1:-}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

mkdir -p paper/run_logs

if [[ -n "$WAIT_PID" ]]; then
  echo "Waiting for PID ${WAIT_PID} before starting CHB-MIT stride runs."
  while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep 60
  done
fi

export WANDB_MODE="${WANDB_MODE:-offline}"

echo "Starting CHB-MIT high-capacity downsample=512 run."
bash paper/generated_runs/seizure_highscore_gru3l128h_ds512_bs8_e20_train_only.sh
bash paper/generated_runs/score_seizure_highscore_gru3l128h_ds512_bs8_e20.sh

echo "Starting CHB-MIT high-capacity downsample=256 run."
bash paper/generated_runs/seizure_highscore_gru3l128h_ds256_bs8_e20_train_only.sh
bash paper/generated_runs/score_seizure_highscore_gru3l128h_ds256_bs8_e20.sh

uv run --offline python paper/collect_results.py --results-root experiments
uv run --offline python paper/experiment_status.py --results-root experiments

echo "CHB-MIT stride runs complete."
