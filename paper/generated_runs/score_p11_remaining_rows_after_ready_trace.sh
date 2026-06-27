#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

ROWS=(
  "gru seg"
  "unet density_hard"
  "unet density_gau"
  "unet seg"
)

echo "Starting queued P11 row-level scoring for rows after gru/density_gau."
for row in "${ROWS[@]}"; do
  read -r model objective <<< "$row"
  echo "Queue waiting/scoring P11 seizure/${model}/${objective}."
  paper/generated_runs/score_p11_row_after_ready_trace.sh "$model" "$objective"
done
echo "Completed queued P11 row-level scoring."
