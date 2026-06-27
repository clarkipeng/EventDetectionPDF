#!/usr/bin/env bash
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

EVENTPDF_PY="${EVENTPDF_PY:-$REPO_ROOT/.venv/bin/python}"
WAIT_PID="${WAIT_PID:-79478}"
RUN_DIR="$REPO_ROOT/paper/generated_runs/finalize_after_density_gau_$(date -u +%Y%m%dT%H%M%SZ)"
LOG_FILE="$RUN_DIR/finalize.log"
STATUS_FILE="$RUN_DIR/status.txt"
mkdir -p "$RUN_DIR"

echo "waiting for density Gaussian queue pid $WAIT_PID" > "$STATUS_FILE"
{
  echo "wait_pid=$WAIT_PID"
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "python=$EVENTPDF_PY"
} | tee "$RUN_DIR/manifest.txt" | tee -a "$LOG_FILE"

while [[ -n "$WAIT_PID" ]] && kill -0 "$WAIT_PID" >/dev/null 2>&1; do
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) waiting for Gaussian queue pid $WAIT_PID" | tee -a "$LOG_FILE"
  sleep 300
done

echo "running final collection" > "$STATUS_FILE"
"$EVENTPDF_PY" paper/experiment_status.py --results-root experiments 2>&1 | tee -a "$LOG_FILE"
status_rc="${PIPESTATUS[0]}"

"$EVENTPDF_PY" paper/collect_results.py --results-root experiments 2>&1 | tee -a "$LOG_FILE"
collect_rc="${PIPESTATUS[0]}"

"$EVENTPDF_PY" paper/make_plots.py --dataset sleep --results-root experiments 2>&1 | tee -a "$LOG_FILE"
sleep_plot_rc="${PIPESTATUS[0]}"

"$EVENTPDF_PY" paper/make_plots.py --dataset seizure --results-root experiments 2>&1 | tee -a "$LOG_FILE"
seizure_plot_rc="${PIPESTATUS[0]}"

if [[ "$status_rc" -eq 0 && "$collect_rc" -eq 0 && "$sleep_plot_rc" -eq 0 && "$seizure_plot_rc" -eq 0 ]]; then
  echo "complete" > "$STATUS_FILE"
else
  echo "failed status=$status_rc collect=$collect_rc sleep_plot=$sleep_plot_rc seizure_plot=$seizure_plot_rc" > "$STATUS_FILE"
fi

echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$RUN_DIR/manifest.txt" | tee -a "$LOG_FILE"
exit "$(( status_rc || collect_rc || sleep_plot_rc || seizure_plot_rc ))"
