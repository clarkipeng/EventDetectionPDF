#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 1 ]]; then
  echo "Usage: $0 [pid-to-watch]" >&2
  exit 2
fi

WAIT_PID="${1:-}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

RUN_TAG="transformer_main_candidate_lr1e3_e20_bs32_eval5"
MODEL="transformer_4l_64h_4a"
RUN_LOG="${EVENTPDF_P15_SCORE_LOG:-paper/run_logs/p15_transformer_candidate_score_after_ready_trace.log}"
OBJECTIVES=(density_gau density_custom seg)

mkdir -p paper/run_logs
export WANDB_MODE="${WANDB_MODE:-offline}"

wait_for_row() {
  local objective="$1"
  local row_dir="experiments/sleep/${MODEL}/${objective}/seed_0_${RUN_TAG}"
  local fold_results="${row_dir}/results/fold_results.csv"

  echo "Waiting for P15 sleep/${MODEL}/${objective} to finish four folds before row scoring."
  while true; do
    if [[ -s "$fold_results" ]]; then
      completed_rows=$(( $(wc -l < "$fold_results") - 1 ))
      if (( completed_rows >= 4 )); then
        echo "P15 sleep/${MODEL}/${objective} has ${completed_rows}/4 folds."
        break
      fi
      echo "Current completed folds for sleep/${MODEL}/${objective}: ${completed_rows}/4."
    else
      echo "Waiting for ${fold_results}."
    fi
    if [[ -n "$WAIT_PID" ]] && ! kill -0 "$WAIT_PID" 2>/dev/null; then
      echo "Training PID ${WAIT_PID} exited before sleep/${MODEL}/${objective} reached four folds." >&2
      exit 1
    fi
    sleep 60
  done
}

score_row() {
  local objective="$1"
  local row_dir="experiments/sleep/${MODEL}/${objective}/seed_0_${RUN_TAG}"

  paper/trace_run.sh \
    --intent "Score completed P15 Transformer main-table candidate sleep/${MODEL}/${objective} row." \
    --setup "dataset=sleep, model=${MODEL}, objective=${objective}, seed=0, epochs=20, folds=4, downsample=10, learning_rate=0.001, smoothing=none/1/10/100/1000, run_tag=${RUN_TAG}" \
    --expect-log "$RUN_LOG" \
    --expect-artifact "${row_dir}/results/scores.csv" \
    -- uv run --offline python eval.py \
      --dataset sleep \
      --model "$MODEL" \
      --objective "$objective" \
      --datadir data/sleep \
      --epochs 20 \
      --folds 4 \
      --bs 32 \
      --downsample 10 \
      --agg_feats stat \
      --use_cat True \
      --normalize True \
      --workers 1 \
      --seed 0 \
      --run_tag "$RUN_TAG" \
      --tune_cutoff_steps 11 \
      --tune_smooth_values none,1,10,100,1000 \
      --tune_alternating True \
      --wandb True \
      --wandb_project event-detection-pdf \
      --wandb_mode offline \
      --wandb_group P15-transformer-main-candidate-score \
      --wandb_tags paper-density-likelihood-rewrite,sleep,P15,transformer-main-candidate,score,trace \
      --device cpu

  uv run --offline python paper/collect_results.py --results-root experiments --outdir paper/results/generated
  uv run --offline python paper/experiment_status.py --results-root experiments --outdir paper/results/generated
}

for objective in "${OBJECTIVES[@]}"; do
  wait_for_row "$objective"
  score_row "$objective"
done

echo "Completed Trace scoring for P15 Transformer candidate rows."
