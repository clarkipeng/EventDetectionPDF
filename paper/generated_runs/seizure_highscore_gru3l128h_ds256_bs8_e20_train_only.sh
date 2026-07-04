#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export WANDB_MODE="${WANDB_MODE:-offline}"
mkdir -p paper/run_logs
exec 9>paper/run_logs/seizure_highscore_gru3l128h_ds256_train.lock
flock 9
echo "Starting guarded CHB-MIT ds256 train-only lane at $(date -u +%Y-%m-%dT%H:%M:%SZ)"

run_if_missing() {
  local objective="$1"
  shift
  local result="experiments/seizure/gru_3l_128h/${objective}/seed_0_seizure_highscore_gru3l128h_ds256_bs8_e20/results/fold_results.csv"
  if [[ -s "$result" ]]; then
    echo "Skipping ${objective} downsample=256: found ${result}"
    return 0
  fi
  "$@"
}

# seizure gru_3l_128h seg seed=0
run_if_missing seg uv run --offline python train.py --dataset seizure --model gru_3l_128h --objective seg --datadir data/seizure --epochs 20 --folds 4 --bs 8 --downsample 256 --agg_feats stat --use_cat True --normalize True --workers 1 --seed 0 --density_prior sparse --best_metric mAP --save_all_epochs False --lr 0.001 --weight_decay 0.0 --clip_grad_norm 0.1 --tune_cutoff_steps 11 --tune_smooth_values none,1,10,100,1000 --tune_alternating True --score_after_train False --eval_every 1 --wandb True --wandb_project event-detection-pdf --wandb_log_artifacts True --tolerance_scale 1.0 --wandb_group P12-seizure-highscore-ds256 --wandb_tags bdl-paper,seizure,P12,highscore-ds256 --run_tag seizure_highscore_gru3l128h_ds256_bs8_e20 --device cuda

# seizure gru_3l_128h density_hard seed=0
run_if_missing density_hard uv run --offline python train.py --dataset seizure --model gru_3l_128h --objective density_hard --datadir data/seizure --epochs 20 --folds 4 --bs 8 --downsample 256 --agg_feats stat --use_cat True --normalize True --workers 1 --seed 0 --density_prior sparse --best_metric mAP --save_all_epochs False --lr 0.001 --weight_decay 0.0 --clip_grad_norm 0.1 --tune_cutoff_steps 11 --tune_smooth_values none,1,10,100,1000 --tune_alternating True --score_after_train False --eval_every 1 --wandb True --wandb_project event-detection-pdf --wandb_log_artifacts True --tolerance_scale 1.0 --wandb_group P12-seizure-highscore-ds256 --wandb_tags bdl-paper,seizure,P12,highscore-ds256 --run_tag seizure_highscore_gru3l128h_ds256_bs8_e20 --device cuda

# seizure gru_3l_128h density_gau seed=0
run_if_missing density_gau uv run --offline python train.py --dataset seizure --model gru_3l_128h --objective density_gau --datadir data/seizure --epochs 20 --folds 4 --bs 8 --downsample 256 --agg_feats stat --use_cat True --normalize True --workers 1 --seed 0 --density_prior sparse --best_metric mAP --save_all_epochs False --lr 0.001 --weight_decay 0.0 --clip_grad_norm 0.1 --tune_cutoff_steps 11 --tune_smooth_values none,1,10,100,1000 --tune_alternating True --score_after_train False --eval_every 1 --wandb True --wandb_project event-detection-pdf --wandb_log_artifacts True --tolerance_scale 1.0 --wandb_group P12-seizure-highscore-ds256 --wandb_tags bdl-paper,seizure,P12,highscore-ds256 --run_tag seizure_highscore_gru3l128h_ds256_bs8_e20 --device cuda

# seizure gru_3l_128h density_custom seed=0
run_if_missing density_custom uv run --offline python train.py --dataset seizure --model gru_3l_128h --objective density_custom --datadir data/seizure --epochs 20 --folds 4 --bs 8 --downsample 256 --agg_feats stat --use_cat True --normalize True --workers 1 --seed 0 --density_prior sparse --best_metric mAP --save_all_epochs False --lr 0.001 --weight_decay 0.0 --clip_grad_norm 0.1 --tune_cutoff_steps 11 --tune_smooth_values none,1,10,100,1000 --tune_alternating True --score_after_train False --eval_every 1 --wandb True --wandb_project event-detection-pdf --wandb_log_artifacts True --tolerance_scale 1.0 --wandb_group P12-seizure-highscore-ds256 --wandb_tags bdl-paper,seizure,P12,highscore-ds256 --run_tag seizure_highscore_gru3l128h_ds256_bs8_e20 --device cuda

echo "Finished guarded CHB-MIT ds256 train-only lane at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
