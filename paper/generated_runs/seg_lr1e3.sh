#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# sleep gru seg seed=0
/home/ec2-user/EventDetectionPDF/.venv/bin/python train.py --dataset sleep --model gru --objective seg --datadir data/sleep --epochs 20 --folds 4 --bs 32 --downsample 10 --agg_feats stat --use_cat True --normalize True --workers 4 --seed 0 --density_prior sparse --best_metric mAP --save_all_epochs False --lr 0.001 --weight_decay 0.0 --clip_grad_norm 0.1 --tune_cutoff_steps 11 --tune_smooth_values none,1,10,100,1000 --tune_alternating True --score_after_train False --eval_every 5 --wandb False --wandb_project event-detection-pdf --wandb_log_artifacts True --tolerance_scale 1.0 --run_tag seg_lr1e3_e20_bs32_eval5 --device cuda

# sleep gru seg_weighted seed=0
/home/ec2-user/EventDetectionPDF/.venv/bin/python train.py --dataset sleep --model gru --objective seg_weighted --datadir data/sleep --epochs 20 --folds 4 --bs 32 --downsample 10 --agg_feats stat --use_cat True --normalize True --workers 4 --seed 0 --density_prior sparse --best_metric mAP --save_all_epochs False --lr 0.001 --weight_decay 0.0 --clip_grad_norm 0.1 --tune_cutoff_steps 11 --tune_smooth_values none,1,10,100,1000 --tune_alternating True --score_after_train False --eval_every 5 --wandb False --wandb_project event-detection-pdf --wandb_log_artifacts True --tolerance_scale 1.0 --run_tag seg_lr1e3_e20_bs32_eval5 --device cuda

# sleep gru seg_focal seed=0
/home/ec2-user/EventDetectionPDF/.venv/bin/python train.py --dataset sleep --model gru --objective seg_focal --datadir data/sleep --epochs 20 --folds 4 --bs 32 --downsample 10 --agg_feats stat --use_cat True --normalize True --workers 4 --seed 0 --density_prior sparse --best_metric mAP --save_all_epochs False --lr 0.001 --weight_decay 0.0 --clip_grad_norm 0.1 --tune_cutoff_steps 11 --tune_smooth_values none,1,10,100,1000 --tune_alternating True --score_after_train False --eval_every 5 --wandb False --wandb_project event-detection-pdf --wandb_log_artifacts True --tolerance_scale 1.0 --run_tag seg_lr1e3_e20_bs32_eval5 --device cuda
