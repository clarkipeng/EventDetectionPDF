#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# seizure gru density_hard seed=0
/home/ec2-user/EventDetectionPDF/.venv/bin/python train.py --dataset seizure --model gru --objective density_hard --datadir data/seizure --epochs 20 --folds 4 --bs 32 --downsample 5 --agg_feats stat --use_cat True --normalize True --workers 4 --seed 0 --density_prior sparse --best_metric mAP --save_all_epochs False --lr 0.0003 --weight_decay 0.0 --clip_grad_norm 0.1 --tune_cutoff_steps 11 --tune_smooth_values none,1,10,100,1000 --tune_alternating True --score_after_train False --eval_every 5 --wandb False --wandb_project event-detection-pdf --wandb_log_artifacts True --tolerance_scale 1.0 --run_tag seizure_rescue_lr3e4_ds5_e20_bs32_eval5 --device cuda

# seizure gru density_gau seed=0
/home/ec2-user/EventDetectionPDF/.venv/bin/python train.py --dataset seizure --model gru --objective density_gau --datadir data/seizure --epochs 20 --folds 4 --bs 32 --downsample 5 --agg_feats stat --use_cat True --normalize True --workers 4 --seed 0 --density_prior sparse --best_metric mAP --save_all_epochs False --lr 0.0003 --weight_decay 0.0 --clip_grad_norm 0.1 --tune_cutoff_steps 11 --tune_smooth_values none,1,10,100,1000 --tune_alternating True --score_after_train False --eval_every 5 --wandb False --wandb_project event-detection-pdf --wandb_log_artifacts True --tolerance_scale 1.0 --run_tag seizure_rescue_lr3e4_ds5_e20_bs32_eval5 --device cuda

# seizure gru seg seed=0
/home/ec2-user/EventDetectionPDF/.venv/bin/python train.py --dataset seizure --model gru --objective seg --datadir data/seizure --epochs 20 --folds 4 --bs 32 --downsample 5 --agg_feats stat --use_cat True --normalize True --workers 4 --seed 0 --density_prior sparse --best_metric mAP --save_all_epochs False --lr 0.0003 --weight_decay 0.0 --clip_grad_norm 0.1 --tune_cutoff_steps 11 --tune_smooth_values none,1,10,100,1000 --tune_alternating True --score_after_train False --eval_every 5 --wandb False --wandb_project event-detection-pdf --wandb_log_artifacts True --tolerance_scale 1.0 --run_tag seizure_rescue_lr3e4_ds5_e20_bs32_eval5 --device cuda

# seizure unet density_hard seed=0
/home/ec2-user/EventDetectionPDF/.venv/bin/python train.py --dataset seizure --model unet --objective density_hard --datadir data/seizure --epochs 20 --folds 4 --bs 32 --downsample 5 --agg_feats stat --use_cat True --normalize True --workers 4 --seed 0 --density_prior sparse --best_metric mAP --save_all_epochs False --lr 0.0003 --weight_decay 0.0 --clip_grad_norm 0.1 --tune_cutoff_steps 11 --tune_smooth_values none,1,10,100,1000 --tune_alternating True --score_after_train False --eval_every 5 --wandb False --wandb_project event-detection-pdf --wandb_log_artifacts True --tolerance_scale 1.0 --run_tag seizure_rescue_lr3e4_ds5_e20_bs32_eval5 --device cuda

# seizure unet density_gau seed=0
/home/ec2-user/EventDetectionPDF/.venv/bin/python train.py --dataset seizure --model unet --objective density_gau --datadir data/seizure --epochs 20 --folds 4 --bs 32 --downsample 5 --agg_feats stat --use_cat True --normalize True --workers 4 --seed 0 --density_prior sparse --best_metric mAP --save_all_epochs False --lr 0.0003 --weight_decay 0.0 --clip_grad_norm 0.1 --tune_cutoff_steps 11 --tune_smooth_values none,1,10,100,1000 --tune_alternating True --score_after_train False --eval_every 5 --wandb False --wandb_project event-detection-pdf --wandb_log_artifacts True --tolerance_scale 1.0 --run_tag seizure_rescue_lr3e4_ds5_e20_bs32_eval5 --device cuda

# seizure unet seg seed=0
/home/ec2-user/EventDetectionPDF/.venv/bin/python train.py --dataset seizure --model unet --objective seg --datadir data/seizure --epochs 20 --folds 4 --bs 32 --downsample 5 --agg_feats stat --use_cat True --normalize True --workers 4 --seed 0 --density_prior sparse --best_metric mAP --save_all_epochs False --lr 0.0003 --weight_decay 0.0 --clip_grad_norm 0.1 --tune_cutoff_steps 11 --tune_smooth_values none,1,10,100,1000 --tune_alternating True --score_after_train False --eval_every 5 --wandb False --wandb_project event-detection-pdf --wandb_log_artifacts True --tolerance_scale 1.0 --run_tag seizure_rescue_lr3e4_ds5_e20_bs32_eval5 --device cuda
