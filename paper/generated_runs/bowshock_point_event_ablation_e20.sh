#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# bowshock gru density_hard seed=0
uv run --offline python train.py --dataset bowshock --model gru --objective density_hard --datadir data --epochs 20 --folds 4 --bs 32 --downsample 10 --agg_feats stat --use_cat True --normalize True --workers 2 --seed 0 --density_prior sparse --best_metric mAP --save_all_epochs False --lr 0.003 --weight_decay 0.0 --clip_grad_norm 0.1 --tune_cutoff_steps 5 --tune_smooth_values none,1,10 --tune_alternating True --score_after_train True --eval_every 5 --wandb True --wandb_project event-detection-pdf --wandb_log_artifacts True --tolerance_scale 1.0 --wandb_group P13-point-event-bowshock --wandb_tags bdl-paper,bowshock,P13,point-event --run_tag point_event_ablation_e20 --device cuda

# bowshock gru density_gau seed=0
uv run --offline python train.py --dataset bowshock --model gru --objective density_gau --datadir data --epochs 20 --folds 4 --bs 32 --downsample 10 --agg_feats stat --use_cat True --normalize True --workers 2 --seed 0 --density_prior sparse --best_metric mAP --save_all_epochs False --lr 0.003 --weight_decay 0.0 --clip_grad_norm 0.1 --tune_cutoff_steps 5 --tune_smooth_values none,1,10 --tune_alternating True --score_after_train True --eval_every 5 --wandb True --wandb_project event-detection-pdf --wandb_log_artifacts True --tolerance_scale 1.0 --wandb_group P13-point-event-bowshock --wandb_tags bdl-paper,bowshock,P13,point-event --run_tag point_event_ablation_e20 --device cuda

# bowshock gru density_custom seed=0
uv run --offline python train.py --dataset bowshock --model gru --objective density_custom --datadir data --epochs 20 --folds 4 --bs 32 --downsample 10 --agg_feats stat --use_cat True --normalize True --workers 2 --seed 0 --density_prior sparse --best_metric mAP --save_all_epochs False --lr 0.003 --weight_decay 0.0 --clip_grad_norm 0.1 --tune_cutoff_steps 5 --tune_smooth_values none,1,10 --tune_alternating True --score_after_train True --eval_every 5 --wandb True --wandb_project event-detection-pdf --wandb_log_artifacts True --tolerance_scale 1.0 --wandb_group P13-point-event-bowshock --wandb_tags bdl-paper,bowshock,P13,point-event --run_tag point_event_ablation_e20 --device cuda

# bowshock gru seg seed=0
uv run --offline python train.py --dataset bowshock --model gru --objective seg --datadir data --epochs 20 --folds 4 --bs 32 --downsample 10 --agg_feats stat --use_cat True --normalize True --workers 2 --seed 0 --density_prior sparse --best_metric mAP --save_all_epochs False --lr 0.003 --weight_decay 0.0 --clip_grad_norm 0.1 --tune_cutoff_steps 5 --tune_smooth_values none,1,10 --tune_alternating True --score_after_train True --eval_every 5 --wandb True --wandb_project event-detection-pdf --wandb_log_artifacts True --tolerance_scale 1.0 --wandb_group P13-point-event-bowshock --wandb_tags bdl-paper,bowshock,P13,point-event --run_tag point_event_ablation_e20 --device cuda
