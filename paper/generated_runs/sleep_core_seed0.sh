#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# sleep gru density_hard seed=0
/tmp/eventpdf-venv/bin/python train.py --dataset sleep --model gru --objective density_hard --datadir data/sleep --epochs 20 --folds 4 --bs 10 --downsample 10 --agg_feats stat --use_cat True --normalize True --workers 4 --seed 0 --density_prior sparse --best_metric mAP --save_all_epochs False --tolerance_scale 1.0

# sleep gru density_gau seed=0
/tmp/eventpdf-venv/bin/python train.py --dataset sleep --model gru --objective density_gau --datadir data/sleep --epochs 20 --folds 4 --bs 10 --downsample 10 --agg_feats stat --use_cat True --normalize True --workers 4 --seed 0 --density_prior sparse --best_metric mAP --save_all_epochs False --tolerance_scale 1.0

# sleep gru density_custom seed=0
/tmp/eventpdf-venv/bin/python train.py --dataset sleep --model gru --objective density_custom --datadir data/sleep --epochs 20 --folds 4 --bs 10 --downsample 10 --agg_feats stat --use_cat True --normalize True --workers 4 --seed 0 --density_prior sparse --best_metric mAP --save_all_epochs False --tolerance_scale 1.0

# sleep gru hard seed=0
/tmp/eventpdf-venv/bin/python train.py --dataset sleep --model gru --objective hard --datadir data/sleep --epochs 20 --folds 4 --bs 10 --downsample 10 --agg_feats stat --use_cat True --normalize True --workers 4 --seed 0 --density_prior sparse --best_metric mAP --save_all_epochs False --tolerance_scale 1.0

# sleep gru gau seed=0
/tmp/eventpdf-venv/bin/python train.py --dataset sleep --model gru --objective gau --datadir data/sleep --epochs 20 --folds 4 --bs 10 --downsample 10 --agg_feats stat --use_cat True --normalize True --workers 4 --seed 0 --density_prior sparse --best_metric mAP --save_all_epochs False --tolerance_scale 1.0

# sleep gru custom seed=0
/tmp/eventpdf-venv/bin/python train.py --dataset sleep --model gru --objective custom --datadir data/sleep --epochs 20 --folds 4 --bs 10 --downsample 10 --agg_feats stat --use_cat True --normalize True --workers 4 --seed 0 --density_prior sparse --best_metric mAP --save_all_epochs False --tolerance_scale 1.0

# sleep gru seg seed=0
/tmp/eventpdf-venv/bin/python train.py --dataset sleep --model gru --objective seg --datadir data/sleep --epochs 20 --folds 4 --bs 10 --downsample 10 --agg_feats stat --use_cat True --normalize True --workers 4 --seed 0 --density_prior sparse --best_metric mAP --save_all_epochs False --tolerance_scale 1.0

# sleep gru seg_weighted seed=0
/tmp/eventpdf-venv/bin/python train.py --dataset sleep --model gru --objective seg_weighted --datadir data/sleep --epochs 20 --folds 4 --bs 10 --downsample 10 --agg_feats stat --use_cat True --normalize True --workers 4 --seed 0 --density_prior sparse --best_metric mAP --save_all_epochs False --tolerance_scale 1.0

# sleep gru seg_focal seed=0
/tmp/eventpdf-venv/bin/python train.py --dataset sleep --model gru --objective seg_focal --datadir data/sleep --epochs 20 --folds 4 --bs 10 --downsample 10 --agg_feats stat --use_cat True --normalize True --workers 4 --seed 0 --density_prior sparse --best_metric mAP --save_all_epochs False --tolerance_scale 1.0

# sleep unet density_hard seed=0
/tmp/eventpdf-venv/bin/python train.py --dataset sleep --model unet --objective density_hard --datadir data/sleep --epochs 20 --folds 4 --bs 10 --downsample 10 --agg_feats stat --use_cat True --normalize True --workers 4 --seed 0 --density_prior sparse --best_metric mAP --save_all_epochs False --tolerance_scale 1.0

# sleep unet density_gau seed=0
/tmp/eventpdf-venv/bin/python train.py --dataset sleep --model unet --objective density_gau --datadir data/sleep --epochs 20 --folds 4 --bs 10 --downsample 10 --agg_feats stat --use_cat True --normalize True --workers 4 --seed 0 --density_prior sparse --best_metric mAP --save_all_epochs False --tolerance_scale 1.0

# sleep unet density_custom seed=0
/tmp/eventpdf-venv/bin/python train.py --dataset sleep --model unet --objective density_custom --datadir data/sleep --epochs 20 --folds 4 --bs 10 --downsample 10 --agg_feats stat --use_cat True --normalize True --workers 4 --seed 0 --density_prior sparse --best_metric mAP --save_all_epochs False --tolerance_scale 1.0

# sleep unet hard seed=0
/tmp/eventpdf-venv/bin/python train.py --dataset sleep --model unet --objective hard --datadir data/sleep --epochs 20 --folds 4 --bs 10 --downsample 10 --agg_feats stat --use_cat True --normalize True --workers 4 --seed 0 --density_prior sparse --best_metric mAP --save_all_epochs False --tolerance_scale 1.0

# sleep unet gau seed=0
/tmp/eventpdf-venv/bin/python train.py --dataset sleep --model unet --objective gau --datadir data/sleep --epochs 20 --folds 4 --bs 10 --downsample 10 --agg_feats stat --use_cat True --normalize True --workers 4 --seed 0 --density_prior sparse --best_metric mAP --save_all_epochs False --tolerance_scale 1.0

# sleep unet custom seed=0
/tmp/eventpdf-venv/bin/python train.py --dataset sleep --model unet --objective custom --datadir data/sleep --epochs 20 --folds 4 --bs 10 --downsample 10 --agg_feats stat --use_cat True --normalize True --workers 4 --seed 0 --density_prior sparse --best_metric mAP --save_all_epochs False --tolerance_scale 1.0

# sleep unet seg seed=0
/tmp/eventpdf-venv/bin/python train.py --dataset sleep --model unet --objective seg --datadir data/sleep --epochs 20 --folds 4 --bs 10 --downsample 10 --agg_feats stat --use_cat True --normalize True --workers 4 --seed 0 --density_prior sparse --best_metric mAP --save_all_epochs False --tolerance_scale 1.0

# sleep unet seg_weighted seed=0
/tmp/eventpdf-venv/bin/python train.py --dataset sleep --model unet --objective seg_weighted --datadir data/sleep --epochs 20 --folds 4 --bs 10 --downsample 10 --agg_feats stat --use_cat True --normalize True --workers 4 --seed 0 --density_prior sparse --best_metric mAP --save_all_epochs False --tolerance_scale 1.0

# sleep unet seg_focal seed=0
/tmp/eventpdf-venv/bin/python train.py --dataset sleep --model unet --objective seg_focal --datadir data/sleep --epochs 20 --folds 4 --bs 10 --downsample 10 --agg_feats stat --use_cat True --normalize True --workers 4 --seed 0 --density_prior sparse --best_metric mAP --save_all_epochs False --tolerance_scale 1.0

# sleep unet_t density_hard seed=0
/tmp/eventpdf-venv/bin/python train.py --dataset sleep --model unet_t --objective density_hard --datadir data/sleep --epochs 20 --folds 4 --bs 10 --downsample 10 --agg_feats stat --use_cat True --normalize True --workers 4 --seed 0 --density_prior sparse --best_metric mAP --save_all_epochs False --tolerance_scale 1.0

# sleep unet_t density_gau seed=0
/tmp/eventpdf-venv/bin/python train.py --dataset sleep --model unet_t --objective density_gau --datadir data/sleep --epochs 20 --folds 4 --bs 10 --downsample 10 --agg_feats stat --use_cat True --normalize True --workers 4 --seed 0 --density_prior sparse --best_metric mAP --save_all_epochs False --tolerance_scale 1.0

# sleep unet_t density_custom seed=0
/tmp/eventpdf-venv/bin/python train.py --dataset sleep --model unet_t --objective density_custom --datadir data/sleep --epochs 20 --folds 4 --bs 10 --downsample 10 --agg_feats stat --use_cat True --normalize True --workers 4 --seed 0 --density_prior sparse --best_metric mAP --save_all_epochs False --tolerance_scale 1.0

# sleep unet_t hard seed=0
/tmp/eventpdf-venv/bin/python train.py --dataset sleep --model unet_t --objective hard --datadir data/sleep --epochs 20 --folds 4 --bs 10 --downsample 10 --agg_feats stat --use_cat True --normalize True --workers 4 --seed 0 --density_prior sparse --best_metric mAP --save_all_epochs False --tolerance_scale 1.0

# sleep unet_t gau seed=0
/tmp/eventpdf-venv/bin/python train.py --dataset sleep --model unet_t --objective gau --datadir data/sleep --epochs 20 --folds 4 --bs 10 --downsample 10 --agg_feats stat --use_cat True --normalize True --workers 4 --seed 0 --density_prior sparse --best_metric mAP --save_all_epochs False --tolerance_scale 1.0

# sleep unet_t custom seed=0
/tmp/eventpdf-venv/bin/python train.py --dataset sleep --model unet_t --objective custom --datadir data/sleep --epochs 20 --folds 4 --bs 10 --downsample 10 --agg_feats stat --use_cat True --normalize True --workers 4 --seed 0 --density_prior sparse --best_metric mAP --save_all_epochs False --tolerance_scale 1.0

# sleep unet_t seg seed=0
/tmp/eventpdf-venv/bin/python train.py --dataset sleep --model unet_t --objective seg --datadir data/sleep --epochs 20 --folds 4 --bs 10 --downsample 10 --agg_feats stat --use_cat True --normalize True --workers 4 --seed 0 --density_prior sparse --best_metric mAP --save_all_epochs False --tolerance_scale 1.0

# sleep unet_t seg_weighted seed=0
/tmp/eventpdf-venv/bin/python train.py --dataset sleep --model unet_t --objective seg_weighted --datadir data/sleep --epochs 20 --folds 4 --bs 10 --downsample 10 --agg_feats stat --use_cat True --normalize True --workers 4 --seed 0 --density_prior sparse --best_metric mAP --save_all_epochs False --tolerance_scale 1.0

# sleep unet_t seg_focal seed=0
/tmp/eventpdf-venv/bin/python train.py --dataset sleep --model unet_t --objective seg_focal --datadir data/sleep --epochs 20 --folds 4 --bs 10 --downsample 10 --agg_feats stat --use_cat True --normalize True --workers 4 --seed 0 --density_prior sparse --best_metric mAP --save_all_epochs False --tolerance_scale 1.0

# sleep prectime density_hard seed=0
/tmp/eventpdf-venv/bin/python train.py --dataset sleep --model prectime --objective density_hard --datadir data/sleep --epochs 20 --folds 4 --bs 10 --downsample 10 --agg_feats stat --use_cat True --normalize True --workers 4 --seed 0 --density_prior sparse --best_metric mAP --save_all_epochs False --tolerance_scale 1.0

# sleep prectime density_gau seed=0
/tmp/eventpdf-venv/bin/python train.py --dataset sleep --model prectime --objective density_gau --datadir data/sleep --epochs 20 --folds 4 --bs 10 --downsample 10 --agg_feats stat --use_cat True --normalize True --workers 4 --seed 0 --density_prior sparse --best_metric mAP --save_all_epochs False --tolerance_scale 1.0

# sleep prectime density_custom seed=0
/tmp/eventpdf-venv/bin/python train.py --dataset sleep --model prectime --objective density_custom --datadir data/sleep --epochs 20 --folds 4 --bs 10 --downsample 10 --agg_feats stat --use_cat True --normalize True --workers 4 --seed 0 --density_prior sparse --best_metric mAP --save_all_epochs False --tolerance_scale 1.0

# sleep prectime hard seed=0
/tmp/eventpdf-venv/bin/python train.py --dataset sleep --model prectime --objective hard --datadir data/sleep --epochs 20 --folds 4 --bs 10 --downsample 10 --agg_feats stat --use_cat True --normalize True --workers 4 --seed 0 --density_prior sparse --best_metric mAP --save_all_epochs False --tolerance_scale 1.0

# sleep prectime gau seed=0
/tmp/eventpdf-venv/bin/python train.py --dataset sleep --model prectime --objective gau --datadir data/sleep --epochs 20 --folds 4 --bs 10 --downsample 10 --agg_feats stat --use_cat True --normalize True --workers 4 --seed 0 --density_prior sparse --best_metric mAP --save_all_epochs False --tolerance_scale 1.0

# sleep prectime custom seed=0
/tmp/eventpdf-venv/bin/python train.py --dataset sleep --model prectime --objective custom --datadir data/sleep --epochs 20 --folds 4 --bs 10 --downsample 10 --agg_feats stat --use_cat True --normalize True --workers 4 --seed 0 --density_prior sparse --best_metric mAP --save_all_epochs False --tolerance_scale 1.0

# sleep prectime seg seed=0
/tmp/eventpdf-venv/bin/python train.py --dataset sleep --model prectime --objective seg --datadir data/sleep --epochs 20 --folds 4 --bs 10 --downsample 10 --agg_feats stat --use_cat True --normalize True --workers 4 --seed 0 --density_prior sparse --best_metric mAP --save_all_epochs False --tolerance_scale 1.0

# sleep prectime seg_weighted seed=0
/tmp/eventpdf-venv/bin/python train.py --dataset sleep --model prectime --objective seg_weighted --datadir data/sleep --epochs 20 --folds 4 --bs 10 --downsample 10 --agg_feats stat --use_cat True --normalize True --workers 4 --seed 0 --density_prior sparse --best_metric mAP --save_all_epochs False --tolerance_scale 1.0

# sleep prectime seg_focal seed=0
/tmp/eventpdf-venv/bin/python train.py --dataset sleep --model prectime --objective seg_focal --datadir data/sleep --epochs 20 --folds 4 --bs 10 --downsample 10 --agg_feats stat --use_cat True --normalize True --workers 4 --seed 0 --density_prior sparse --best_metric mAP --save_all_epochs False --tolerance_scale 1.0
