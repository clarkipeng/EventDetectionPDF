#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# seizure gru density_hard seed=0
/tmp/eventpdf-venv/bin/python train.py --dataset seizure --model gru --objective density_hard --datadir data/seizure --epochs 20 --folds 4 --bs 10 --downsample 10 --agg_feats stat --use_cat True --normalize True --workers 4 --seed 0 --density_prior sparse --best_metric mAP --save_all_epochs False --tolerance_scale 1.0

# seizure gru density_gau seed=0
/tmp/eventpdf-venv/bin/python train.py --dataset seizure --model gru --objective density_gau --datadir data/seizure --epochs 20 --folds 4 --bs 10 --downsample 10 --agg_feats stat --use_cat True --normalize True --workers 4 --seed 0 --density_prior sparse --best_metric mAP --save_all_epochs False --tolerance_scale 1.0

# seizure gru density_custom seed=0
/tmp/eventpdf-venv/bin/python train.py --dataset seizure --model gru --objective density_custom --datadir data/seizure --epochs 20 --folds 4 --bs 10 --downsample 10 --agg_feats stat --use_cat True --normalize True --workers 4 --seed 0 --density_prior sparse --best_metric mAP --save_all_epochs False --tolerance_scale 1.0

# seizure gru hard seed=0
/tmp/eventpdf-venv/bin/python train.py --dataset seizure --model gru --objective hard --datadir data/seizure --epochs 20 --folds 4 --bs 10 --downsample 10 --agg_feats stat --use_cat True --normalize True --workers 4 --seed 0 --density_prior sparse --best_metric mAP --save_all_epochs False --tolerance_scale 1.0

# seizure gru gau seed=0
/tmp/eventpdf-venv/bin/python train.py --dataset seizure --model gru --objective gau --datadir data/seizure --epochs 20 --folds 4 --bs 10 --downsample 10 --agg_feats stat --use_cat True --normalize True --workers 4 --seed 0 --density_prior sparse --best_metric mAP --save_all_epochs False --tolerance_scale 1.0

# seizure gru custom seed=0
/tmp/eventpdf-venv/bin/python train.py --dataset seizure --model gru --objective custom --datadir data/seizure --epochs 20 --folds 4 --bs 10 --downsample 10 --agg_feats stat --use_cat True --normalize True --workers 4 --seed 0 --density_prior sparse --best_metric mAP --save_all_epochs False --tolerance_scale 1.0

# seizure gru seg seed=0
/tmp/eventpdf-venv/bin/python train.py --dataset seizure --model gru --objective seg --datadir data/seizure --epochs 20 --folds 4 --bs 10 --downsample 10 --agg_feats stat --use_cat True --normalize True --workers 4 --seed 0 --density_prior sparse --best_metric mAP --save_all_epochs False --tolerance_scale 1.0

# seizure gru seg_weighted seed=0
/tmp/eventpdf-venv/bin/python train.py --dataset seizure --model gru --objective seg_weighted --datadir data/seizure --epochs 20 --folds 4 --bs 10 --downsample 10 --agg_feats stat --use_cat True --normalize True --workers 4 --seed 0 --density_prior sparse --best_metric mAP --save_all_epochs False --tolerance_scale 1.0

# seizure gru seg_focal seed=0
/tmp/eventpdf-venv/bin/python train.py --dataset seizure --model gru --objective seg_focal --datadir data/seizure --epochs 20 --folds 4 --bs 10 --downsample 10 --agg_feats stat --use_cat True --normalize True --workers 4 --seed 0 --density_prior sparse --best_metric mAP --save_all_epochs False --tolerance_scale 1.0

# seizure unet density_hard seed=0
/tmp/eventpdf-venv/bin/python train.py --dataset seizure --model unet --objective density_hard --datadir data/seizure --epochs 20 --folds 4 --bs 10 --downsample 10 --agg_feats stat --use_cat True --normalize True --workers 4 --seed 0 --density_prior sparse --best_metric mAP --save_all_epochs False --tolerance_scale 1.0

# seizure unet density_gau seed=0
/tmp/eventpdf-venv/bin/python train.py --dataset seizure --model unet --objective density_gau --datadir data/seizure --epochs 20 --folds 4 --bs 10 --downsample 10 --agg_feats stat --use_cat True --normalize True --workers 4 --seed 0 --density_prior sparse --best_metric mAP --save_all_epochs False --tolerance_scale 1.0

# seizure unet density_custom seed=0
/tmp/eventpdf-venv/bin/python train.py --dataset seizure --model unet --objective density_custom --datadir data/seizure --epochs 20 --folds 4 --bs 10 --downsample 10 --agg_feats stat --use_cat True --normalize True --workers 4 --seed 0 --density_prior sparse --best_metric mAP --save_all_epochs False --tolerance_scale 1.0

# seizure unet hard seed=0
/tmp/eventpdf-venv/bin/python train.py --dataset seizure --model unet --objective hard --datadir data/seizure --epochs 20 --folds 4 --bs 10 --downsample 10 --agg_feats stat --use_cat True --normalize True --workers 4 --seed 0 --density_prior sparse --best_metric mAP --save_all_epochs False --tolerance_scale 1.0

# seizure unet gau seed=0
/tmp/eventpdf-venv/bin/python train.py --dataset seizure --model unet --objective gau --datadir data/seizure --epochs 20 --folds 4 --bs 10 --downsample 10 --agg_feats stat --use_cat True --normalize True --workers 4 --seed 0 --density_prior sparse --best_metric mAP --save_all_epochs False --tolerance_scale 1.0

# seizure unet custom seed=0
/tmp/eventpdf-venv/bin/python train.py --dataset seizure --model unet --objective custom --datadir data/seizure --epochs 20 --folds 4 --bs 10 --downsample 10 --agg_feats stat --use_cat True --normalize True --workers 4 --seed 0 --density_prior sparse --best_metric mAP --save_all_epochs False --tolerance_scale 1.0

# seizure unet seg seed=0
/tmp/eventpdf-venv/bin/python train.py --dataset seizure --model unet --objective seg --datadir data/seizure --epochs 20 --folds 4 --bs 10 --downsample 10 --agg_feats stat --use_cat True --normalize True --workers 4 --seed 0 --density_prior sparse --best_metric mAP --save_all_epochs False --tolerance_scale 1.0

# seizure unet seg_weighted seed=0
/tmp/eventpdf-venv/bin/python train.py --dataset seizure --model unet --objective seg_weighted --datadir data/seizure --epochs 20 --folds 4 --bs 10 --downsample 10 --agg_feats stat --use_cat True --normalize True --workers 4 --seed 0 --density_prior sparse --best_metric mAP --save_all_epochs False --tolerance_scale 1.0

# seizure unet seg_focal seed=0
/tmp/eventpdf-venv/bin/python train.py --dataset seizure --model unet --objective seg_focal --datadir data/seizure --epochs 20 --folds 4 --bs 10 --downsample 10 --agg_feats stat --use_cat True --normalize True --workers 4 --seed 0 --density_prior sparse --best_metric mAP --save_all_epochs False --tolerance_scale 1.0

# seizure unet_t density_hard seed=0
/tmp/eventpdf-venv/bin/python train.py --dataset seizure --model unet_t --objective density_hard --datadir data/seizure --epochs 20 --folds 4 --bs 10 --downsample 10 --agg_feats stat --use_cat True --normalize True --workers 4 --seed 0 --density_prior sparse --best_metric mAP --save_all_epochs False --tolerance_scale 1.0

# seizure unet_t density_gau seed=0
/tmp/eventpdf-venv/bin/python train.py --dataset seizure --model unet_t --objective density_gau --datadir data/seizure --epochs 20 --folds 4 --bs 10 --downsample 10 --agg_feats stat --use_cat True --normalize True --workers 4 --seed 0 --density_prior sparse --best_metric mAP --save_all_epochs False --tolerance_scale 1.0

# seizure unet_t density_custom seed=0
/tmp/eventpdf-venv/bin/python train.py --dataset seizure --model unet_t --objective density_custom --datadir data/seizure --epochs 20 --folds 4 --bs 10 --downsample 10 --agg_feats stat --use_cat True --normalize True --workers 4 --seed 0 --density_prior sparse --best_metric mAP --save_all_epochs False --tolerance_scale 1.0

# seizure unet_t hard seed=0
/tmp/eventpdf-venv/bin/python train.py --dataset seizure --model unet_t --objective hard --datadir data/seizure --epochs 20 --folds 4 --bs 10 --downsample 10 --agg_feats stat --use_cat True --normalize True --workers 4 --seed 0 --density_prior sparse --best_metric mAP --save_all_epochs False --tolerance_scale 1.0

# seizure unet_t gau seed=0
/tmp/eventpdf-venv/bin/python train.py --dataset seizure --model unet_t --objective gau --datadir data/seizure --epochs 20 --folds 4 --bs 10 --downsample 10 --agg_feats stat --use_cat True --normalize True --workers 4 --seed 0 --density_prior sparse --best_metric mAP --save_all_epochs False --tolerance_scale 1.0

# seizure unet_t custom seed=0
/tmp/eventpdf-venv/bin/python train.py --dataset seizure --model unet_t --objective custom --datadir data/seizure --epochs 20 --folds 4 --bs 10 --downsample 10 --agg_feats stat --use_cat True --normalize True --workers 4 --seed 0 --density_prior sparse --best_metric mAP --save_all_epochs False --tolerance_scale 1.0

# seizure unet_t seg seed=0
/tmp/eventpdf-venv/bin/python train.py --dataset seizure --model unet_t --objective seg --datadir data/seizure --epochs 20 --folds 4 --bs 10 --downsample 10 --agg_feats stat --use_cat True --normalize True --workers 4 --seed 0 --density_prior sparse --best_metric mAP --save_all_epochs False --tolerance_scale 1.0

# seizure unet_t seg_weighted seed=0
/tmp/eventpdf-venv/bin/python train.py --dataset seizure --model unet_t --objective seg_weighted --datadir data/seizure --epochs 20 --folds 4 --bs 10 --downsample 10 --agg_feats stat --use_cat True --normalize True --workers 4 --seed 0 --density_prior sparse --best_metric mAP --save_all_epochs False --tolerance_scale 1.0

# seizure unet_t seg_focal seed=0
/tmp/eventpdf-venv/bin/python train.py --dataset seizure --model unet_t --objective seg_focal --datadir data/seizure --epochs 20 --folds 4 --bs 10 --downsample 10 --agg_feats stat --use_cat True --normalize True --workers 4 --seed 0 --density_prior sparse --best_metric mAP --save_all_epochs False --tolerance_scale 1.0

# seizure prectime density_hard seed=0
/tmp/eventpdf-venv/bin/python train.py --dataset seizure --model prectime --objective density_hard --datadir data/seizure --epochs 20 --folds 4 --bs 10 --downsample 10 --agg_feats stat --use_cat True --normalize True --workers 4 --seed 0 --density_prior sparse --best_metric mAP --save_all_epochs False --tolerance_scale 1.0

# seizure prectime density_gau seed=0
/tmp/eventpdf-venv/bin/python train.py --dataset seizure --model prectime --objective density_gau --datadir data/seizure --epochs 20 --folds 4 --bs 10 --downsample 10 --agg_feats stat --use_cat True --normalize True --workers 4 --seed 0 --density_prior sparse --best_metric mAP --save_all_epochs False --tolerance_scale 1.0

# seizure prectime density_custom seed=0
/tmp/eventpdf-venv/bin/python train.py --dataset seizure --model prectime --objective density_custom --datadir data/seizure --epochs 20 --folds 4 --bs 10 --downsample 10 --agg_feats stat --use_cat True --normalize True --workers 4 --seed 0 --density_prior sparse --best_metric mAP --save_all_epochs False --tolerance_scale 1.0

# seizure prectime hard seed=0
/tmp/eventpdf-venv/bin/python train.py --dataset seizure --model prectime --objective hard --datadir data/seizure --epochs 20 --folds 4 --bs 10 --downsample 10 --agg_feats stat --use_cat True --normalize True --workers 4 --seed 0 --density_prior sparse --best_metric mAP --save_all_epochs False --tolerance_scale 1.0

# seizure prectime gau seed=0
/tmp/eventpdf-venv/bin/python train.py --dataset seizure --model prectime --objective gau --datadir data/seizure --epochs 20 --folds 4 --bs 10 --downsample 10 --agg_feats stat --use_cat True --normalize True --workers 4 --seed 0 --density_prior sparse --best_metric mAP --save_all_epochs False --tolerance_scale 1.0

# seizure prectime custom seed=0
/tmp/eventpdf-venv/bin/python train.py --dataset seizure --model prectime --objective custom --datadir data/seizure --epochs 20 --folds 4 --bs 10 --downsample 10 --agg_feats stat --use_cat True --normalize True --workers 4 --seed 0 --density_prior sparse --best_metric mAP --save_all_epochs False --tolerance_scale 1.0

# seizure prectime seg seed=0
/tmp/eventpdf-venv/bin/python train.py --dataset seizure --model prectime --objective seg --datadir data/seizure --epochs 20 --folds 4 --bs 10 --downsample 10 --agg_feats stat --use_cat True --normalize True --workers 4 --seed 0 --density_prior sparse --best_metric mAP --save_all_epochs False --tolerance_scale 1.0

# seizure prectime seg_weighted seed=0
/tmp/eventpdf-venv/bin/python train.py --dataset seizure --model prectime --objective seg_weighted --datadir data/seizure --epochs 20 --folds 4 --bs 10 --downsample 10 --agg_feats stat --use_cat True --normalize True --workers 4 --seed 0 --density_prior sparse --best_metric mAP --save_all_epochs False --tolerance_scale 1.0

# seizure prectime seg_focal seed=0
/tmp/eventpdf-venv/bin/python train.py --dataset seizure --model prectime --objective seg_focal --datadir data/seizure --epochs 20 --folds 4 --bs 10 --downsample 10 --agg_feats stat --use_cat True --normalize True --workers 4 --seed 0 --density_prior sparse --best_metric mAP --save_all_epochs False --tolerance_scale 1.0
