#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# sleep gru density_custom seed=0
/tmp/eventpdf-venv/bin/python train.py --dataset sleep --model gru --objective density_custom --datadir data --epochs 1 --folds 2 --bs 10 --downsample 10 --agg_feats stat --use_cat True --normalize True --workers 4 --seed 0 --density_prior sparse --best_metric mAP --save_all_epochs False --tolerance_scale 1.0

# sleep gru custom seed=0
/tmp/eventpdf-venv/bin/python train.py --dataset sleep --model gru --objective custom --datadir data --epochs 1 --folds 2 --bs 10 --downsample 10 --agg_feats stat --use_cat True --normalize True --workers 4 --seed 0 --density_prior sparse --best_metric mAP --save_all_epochs False --tolerance_scale 1.0

# sleep gru seg seed=0
/tmp/eventpdf-venv/bin/python train.py --dataset sleep --model gru --objective seg --datadir data --epochs 1 --folds 2 --bs 10 --downsample 10 --agg_feats stat --use_cat True --normalize True --workers 4 --seed 0 --density_prior sparse --best_metric mAP --save_all_epochs False --tolerance_scale 1.0
