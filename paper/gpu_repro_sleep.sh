#!/usr/bin/env bash
# Remote GPU reproduction for the sleep prediction figure.
# Catapult builds a .venv from requirements.txt; we add kaggle, download the
# data, train the two objectives the figure needs on CUDA, and declare the
# resulting per-series predictions as artifacts.
set -euo pipefail

PY=.venv/bin/python
DATADIR=data/sleep
RUN_TAG=objective_e20_bs32_eval5

# Install everything (incl. torch) here in the command phase. The setup phase
# has a 180s no-heartbeat watchdog; the command phase heartbeats every 30s, so a
# full install (torch included) is safe here. Installing torch fresh into the
# venv avoids depending on the image's interpreter being inherited.
echo "== install python deps (incl. torch) =="
uv pip install --python "$PY" \
  torch==2.2.2 torchvision==0.17.2 \
  matplotlib==3.8.3 numba==0.59.1 numpy==1.25.2 pandas==2.0.3 Pillow==11.3.0 \
  polars==0.20.23 pyarrow==15.0.2 scikit_learn==1.4.1.post1 scipy==1.13.0 \
  timm==0.9.16 tqdm==4.66.2 wandb==0.26.1 kaggle
echo "== cuda check =="
$PY -c "import torch, timm, polars, pyarrow, sklearn, scipy, numba, pandas, kaggle; from timm.scheduler import CosineLRScheduler; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

echo "== download sleep data (public mirror; no competition rules needed) =="
mkdir -p "$DATADIR"
$PY - <<PYDL
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
api.dataset_download_files("error404ntfound404/child-mind-institute-detect-sleep-states",
                           path="$DATADIR", unzip=True, quiet=False)
print("download complete")
PYDL
ls -lh "$DATADIR"

for spec in "density_hard 3e-3" "seg 1e-3"; do
  set -- $spec
  OBJ=$1; LR=$2
  echo "== train sleep gru $OBJ (lr $LR) =="
  $PY train.py --dataset sleep --model gru --objective "$OBJ" \
    --datadir "$DATADIR" --run_tag "$RUN_TAG" \
    --epochs 20 --folds 2 --bs 32 --lr "$LR" \
    --device cuda --workers 4 --eval_every 20 \
    --score_after_train False --seed 0
done

echo "== declare artifacts =="
mkdir -p .catapult
cat > .catapult/artifacts.json <<JSON
{
  "artifacts": [
    {"path": "experiments/sleep/gru/density_hard/seed_0_${RUN_TAG}/predictions",
     "kind": "predictions", "reason": "BDL (density_hard) per-series predictions"},
    {"path": "experiments/sleep/gru/seg/seed_0_${RUN_TAG}/predictions",
     "kind": "predictions", "reason": "segmentation per-series predictions"},
    {"path": "experiments/sleep/gru/density_hard/seed_0_${RUN_TAG}/results/fold_results.csv",
     "kind": "metrics", "reason": "density fold metrics"},
    {"path": "experiments/sleep/gru/seg/seed_0_${RUN_TAG}/results/fold_results.csv",
     "kind": "metrics", "reason": "seg fold metrics"}
  ]
}
JSON

echo "== target series predictions present? =="
ls -l experiments/sleep/gru/density_hard/seed_0_${RUN_TAG}/predictions/0402a003dae9.npy
ls -l experiments/sleep/gru/seg/seed_0_${RUN_TAG}/predictions/0402a003dae9.npy
echo "DONE"
