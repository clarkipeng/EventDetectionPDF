# Event Detection via Probability Density Function Regression

This repository may be used to train all the the models used for experiments in the [paper](https://arxiv.org/abs/2408.12792)

## Contents

- [Overview](#overview)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [Data Preparation](#data-preparation)
- [Training and Evaluation](#training-and-evaluation)
- [References](#references)


## Overview
This document describes the official software package developed for and used to create the general regression-based approach for sleep CPD. It features different models, like [PrecTime](https://arxiv.org/ftp/arxiv/papers/2302/2302.10182.pdf), 1D UNets, and Bidirectional RNNS.

This software allows the training of binary sleep CPD models using Child Mind Institute's [sleep detection dataset](https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states/data) and seizure event detection models using Physionet's [CHB-MIT Scalp EEG Database](https://archive.physionet.org/physiobank/database/chbmit/) from <cite>[Shoeb, Ali, 2009][2]</cite>. It features a command-line interface for training and evaluating models without needing to modify the underlying codebase.

## System Requirements
**Hardware Requirements**

For training our models from scratch,  we recommend using a Linux based computer with at least the following hardware specifications:

* 4+ CPU cores
* 26+ GiB RAM
* 5+ GiB physical storage space*
* 1 CUDA enabled GPU (please refer to [https://developer.nvidia.com/cuda-gpus](https://developer.nvidia.com/cuda-gpus) for a detailed list).

It is possible to train the model on smaller machines, and without GPUs, but doing so may take considerable time (1-2 min vs 8 sec per epoch). Likewise, more resources will speed up training. The data is automatically preprocessed in the script. However, if the preprocessing step exceeds the system memory, data should be preprocessed with the [```process_sleep_dataset```](https://github.com/clarkipeng/SleepRegressionCPD/blob/main/src/load_dataset.py#L214) function on virtual machine with more system memory, e.g., Kaggle's kernels or Google Colab's notebooks.

*The required hard-disk space depends on number of models/objectives used. The predictions for each model are cached, each new model/objective combination takes ~100Mb more disk memory each.

**Software Requirements:**

If you are going to run these scripts yourself from scratch, we highly recommend doing so on a GPU. In order to run the scripts with a GPU, the `pytorch` (`v2.2.2`) library is used. For this, the following additional software is required on your system:

* [NVIDIA GPU drivers](https://www.nvidia.com/drivers)
* [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive)
* [cuDNN SDK](https://developer.nvidia.com/cudnn)

Please refer to [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) for additional details.

## Installation Guide
On a computer with `pip` installed, run the following commands to download the required packages.

```
git clone https://github.com/clark/.git
pip install -r regressioneventdetection/requirements.txt
```

## Data Preparation:

Download the [sleep detection dataset](https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states/data) or [seizure detection dataset](https://www.kaggle.com/datasets/werus23/chb-mit-scalp-eeg-database-seizure-only) from Kaggle.com. Place the downloaded dataset in a new directory called `data`. We require the directory structure to include the following:
```
path/to/repo/data
  train_series.parquet
  train_events.csv
```
or
```
path/to/repo/data
  seizure_256Hz_dataset
  seizure_events.csv
```

This repository also provides support for other datasets, such as the [bowshock detection dataset](https://archive.org/download/martian_bow_shock_dataset/martian_bow_shock_dataset.pkl) and [fraud detection dataset](https://archive.org/download/credit_card_fraud_dataset/credit_card_fraud_dataset.csv) as well as their [bowshock event labels](https://archive.org/download/martian_bow_shock_events/martian_bow_shock_events.csv) and [fraud event labels](https://archive.org/download/credit_card_fraud_events/credit_card_fraud_events.csv) provided by <cite>[Azib et al, 2023][1]</cite>. Place the downloaded datasets in the `data` directory. The required directory structure is:
```
path/to/repo/data
  martian_bow_shock_dataset.pkl
  martian_bow_shock_events.csv
  credit_card_fraud_dataset.csv
  credit_card_fraud_events.csv
```

For the use of external datasets, a ```DataClass``` enum and a ```torch.utils.data.Dataset``` constructor must be defined. ```DataClass``` lays out the parameters and types of the dataset, such as the number of features, whether events are point-based or time-interval-based, and the thresholds to use in evaluation. The ```Dataset``` constructor must construct the inputs to the model, such as the input timeseries and target series. 

More information can be found with the dataloader scripts found in [src](src).

## Training and Evaluation
We have 3 different scripts to aid in training and evaluation. These scripts are [train_all.py](train_all.py), [train.py](train.py), and [eval.py](eval.py).
After you have done all the necessary steps listed above, you are ready to train and evaluate the models. In order to train a model on a certain objective, you can simply run the following script with the names of the models and objectives: 

```
python train.py --dataset [dataset_name] --model [model_name] --objective [objective_name] --datadir [path_to_dataset]
```
Model choices vary: *rnn* (or *lstm* and *gru*), *unet* (or *unet_t*), and *prectime*. More information about model choices can be found at [load_model.py](models/load_model.py). Objectives can be *hard*, *gau*, *custom*, *seg1*, *seg2*, or *seg* (a combination of both segmentation methods).

In order to evaluate the trained models, run: 
```
python eval.py --dataset [dataset_name] --datadir [path_to_dataset]
```

In order to train the 5 main models (*seg*, *gru*, *unet*, *unet_t* and *prectime*) on all objectives, run: 
```
python train_all.py --dataset [dataset_name] --datadir [path_to_dataset]
```

#### Example
Here is a example of running the training script on a machine with an A100 GPU and 32 GB of RAM:
```
python train.py --dataset sleep --model gru --objective seg --epochs 10 --folds 4
```
Which has the following output:
```
fold 0, epoch 1/10: train loss: 6.567, valid loss: 6.930, valid mAP: 0.023
fold 0, epoch 2/10: train loss: 6.780, valid loss: 6.907, valid mAP: 0.109
fold 0, epoch 3/10: train loss: 6.679, valid loss: 6.875, valid mAP: 0.237
fold 0, epoch 4/10: train loss: 6.765, valid loss: 6.803, valid mAP: 0.416
fold 0, epoch 5/10: train loss: 6.409, valid loss: 6.758, valid mAP: 0.519
fold 0, epoch 6/10: train loss: 6.453, valid loss: 6.722, valid mAP: 0.574
fold 0, epoch 7/10: train loss: 6.510, valid loss: 6.703, valid mAP: 0.602
fold 0, epoch 8/10: train loss: 6.657, valid loss: 6.692, valid mAP: 0.620
fold 0, epoch 9/10: train loss: 6.529, valid loss: 6.688, valid mAP: 0.614
fold 0, epoch 10/10: train loss: 6.454, valid loss: 6.687, valid mAP: 0.610
fold 1, epoch 1/10: train loss: 6.878, valid loss: 7.165, valid mAP: 0.026
fold 1, epoch 2/10: train loss: 6.660, valid loss: 7.139, valid mAP: 0.081
fold 1, epoch 3/10: train loss: 6.648, valid loss: 7.100, valid mAP: 0.208
fold 1, epoch 4/10: train loss: 6.560, valid loss: 7.026, valid mAP: 0.431
fold 1, epoch 5/10: train loss: 6.436, valid loss: 6.962, valid mAP: 0.553
fold 1, epoch 6/10: train loss: 6.188, valid loss: 6.921, valid mAP: 0.592
fold 1, epoch 7/10: train loss: 6.218, valid loss: 6.896, valid mAP: 0.608
fold 1, epoch 8/10: train loss: 6.160, valid loss: 6.884, valid mAP: 0.602
fold 1, epoch 9/10: train loss: 6.352, valid loss: 6.881, valid mAP: 0.611
fold 1, epoch 10/10: train loss: 6.206, valid loss: 6.880, valid mAP: 0.608
fold 2, epoch 1/10: train loss: 6.805, valid loss: 6.674, valid mAP: 0.0163
fold 2, epoch 2/10: train loss: 6.774, valid loss: 6.653, valid mAP: 0.103
fold 2, epoch 3/10: train loss: 6.763, valid loss: 6.635, valid mAP: 0.213
fold 2, epoch 4/10: train loss: 6.904, valid loss: 6.593, valid mAP: 0.429
fold 2, epoch 5/10: train loss: 6.684, valid loss: 6.515, valid mAP: 0.538
fold 2, epoch 6/10: train loss: 6.731, valid loss: 6.483, valid mAP: 0.589
fold 2, epoch 7/10: train loss: 6.780, valid loss: 6.463, valid mAP: 0.600
fold 2, epoch 8/10: train loss: 6.468, valid loss: 6.451, valid mAP: 0.612
fold 2, epoch 9/10: train loss: 6.542, valid loss: 6.444, valid mAP: 0.625
fold 2, epoch 10/10: train loss: 6.504, valid loss: 6.443, valid mAP: 0.628
fold 3, epoch 1/10: train loss: 7.531, valid loss: 6.099, valid mAP: 0.012
fold 3, epoch 2/10: train loss: 6.723, valid loss: 6.081, valid mAP: 0.038
fold 3, epoch 3/10: train loss: 6.891, valid loss: 6.068, valid mAP: 0.119
fold 3, epoch 4/10: train loss: 7.002, valid loss: 6.023, valid mAP: 0.305
fold 3, epoch 5/10: train loss: 7.033, valid loss: 5.962, valid mAP: 0.474
fold 3, epoch 6/10: train loss: 6.730, valid loss: 5.928, valid mAP: 0.552
fold 3, epoch 7/10: train loss: 6.812, valid loss: 5.905, valid mAP: 0.589
fold 3, epoch 8/10: train loss: 6.796, valid loss: 5.895, valid mAP: 0.603
fold 3, epoch 9/10: train loss: 6.8624, valid loss: 5.890, valid mAP: 0.599
fold 3, epoch 10/10: train loss: 6.608, valid loss: 5.890, valid mAP: 0.594
gru hard results: 
 default scores: mAP = 0.565, maxf1 = 0.663, 
 optimizing hyperparams for mAP:
  best params: cutoff = 0.0, smoothing = 40
  best scores: mAP = 0.620, maxf1 = 0.681, 
   tolerance 12 : mAP = 0.037, maxf1 = 0.186, 
   tolerance 36 : mAP = 0.303, maxf1 = 0.518, 
   tolerance 60 : mAP = 0.529, maxf1 = 0.664, 
   tolerance 90 : mAP = 0.662, maxf1 = 0.731, 
   tolerance 120 : mAP = 0.711, maxf1 = 0.755, 
   tolerance 150 : mAP = 0.740, maxf1 = 0.771, 
   tolerance 180 : mAP = 0.759, maxf1 = 0.780, 
   tolerance 240 : mAP = 0.784, maxf1 = 0.792, 
   tolerance 300 : mAP = 0.805, maxf1 = 0.802, 
   tolerance 360 : mAP = 0.817, maxf1 = 0.808,
 optimizing hyperparams for maxf1:
  best params: cutoff = 0.0, smoothing = 40
  best scores: mAP = 0.620, maxf1 = 0.681, 
   tolerance 12 : mAP = 0.037, maxf1 = 0.186, 
   tolerance 36 : mAP = 0.303, maxf1 = 0.518, 
   tolerance 60 : mAP = 0.529, maxf1 = 0.664, 
   tolerance 90 : mAP = 0.662, maxf1 = 0.731, 
   tolerance 120 : mAP = 0.711, maxf1 = 0.755, 
   tolerance 150 : mAP = 0.740, maxf1 = 0.771, 
   tolerance 180 : mAP = 0.759, maxf1 = 0.780, 
   tolerance 240 : mAP = 0.784, maxf1 = 0.792, 
   tolerance 300 : mAP = 0.805, maxf1 = 0.802, 
   tolerance 360 : mAP = 0.817, maxf1 = 0.808, 
```

## References
[1]: https://github.com/menouarazib/eventdetector
[2]: https://archive.physionet.org/physiobank/database/chbmit/

