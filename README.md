# General Regression-based Change Point Detection

This repository may be used to train all the the models used for experiments in the paper

## Contents

- [Overview](#overview)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [Data Preparation](#data-preparation)
- [Training and Evaluation](#training-and-evaluation)
- [References](#references)


## Overview
This document describes the official software package developed for and used to create the general regression-based approach for sleep CPD. It features different models, like [PrecTime](https://arxiv.org/ftp/arxiv/papers/2302/2302.10182.pdf), 1D UNets, and Bidirectional RNNS.

This software allows the training of binary sleep CPD models across Child Mind Institute's [sleep detection dataset](https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states/data). It features a command-line interface for training and evaluating models without needing to modify the underlying codebase.

## System Requirements
**Minimal Hardware Requirements**

For training our models from scratch,  we recommend using a Linux based computer with at least the following hardware specifications:

* 4+ CPU cores
* 26+ GiB RAM
* 5+ GiB physical storage space*
* 1 CUDA enabled GPU (please refer to [https://developer.nvidia.com/cuda-gpus](https://developer.nvidia.com/cuda-gpus) for a detailed list).

It is possible to train the model on smaller machines, and without GPUs, but doing so may take considerable time (1-2 min vs 8 sec per epoch). Likewise, more resources will speed up training. The data is automatically preprocessed in the script. However, if the preprocessing step exceeds the system memory, data should be preprocessed with the [```process_sleep_dataset```](https://github.com/clarkipeng/SleepRegressionCPD/blob/b39819c3f26d81214b49c3d9914f6613f4227078/sleep/load_dataset.py#L214) function on virtual machine with more system memory, e.g., Kaggle's kernels or Google Colab's notebooks.

*The required hard-disk space depends on number of models/objectives used. The predictions for each model are cached, each new model/objective combination takes ~100Mb more disk memory each.

**Software Requirements:**

If you are going to run these scripts yourself from scratch, we highly recommend doing so on a GPU. In order to run the scripts with a GPU, the `pytorch` (`v2.2.2`) library is required. For this, the following additional software is required on your system:

* [NVIDIA GPU drivers](https://www.nvidia.com/drivers)
* [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive)
* [cuDNN SDK](https://developer.nvidia.com/cudnn)

Please refer to [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) for additional details.

## Installation Guide
On a Linux machine with at least 1 CUDA enabled GPU available and `anaconda` 
or `miniconda` installed, run the following commands to download the required packages.

```
git clone https://github.com/clark/.git
pip install -r regressioneventdetection/requirements.txt
```

## Data Preparation:

Download the [sleep detection dataset](https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states/data) from Kaggle.com. Place the downloaded dataset in a new directory in called ```data```. We require the directory structure to include the following: 
```
path/to/repo/data
  train_series.parquet    # sleep timeseries
  train_events.csv        # sleep events
```

#### Requirements
- Completion of the steps outlined in the [Installation Guide](#installation-guide).
- Minimum of `8 GiB` of available disk-space on your computer.


## Training and Evaluation
We have 3 different scripts to aid in training and evaluation. These scripts are train_all.py, train.py, and eval.py.
After you have done all the necessary steps listed above, you are ready to train and evaluate the models. In order to train a model on a certain objective, you can simply run the following script with the names of the models and objectives: 

```
python train.py --datadir [path_to_dataset] --model [model_name] --objective[objective_name]
```
Model choices can be *rnn*, *unet*, *unet_t*, or *prectime*. Objectives can be *hard*, *gau*, *custom*, *seg1*, *seg2*, or *seg* (which evaluates both segmentation methods),.

In order to evaluate the trained models, run: 
```
python eval.py --datadir [path_to_dataset]
```

In order to train all models on all objectives, run: 
```
python train_all.py --datadir [path_to_dataset]
```

Here is a sample of the training script's output: 
```
fold 0, epoch 1/10: 0.5671964131650471, 0.4234076938458851, 0.31992529231908395
fold 0, epoch 2/10: 0.29704598230975016, 0.6864788130990096, 0.08083032501344567
fold 0, epoch 3/10: 0.2191676085903531, 0.5923197992146015, 0.11066386464176817
fold 0, epoch 4/10: 0.20828872919082642, 0.9550961809498923, 0.07028854306906512
fold 0, epoch 5/10: 0.19250523476373582, 0.44821978711656163, 0.16583883340180539
fold 0, epoch 6/10: 0.1849517673254013, 0.454849069246224, 0.1284257481328586
fold 0, epoch 7/10: 0.17295101214022862, 0.4682254411280155, 0.1978212962032277
fold 0, epoch 8/10: 0.17070088429110392, 0.432587339037231, 0.21699563620228057
fold 0, epoch 9/10: 0.1600088438107854, 0.49922254362276625, 0.18229525573739774
fold 0, epoch 10/10: 0.15780407119364964, 0.4201317543962172, 0.23296281634201355
fold 1, epoch 1/10: 0.5483487645785013, 0.4680862590886544, 0.18323421882567298
fold 1, epoch 2/10: 0.3187335119360969, 0.46725796659787494, 0.10480827214394019
fold 1, epoch 3/10: 0.2443289224590574, 0.37346307069495105, 0.1960867786987105
fold 1, epoch 4/10: 0.20523616884435927, 0.3566333326524582, 0.1875781833001176
fold 1, epoch 5/10: 0.18336566431181772, 0.3343870410884636, 0.19519207240391212
fold 1, epoch 6/10: 0.20561428226175762, 0.46166559082010517, 0.16433296611616088
fold 1, epoch 7/10: 0.17510004057770684, 0.4174952756451524, 0.17953786043311037
fold 1, epoch 8/10: 0.16072067263580503, 0.4000339227310125, 0.15939044648403256
fold 1, epoch 9/10: 0.16655622990358443, 0.35755097336959146, 0.19767021852032862
fold 1, epoch 10/10: 0.15015163272619247, 0.29273444457330566, 0.2539439317167594
fold 2, epoch 1/10: 0.5716611416566939, 0.44695411557736603, 0.3704339427564968
fold 2, epoch 2/10: 0.30280167148226783, 0.5337080663960913, 0.07620227404235262
fold 2, epoch 3/10: 0.25504277859415325, 0.33226127976524655, 0.12377035172468662
fold 2, epoch 4/10: 0.21594843836057753, 0.2971419591618621, 0.20882337498810477
fold 2, epoch 5/10: 0.19351212609381901, 0.24783526242211246, 0.26625211557722595
fold 2, epoch 6/10: 0.17813395034699214, 0.21275146764473638, 0.3853267492283511
fold 2, epoch 7/10: 0.1783287145552181, 0.20156077646474907, 0.3576166921813896
fold 2, epoch 8/10: 0.1534084223565601, 0.2731242938965991, 0.2476186956908779
fold 2, epoch 9/10: 0.16390727105594816, 0.25416029028702475, 0.2957066108423385
fold 2, epoch 10/10: 0.15014894058307013, 0.25443818193414935, 0.28567584360145826
fold 3, epoch 1/10: 0.5586394227686382, 0.48854703229406604, 0.25436108213664976
fold 3, epoch 2/10: 0.2893033680461702, 0.3698693669360617, 0.19252852677887927
fold 3, epoch 3/10: 0.21832286105269477, 0.5886409913284191, 0.06944601142260409
fold 3, epoch 4/10: 0.23850345753488086, 0.2573538360496362, 0.2743737832262192
fold 3, epoch 5/10: 0.2123905136471703, 0.24408283613730167, 0.3160624596505167
fold 3, epoch 6/10: 0.19378373700947987, 0.20700542445200076, 0.3928765627199209
fold 3, epoch 7/10: 0.18373608873004005, 0.19282431204033934, 0.3524866896348051
fold 3, epoch 8/10: 0.1810820684546516, 0.19001440012800522, 0.40478266054188033
fold 3, epoch 9/10: 0.170177637111573, 0.18557492951336113, 0.40961172855500694
fold 3, epoch 10/10: 0.17092583115611756, 0.18621937409583209, 0.41553354565278633
best cutoff = 1.0
best score = 0.2918882416699422
tolerance 12 = 0.010204054951315433
tolerance 36 = 0.07261413401159846
tolerance 60 = 0.15295192056427098
tolerance 90 = 0.2517985226366843
tolerance 120 = 0.3253561944472322
tolerance 150 = 0.3697693880261175
tolerance 180 = 0.39780092367320635
tolerance 240 = 0.4301438533392494
tolerance 300 = 0.44834313361339495
tolerance 360 = 0.45965576171985667
```

## References

