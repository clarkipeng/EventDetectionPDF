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

It is possible to train the model on smaller machines, and without GPUs, but doing so may take considerable time (1-2 min vs 8 sec per epoch). Likewise, more resources will speed up training. The data is automatically preprocessed in the script. However, if the preprocessing step exceeds the system memory, data should be preprocessed with the [```process_sleep_dataset```](https://github.com/clarkipeng/SleepRegressionCPD/blob/main/src/load_dataset.py#L214) function on virtual machine with more system memory, e.g., Kaggle's kernels or Google Colab's notebooks.

*The required hard-disk space depends on number of models/objectives used. The predictions for each model are cached, each new model/objective combination takes ~100Mb more disk memory each.

**Software Requirements:**

If you are going to run these scripts yourself from scratch, we highly recommend doing so on a GPU. In order to run the scripts with a GPU, the `pytorch` (`v2.2.2`) library is required. For this, the following additional software is required on your system:

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
python train.py --datadir [path_to_dataset] --model [model_name] --objective [objective_name]
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

#### Example
Here is a example of running the training script on a machine with an A100 GPU and 32 GB of RAM:
```
python train.py --model rnn --objective seg --epochs 10 --folds 4
```
Which has the following output:
```
fold 0, epoch 1/10: train loss: 6.567397832870483, valid loss: 6.930822197241443, valid mAP: 0.023725936361229667
fold 0, epoch 2/10: train loss: 6.78070236387707, valid loss: 6.907888797138418, valid mAP: 0.10914835690691285
fold 0, epoch 3/10: train loss: 6.679683798835391, valid loss: 6.875822243280709, valid mAP: 0.23762788267994644
fold 0, epoch 4/10: train loss: 6.7658633050464445, valid loss: 6.803971287767802, valid mAP: 0.4163512646526496
fold 0, epoch 5/10: train loss: 6.409579731169201, valid loss: 6.758766402517046, valid mAP: 0.5198809448631438
fold 0, epoch 6/10: train loss: 6.453846749805269, valid loss: 6.72255140713283, valid mAP: 0.5747572606165572
fold 0, epoch 7/10: train loss: 6.510752859569731, valid loss: 6.703539259944644, valid mAP: 0.6029398896699416
fold 0, epoch 8/10: train loss: 6.657302674793062, valid loss: 6.692005929883037, valid mAP: 0.6203016541866593
fold 0, epoch 9/10: train loss: 6.529280798775809, valid loss: 6.688343141334397, valid mAP: 0.614679353918786
fold 0, epoch 10/10: train loss: 6.4540394601367765, valid loss: 6.687655284255743, valid mAP: 0.6100686457820407
fold 1, epoch 1/10: train loss: 6.878575847262428, valid loss: 7.165012356379758, valid mAP: 0.026449483645454887
fold 1, epoch 2/10: train loss: 6.660974343617757, valid loss: 7.139738639816642, valid mAP: 0.08194746778660686
fold 1, epoch 3/10: train loss: 6.648147060757592, valid loss: 7.100208722810814, valid mAP: 0.20839383224864505
fold 1, epoch 4/10: train loss: 6.560608795710972, valid loss: 7.02604735000194, valid mAP: 0.431084970328621
fold 1, epoch 5/10: train loss: 6.436552115849087, valid loss: 6.96284599310678, valid mAP: 0.5535395656379856
fold 1, epoch 6/10: train loss: 6.188678911754063, valid loss: 6.9215892566287, valid mAP: 0.5928761880922007
fold 1, epoch 7/10: train loss: 6.218804075604393, valid loss: 6.896084146871083, valid mAP: 0.608177435122389
fold 1, epoch 8/10: train loss: 6.160753295535133, valid loss: 6.884603602838689, valid mAP: 0.6023296312636748
fold 1, epoch 9/10: train loss: 6.352500415983654, valid loss: 6.881422186351341, valid mAP: 0.6119293322268933
fold 1, epoch 10/10: train loss: 6.206567832401821, valid loss: 6.880311411144077, valid mAP: 0.6084792277383417
fold 2, epoch 1/10: train loss: 6.805354277292888, valid loss: 6.674842920356794, valid mAP: 0.01632034699850942
fold 2, epoch 2/10: train loss: 6.774743068785894, valid loss: 6.653295650443845, valid mAP: 0.10311424578058237
fold 2, epoch 3/10: train loss: 6.763583569299607, valid loss: 6.63547653440332, valid mAP: 0.21338767548111703
fold 2, epoch 4/10: train loss: 6.904300212860107, valid loss: 6.59302644615156, valid mAP: 0.4290074214995284
fold 2, epoch 5/10: train loss: 6.6846467199779696, valid loss: 6.515500627674054, valid mAP: 0.538883034249319
fold 2, epoch 6/10: train loss: 6.73179292678833, valid loss: 6.483213160348975, valid mAP: 0.5899375408186134
fold 2, epoch 7/10: train loss: 6.780517248880296, valid loss: 6.463352131238882, valid mAP: 0.6002036830425237
fold 2, epoch 8/10: train loss: 6.468309697650728, valid loss: 6.451126497616802, valid mAP: 0.6125391887380383
fold 2, epoch 9/10: train loss: 6.542498497735886, valid loss: 6.444739273168903, valid mAP: 0.6256097831702581
fold 2, epoch 10/10: train loss: 6.504251502809071, valid loss: 6.443862869795682, valid mAP: 0.6286895503900445
fold 3, epoch 1/10: train loss: 7.531470934549968, valid loss: 6.099467979825062, valid mAP: 0.012911559282793049
fold 3, epoch 2/10: train loss: 6.723584493001302, valid loss: 6.081378569178607, valid mAP: 0.038799227297509536
fold 3, epoch 3/10: train loss: 6.891559010460263, valid loss: 6.068587892391867, valid mAP: 0.11954856493338833
fold 3, epoch 4/10: train loss: 7.002564180464971, valid loss: 6.023344616365174, valid mAP: 0.3053650198495621
fold 3, epoch 5/10: train loss: 7.033120382399786, valid loss: 5.962538446367219, valid mAP: 0.4746323772465527
fold 3, epoch 6/10: train loss: 6.73004842939831, valid loss: 5.928792620536642, valid mAP: 0.5526360826985095
fold 3, epoch 7/10: train loss: 6.812300954546247, valid loss: 5.905685964215925, valid mAP: 0.5891628880283393
fold 3, epoch 8/10: train loss: 6.796495460328602, valid loss: 5.895802779150182, valid mAP: 0.6033776688176755
fold 3, epoch 9/10: train loss: 6.862480027335031, valid loss: 5.890804661356884, valid mAP: 0.5996965166696722
fold 3, epoch 10/10: train loss: 6.608388991582961, valid loss: 5.89002806790497, valid mAP: 0.5943980291333677
rnn hard results: 
 default score = 0.5664155029082533
 optimize hyperparams:
  best params: cutoff = 0.0, smoothing = 11
  best score = 0.6166905528872586
   tolerance 12 = 0.03872348644742643
   tolerance 36 = 0.2953151010695607
   tolerance 60 = 0.5226400268910409
   tolerance 90 = 0.6570232916251942
   tolerance 120 = 0.712453832558264
   tolerance 150 = 0.7449724346306841
   tolerance 180 = 0.7662430131158864
   tolerance 240 = 0.7922328553750166
   tolerance 300 = 0.8119013357894826
   tolerance 360 = 0.82540015137003
```

## References

