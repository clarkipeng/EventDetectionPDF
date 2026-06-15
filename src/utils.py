import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import gc
import os
import random
from typing import Dict, List, Tuple

import numpy as np

import pyarrow as pa
import pandas as pd
import polars as pl

from tqdm.auto import tqdm

from pathlib import Path

from pyarrow.parquet import ParquetFile
from sklearn.model_selection import KFold

import torch
from torch.utils.data import Dataset

import matplotlib.pyplot as plt

from math import pi, sqrt, exp

MSE_OBJECTIVES = ["hard", "gau", "custom"]
DENSITY_OBJECTIVES = [f"density_{objective}" for objective in MSE_OBJECTIVES]
SEGMENTATION_OBJECTIVES = ["seg", "seg1", "seg2"]
OBJECTIVE_CHOICES = SEGMENTATION_OBJECTIVES + MSE_OBJECTIVES + DENSITY_OBJECTIVES


def is_segmentation_objective(objective):
    return objective[:3] == "seg"


def is_density_objective(objective):
    return objective.startswith("density_")


def base_objective(objective):
    if is_density_objective(objective):
        return objective[len("density_") :]
    return objective


def downsample_sequence(
    x,
    downsample_factor,
    method="standard",
):
    if method == "standard":
        return x[::downsample_factor]
    else:
        # cut out end of array so that downsample divides the length of x
        end = downsample_factor * int(len(x) / downsample_factor)
        x = x[:end]

        # reshape x for downsampling and feature aggregation
        x = x.T
        x = np.reshape(
            x, (x.shape[0], x.shape[1] // downsample_factor, downsample_factor)
        )

    if method == "average":
        return np.mean(x, -1).T
    elif method == "max":
        return np.max(x, -1).T
    else:
        raise ValueError("method not available")


def downsample_feats(x, downsample_factor, cat_feat=2, agg_feats=True):
    # cut out end of array so that downsample divides the length of x
    length = downsample_factor * (x.shape[0] // downsample_factor)
    feats = x.shape[1]
    x = x[:length]

    # reshape x for downsampling and feature aggregation
    x = x.T
    x = np.reshape(x, (feats, length // downsample_factor, downsample_factor))

    if agg_feats == "stat":
        # aggregating features
        if cat_feat > 0:
            x = np.concatenate(
                [
                    np.max(x[:-cat_feat], -1),
                    np.min(x[:-cat_feat], -1),
                    np.mean(x[:-cat_feat], -1),
                    np.std(x[:-cat_feat], -1),
                    x[-cat_feat:, ..., downsample_factor // 2],
                ],
                axis=0,
            ).T
        else:
            x = np.concatenate(
                [
                    np.max(x, -1),
                    np.min(x, -1),
                    np.mean(x, -1),
                    np.std(x, -1),
                ],
                axis=0,
            ).T
    elif agg_feats == "all":
        if cat_feat > 0:
            x = np.concatenate(
                [x[:-cat_feat, ..., i] for i in range(downsample_factor)]
                + [
                    x[-cat_feat:, ..., downsample_factor // 2],
                ],
                axis=0,
            ).T
        else:
            x = np.concatenate(
                [x[:, ..., i] for i in range(downsample_factor)],
                axis=0,
            ).T
    elif agg_feats == "none":
        if cat_feat > 0:
            x = np.concatenate(
                [
                    np.mean(x[:-cat_feat], -1),
                    x[-cat_feat:, ..., downsample_factor // 2],
                ],
                axis=0,
            ).T
        else:
            x = np.mean(x, -1).T
    return x


def get_target_distribution(dataclass, ttype="gau", unit_mass=False):
    ttype = base_objective(ttype)

    if ttype == "hard":
        distribution = np.ones(1)
    elif ttype == "gau":
        sigma = dataclass.gaussian_sigma
        dlength = int(sigma * 3)
        r = range(-dlength, dlength + 1)
        distribution = np.array([exp((-float(x / sigma) ** 2) / 2) for x in r])
    elif ttype == "custom":
        dlength = max(dataclass.tolerances)
        distribution = np.zeros(dlength * 2 + 1)
        for w in dataclass.tolerances:
            i1, i2 = dlength - w, dlength + w + 1
            distribution[i1:i2] += 1 / len(dataclass.tolerances)
    else:
        raise ValueError(f"{ttype} is not implemented")

    if unit_mass:
        mass = np.sum(distribution)
        if mass > 0:
            distribution = distribution / mass

    return distribution


def normalize_error(dataclass, ttype="gau"):
    distribution = get_target_distribution(dataclass, ttype)

    return np.sqrt(np.sum(distribution**2) / dataclass.day_length)


def get_targets(dataclass, length, locations, ttype="gau", normalize=True):
    if dataclass.event_type == "interval":
        target = np.zeros((length, 2))
    elif dataclass.event_type == "point":
        target = np.zeros((length, 1))
    else:
        raise ValueError(f"{dataclass.event_type} is not implemented")

    if is_segmentation_objective(ttype):
        target = np.zeros((length, 1))

        if dataclass.event_type == "interval":
            for start, end in zip(*locations):
                target[int(start) : int(end), 0] = 1
        elif dataclass.event_type == "point":
            for loc in locations:
                target[loc, 0] = 1

        return target

    density_target = is_density_objective(ttype)
    distribution = get_target_distribution(
        dataclass, ttype, unit_mass=density_target
    )
    dlength = len(distribution) // 2

    def add_boundary(i, c):
        i1, i2 = max(0, i - dlength), min(length, i + dlength + 1)
        dist_i1 = i1 - (i - dlength)
        dist_i2 = len(distribution) - ((i + dlength + 1) - i2)
        values = distribution[dist_i1:dist_i2]
        if density_target:
            target[i1:i2, c] += values
        else:
            target[i1:i2, c] = np.maximum(target[i1:i2, c], values)

    if dataclass.event_type == "interval":
        for c, loc in enumerate(locations):
            for i in loc:
                add_boundary(int(i), c)
    elif dataclass.event_type == "point":
        for i in locations:
            add_boundary(int(i), 0)

    if normalize and not density_target:
        target_variance = normalize_error(dataclass, ttype)
        target = target / target_variance

    return target


def maskpad_to_sequence_length(
    X, y, mask, sequence_length=7 * (24 * 60 * 12), train=True
):
    if train:
        if len(X) > sequence_length:
            st = int(random.random() * (len(X) - sequence_length))
            ed = st + sequence_length
            X = X[st:ed, :]
            y = y[st:ed, :]
            mask = np.array([0, sequence_length])
        else:
            ed = sequence_length - len(X)

            mask = np.array([0, len(X)])
            X = np.concatenate([X, np.zeros((ed, X.shape[1]))])
            y = np.concatenate([y, np.zeros((ed, y.shape[1]))])

    else:
        Xs = []
        ys = []
        masks = []
        if len(X) < sequence_length:
            ed = sequence_length - len(X)

            mask_ = np.array([0, len(X)])
            X_ = np.concatenate([X, np.zeros((ed, X.shape[1]))])
            y_ = np.concatenate([y, np.zeros((ed, y.shape[1]))])

            Xs.append(X_)
            ys.append(y_)
            masks.append(mask_)
        else:
            for i in range((len(X) + sequence_length - 1) // sequence_length):
                if (i + 1) * sequence_length <= len(X):
                    mask_ = np.array([0, sequence_length])
                    X_ = X[i * sequence_length : (i + 1) * sequence_length]
                    y_ = y[i * sequence_length : (i + 1) * sequence_length]
                elif len(X) < (i + 1) * sequence_length:
                    start = sequence_length - (len(X) % sequence_length)

                    mask_ = np.array([start, sequence_length])
                    X_ = X[-sequence_length:]
                    y_ = y[-sequence_length:]

                Xs.append(X_)
                ys.append(y_)
                masks.append(mask_)
        X = Xs
        y = ys
        mask = masks
    return X, y, mask


def boundary_prior_rate(dataclass=None, downsample=1, eps=1e-8):
    if dataclass is None:
        return 1e-4
    return max(float(downsample) / float(dataclass.day_length), eps)


def inverse_softplus(value):
    if value > 20:
        return value
    return float(np.log(np.expm1(value)))


def density_logits_to_rates(prediction, dataclass=None, downsample=1, eps=1e-8):
    prior_rate = boundary_prior_rate(dataclass, downsample, eps)
    bias = torch.as_tensor(
        inverse_softplus(prior_rate),
        dtype=prediction.dtype,
        device=prediction.device,
    )
    return torch.nn.functional.softplus(prediction + bias) + eps


class BoundaryDensityLoss(nn.Module):
    def __init__(self, dataclass=None, downsample=1, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.prior_rate = boundary_prior_rate(dataclass, downsample, eps)
        self.register_buffer(
            "rate_bias",
            torch.tensor(inverse_softplus(self.prior_rate), dtype=torch.float32),
        )

    def forward(self, prediction, target):
        rate_bias = self.rate_bias.to(dtype=prediction.dtype, device=prediction.device)
        rate = torch.nn.functional.softplus(prediction + rate_bias) + self.eps
        return rate - target * torch.log(rate)


def get_loss(objective, dataclass=None, downsample=1):
    if is_segmentation_objective(objective):
        return nn.BCEWithLogitsLoss(reduction="none")
    if is_density_objective(objective):
        return BoundaryDensityLoss(dataclass=dataclass, downsample=downsample)
    return nn.MSELoss(reduction="none")


def set_random_seed(seed=0):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class DataClass:
    def __init__(
        self,
        name,  # the name of the data
        combine_series_id,  # whether to combine series_id in the evaluation portion, set to true if the dataset really just one whole time series
        event_type,  # whether the event is interval or point-based
        num_feats,  # how many numerical features in the time series
        cat_feats,  # how many categorical features in the time series
        cat_uniq,  # maximum number of unique categories over all categorical features
        tolerances,  # the tolerances values to use in evaluation
        column_names,  # dictionary containing column names for evaluation
        max_distance,  # the minimum timesteps between consecutive events
        gaussian_sigma,  # for the gau objective, what sigma to use
        day_length,  # roughly how many steps for every event
        default_sequence_length,  # the default sequence length for the model input
        dataset_construct,  # the torch.utils.data.Dataset constructor for the dataset
        evaluation_metrics: List[str] = ["mAP", "mf1"],  # the metrics to evaluate
        hyperparams_tune: List[str] = [
            "cutoff",
            "smooth",
        ],  # the hyperparameters to tune
    ):
        self.name = name
        self.combine_series_id = combine_series_id
        self.event_type = event_type
        self.num_feats = num_feats
        self.cat_feats = cat_feats
        self.cat_uniq = cat_uniq
        self.tolerances = tolerances
        self.column_names = column_names
        self.max_distance = max_distance
        self.gaussian_sigma = gaussian_sigma
        self.day_length = day_length
        self.default_sequence_length = default_sequence_length
        self.dataset_construct = dataset_construct

        self.evaluation_metrics = evaluation_metrics
        self.hyperparams_tune = hyperparams_tune
