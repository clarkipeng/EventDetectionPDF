import gc
import os
import random

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

from src.utils import *


class BowShockDataset(Dataset):
    def __init__(
        self,
        dataclass: DataClass,
        data_dir: Path,
        fold: int,
        kfold: object = KFold(n_splits=5, shuffle=True, random_state=0),
        target_type: str = "gau",
        training: bool = True,
        downsample: int = 5,
        agg_feats: bool = True,
        sequence_length: int = None,
        normalize: bool = True,
        use_cat: bool = True,
    ):
        data_dir = Path(data_dir)
        self.dataclass = dataclass
        self.downsample = downsample
        self.agg_feats = agg_feats
        self.sequence_length = sequence_length
        self.training = training
        self.target_type = target_type
        self.normalize = normalize

        self.cat_feats = 0

        ds = pd.read_pickle(data_dir / "martian_bow_shock_dataset.pkl")
        events = pd.read_csv(data_dir / "martian_bow_shock_events.csv")

        ds["step"] = ds.groupby(pd.Grouper(freq="D")).cumcount()
        ds["series_id"] = (ds.index.dayofyear - 1).astype(str)

        events["events"] = pd.to_datetime(events["events"])
        events["event"] = "event"
        events["step"] = (
            (
                (
                    events["events"].dt.hour * 3600
                    + events["events"].dt.minute * 60
                    + events["events"].dt.second
                    - 2
                )
                / 4
            )
            .round()
            .astype(int)
        )
        events["series_id"] = (events["events"].dt.dayofyear - 1).astype(str)

        ds[["ws_totels_1_new", "ws_totels_8", "ws_totels_6", "ws_rho"]] = (
            ds[["ws_totels_1_new", "ws_totels_8", "ws_totels_6", "ws_rho"]]
            .fillna(0)
            .astype(np.float32)
        )
        ds = ds.reset_index(drop=True)

        self.ids = ds["series_id"].unique()
        self.events = events
        self.data = {id: group for id, group in ds.groupby("series_id")}
        self.targets = {
            id: event["step"].tolist() for id, event in events.groupby("series_id")
        }
        for id in self.ids:
            if id not in self.targets:
                self.targets[id] = []

        if fold != -1:
            idxs = list(kfold.split(self.ids))[fold][0 if training else 1]
            self.ids = self.ids[idxs]
            del idxs

        self.events = self.events.loc[self.events.series_id.isin(self.ids)]
        self.data = {id: self.data[id] for id in self.ids}
        self.targets = {id: self.targets[id] for id in self.ids}

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        series_id = self.ids[index]

        feats = ["ws_totels_1_new", "ws_totels_8", "ws_totels_6", "ws_rho"]

        X = self.data[series_id][feats].values

        y = get_targets(
            self.dataclass,
            X.shape[0],
            self.targets[series_id],
            self.target_type,
            normalize=self.normalize,
        )

        #         X, y = downsample_sequence(X,self.downsample), downsample_sequence(y,self.downsample)
        mask = np.array([0, self.sequence_length])

        if self.training:
            if self.sequence_length:
                X, y, mask = maskpad_to_sequence_length(
                    X, y, mask, sequence_length=self.sequence_length, train=True
                )
            X = downsample_feats(X, self.downsample, self.cat_feats, self.agg_feats)
            y = downsample_sequence(y, self.downsample, "max")
            mask = mask // self.downsample

            return (
                torch.from_numpy(X),
                torch.from_numpy(y),
                torch.from_numpy(mask),
                series_id,
            )
        else:
            if self.sequence_length:
                Xs, ys, masks = maskpad_to_sequence_length(
                    X, y, mask, sequence_length=self.sequence_length, train=False
                )

            for i in range(len(Xs)):
                Xs[i] = downsample_feats(
                    Xs[i], self.downsample, self.cat_feats, self.agg_feats
                )
                ys[i] = downsample_sequence(ys[i], self.downsample, "max")
                masks[i] = masks[i] // self.downsample

            return Xs, ys, masks, series_id


def get_bowshock_dataclass():
    return DataClass(
        name="bowshock",
        combine_series_id=True,
        event_type="point",
        num_feats=4,
        cat_feats=0,
        cat_uniq=0,
        tolerances=[15, 30, 45, 60, 75, 150],
        column_names={
            "series_id_column_name": "series_id",
            "time_column_name": "step",
            "event_column_name": "event",
            "score_column_name": "score",
        },
        max_distance=1,
        gaussian_sigma=15,
        day_length=4510,  # length of total time series / total event length
        default_sequence_length=(24 * 60 * 15),
        dataset_construct=BowShockDataset,
    )
