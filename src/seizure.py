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
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import Dataset

import matplotlib.pyplot as plt

from math import pi, sqrt, exp

from src.utils import *


class SeizureDataset(Dataset):
    def __init__(
        self,
        dataclass: DataClass,
        data_dir: Path,
        fold: int,
        kfold: object = KFold(n_splits=5, shuffle=True, random_state=0),
        target_type: str = "gau",
        training: bool = True,
        downsample: int = 5,
        agg_feats: str = True,
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

        self.feats = [
            "FP1-F7",
            "F7-T7",
            "T7-P7",
            "P7-O1",
            "FP1-F3",
            "F3-C3",
            "C3-P3",
            "P3-O1",
            "FP2-F4",
            "F4-C4",
            "C4-P4",
            "P4-O2",
            "FP2-F8",
            "F8-T8",
            "T8-P8",
            "P8-O2",
            "FZ-CZ",
            "CZ-PZ",
            "P7-T7",
            "T7-FT9",
            "FT9-FT10",
            "FT10-T8",
        ]

        events = pd.read_csv(data_dir / "seizure_events.csv")
        self.ds_dir = data_dir / "seizure_256Hz_dataset"

        events["onset"] *= 256
        events["offset"] *= 256

        self.ids = events["series_id"].unique()
        self.targets = {
            id: (event["onset"].values, event["offset"].values)
            for id, event in events.groupby("series_id")
        }

        onset = events[["series_id", "onset"]]
        onset = onset.rename({"onset": "step"}, axis=1)
        onset["event"] = "onset"
        offset = events[["series_id", "offset"]]
        offset = offset.rename({"offset": "step"}, axis=1)
        offset["event"] = "offset"
        self.events = pd.concat([onset, offset], axis=0).sort_values(
            ["series_id", "step"]
        )

        for id in self.ids:
            if id not in self.targets:
                self.targets[id] = []

        if fold != -1:
            idxs = list(kfold.split(self.ids))[fold][0 if training else 1]
            self.ids = self.ids[idxs]
            del idxs

        self.events = self.events.loc[self.events.series_id.isin(self.ids)]
        self.targets = {id: self.targets[id] for id in self.ids}

    def get_step(self, series_id, idx):
        filepath = series_id.split(".")[0] + ".ftr"
        df = pd.read_feather(self.ds_dir / filepath, columns=["series_id"]).iloc[idx]
        df["step"] = df.index.astype(np.int32)
        return df

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        series_id = self.ids[index]

        filepath = series_id.split(".")[0] + ".ftr"
        X = (
            pd.read_feather(self.ds_dir / filepath)[self.feats]
            .fillna(0)
            .astype(np.float64)
            .values
            / 100
        )  # attempt to normalize
        # X = StandardScaler().fit_transform(
        #     pd.read_feather(self.ds_dir / filepath)[self.feats]
        #     .fillna(0)
        #     .astype(np.float64)
        #     .values
        # )  # attempt to normalize

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


def get_seizure_dataclass():
    return DataClass(
        name="seizure",
        combine_series_id=False,
        event_type="interval",
        num_feats=22,
        cat_feats=0,
        cat_uniq=0,
        tolerances=[256, 512, 1280, 2560, 5120, 15360],
        column_names={
            "series_id_column_name": "series_id",
            "time_column_name": "step",
            "event_column_name": "event",
            "score_column_name": "score",
        },
        max_distance=25600,
        gaussian_sigma=512,
        day_length=883728,  # length of total time series / total event length
        default_sequence_length=(1 * 60 * 60 * 256),  # 1 hour
        dataset_construct=SeizureDataset,
        evaluation_metrics=["mAP", "mf1"],
        hyperparams_tune=[
            "cutoff",
            "smooth",
        ],
    )
