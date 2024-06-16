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


class data_reader:
    def __init__(self, base_path: Path):
        super().__init__()
        self.names_mapping = {
            "submission": {
                "path": base_path / "sample_submission.csv",
                "is_parquet": False,
                "has_timestamp": False,
            },
            "train_events": {
                "path": base_path / "train_events.csv",
                "is_parquet": False,
                "has_timestamp": True,
            },
            "train_series": {
                "path": base_path / "train_series.parquet",
                "is_parquet": True,
                "has_timestamp": True,
            },
            "test_series": {
                "path": base_path / "train_series.parquet",
                "is_parquet": True,
                "has_timestamp": True,
            },
        }
        self.valid_names = ["submission", "train_events", "train_series", "test_series"]

    def verify(self, data_name):
        "function for data name verification"
        if data_name not in self.valid_names:
            print(
                "PLEASE ENTER A VALID DATASET NAME, VALID NAMES ARE : ",
                self.valid_names,
            )
        return

    def cleaning(self, data):
        "cleaning function : drop na values"
        before_cleaning = len(data)
        data = pl.DataFrame(data.to_pandas().dropna(subset=["timestamp"]))
        after_cleaning = len(data)
        return data

    def load_data(self, data_name):
        "function for data loading"
        self.verify(data_name)
        data_props = self.names_mapping[data_name]
        if data_props["is_parquet"]:
            data = pl.read_parquet(data_props["path"])
        else:
            data = pl.read_csv(data_props["path"])

        if data_props["has_timestamp"]:
            data = self.cleaning(data)

        data = data.sort(["series_id", "step"])
        data = data.with_columns(
            pl.col("step").cast(pl.Int32)
        )  # ensure datatypes match
        return data


def process_sleep_dataset(data_dir: str, use_cat: bool = False):
    data_dir = Path(data_dir)
    processed_filepath = data_dir / "processed_data.npy"

    if not os.path.isfile(processed_filepath):

        reader = data_reader(data_dir)
        series = reader.load_data(data_name="train_series")
        events = reader.load_data(data_name="train_events")
        del reader

        # merge series with events to get the relevant event times
        series = series.join(
            events.select(["series_id", "step", "event"]),
            on=["series_id", "step"],
            how="left",
        )

        feats = ["anglez", "enmo"]
        # add time categorical features
        if use_cat:
            series = series.with_columns(
                pl.col("timestamp").str.to_datetime(format="%Y-%m-%dT%H:%M:%S%z"),
            )
            series = series.with_columns(
                pl.col("timestamp").dt.hour().alias("hour"),
                pl.col("timestamp").dt.weekday().alias("wd") - 1,
            )
            feats += ["hour", "wd"]
            gc.collect()

        # pandas dataframe can iterate throught groups
        series = (
            series.to_pandas().sort_values(["series_id", "step"]).reset_index(drop=True)
        )
        events = (
            events.to_pandas().sort_values(["series_id", "step"]).reset_index(drop=True)
        )
        series.loc[series.event == "wakeup", "event"] = "offset"
        events.loc[events.event == "wakeup", "event"] = "offset"
        ids = series.series_id.unique()

        # store the data and targets into dictionaries
        data = {}
        targets = {}

        for series_id, viz_series in tqdm(
            series[["series_id", "step"] + feats].groupby("series_id", sort=False)
        ):
            data[series_id] = viz_series
            targets[series_id] = ([], [])

        for series_id, event in tqdm(
            series.loc[
                series.event.isin(["onset", "offset"]),
                [
                    "series_id",
                    "event",
                    "step",
                ],
            ].groupby("series_id", sort=False)
        ):
            onset, offset = [], []

            for i in range(len(event) - 1):
                if (
                    event.iloc[i].event == "onset"
                    and event.iloc[i + 1].event == "offset"
                ):
                    onset.append(event.iloc[i].step)
                    offset.append(event.iloc[i + 1].step)
            targets[series_id] = (np.array(onset), np.array(offset))

        np.save(processed_filepath, (ids, events, data, targets))
    return processed_filepath


class SleepDataset(Dataset):
    def __init__(
        self,
        dataclass: DataClass,
        data_dir: Path,
        fold: int,
        kfold: object = KFold(n_splits=5, shuffle=True, random_state=0),
        target_type: str = "gau",
        training: bool = True,
        downsample: int = 5,
        agg_feats: str = "stat",
        sequence_length: int = None,
        normalize: bool = True,
        use_cat: bool = True,
    ):
        self.dataclass = dataclass
        self.downsample = downsample
        self.agg_feats = agg_feats
        self.sequence_length = sequence_length
        self.training = training
        self.target_type = target_type
        self.normalize = normalize

        if use_cat:
            self.cat_feats = 2
        else:
            self.cat_feats = 0

        data_path = process_sleep_dataset(data_dir, use_cat)
        self.ids, self.events, self.data, self.targets = np.load(
            data_path, allow_pickle=True
        )

        if fold != -1:
            idxs = list(kfold.split(self.ids))[fold][0 if training else 1]
            self.ids = self.ids[idxs]
            del idxs

        self.events = self.events.loc[self.events.series_id.isin(self.ids)]
        self.data = {id: self.data[id] for id in self.ids}
        self.targets = {id: self.targets[id] for id in self.ids}

    def get_step(self, series_id, idx):
        return self.data[series_id][["step"]].iloc[idx].astype(np.int32)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        series_id = self.ids[index]

        feats = ["anglez", "enmo"]
        if self.cat_feats:
            feats += ["hour", "wd"]

        X = self.data[series_id][feats].values

        y = get_targets(
            self.dataclass,
            X.shape[0],
            self.targets[series_id],
            self.target_type,
            normalize=self.normalize,
        )

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


def get_sleep_dataclass():
    return DataClass(
        name="sleep",
        combine_series_id=False,
        event_type="interval",
        num_feats=2,
        cat_feats=2,
        cat_uniq=24,
        tolerances=[12, 36, 60, 90, 120, 150, 180, 240, 300, 360],
        column_names={
            "series_id_column_name": "series_id",
            "time_column_name": "step",
            "event_column_name": "event",
            "score_column_name": "score",
        },
        max_distance=(30 * 12),
        gaussian_sigma=50,
        day_length=(24 * 60 * 12),
        default_sequence_length=7 * (24 * 60 * 12),
        dataset_construct=SleepDataset,
        evaluation_metrics=["mAP", "mf1"],
        hyperparams_tune=["cutoff", "smooth"],
    )
