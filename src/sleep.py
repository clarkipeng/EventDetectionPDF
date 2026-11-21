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


FEATS = ["anglez", "enmo", "hour", "wd"]


def _load_events(data_dir: Path) -> pd.DataFrame:
    events = pl.read_csv(data_dir / "train_events.csv")
    events = events.filter(pl.col("timestamp").is_not_null())
    events = events.sort(["series_id", "step"]).with_columns(pl.col("step").cast(pl.Int32))
    events = events.to_pandas().sort_values(["series_id", "step"]).reset_index(drop=True)
    events.loc[events.event == "wakeup", "event"] = "offset"
    return events


def process_sleep_dataset(data_dir: str, use_cat: bool = False):
    """Stream train_series.parquet row group by row group into flat arrays on disk.

    The full series is ~128M rows; materializing it as pandas frames needs well over
    20 GiB, so we write one contiguous array per column plus per-series offsets and
    memory-map them at load time. Features match the original pipeline: anglez, enmo,
    and the UTC hour and weekday (0-6) parsed from the timestamp.
    """
    data_dir = Path(data_dir)
    out = data_dir / "processed"
    if (out / "done").exists():
        return out
    out.mkdir(exist_ok=True)

    pf = ParquetFile(data_dir / "train_series.parquet")
    n = pf.metadata.num_rows
    steps = np.lib.format.open_memmap(out / "step.npy", "w+", np.int32, (n,))
    feats = np.lib.format.open_memmap(out / "feats.npy", "w+", np.float32, (n, len(FEATS)))
    sid_codes = np.empty(n, dtype=np.int32)
    names: dict = {}
    i = 0
    for rg in tqdm(range(pf.num_row_groups), desc="preprocess"):
        df = pl.from_arrow(pf.read_row_group(rg))
        df = df.with_columns(
            pl.col("timestamp").str.to_datetime(format="%Y-%m-%dT%H:%M:%S%z")
        ).with_columns(
            pl.col("timestamp").dt.hour().alias("hour"),
            (pl.col("timestamp").dt.weekday() - 1).alias("wd"),
        )
        k = len(df)
        steps[i : i + k] = df["step"].to_numpy()
        feats[i : i + k] = df.select(FEATS).to_numpy().astype(np.float32)
        sid_codes[i : i + k] = [names.setdefault(x, len(names)) for x in df["series_id"].to_list()]
        i += k
        del df
    assert i == n

    # rows must be grouped by series and sorted by step within each series
    order = np.argsort(np.array(list(names.keys())))
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    key = rank[sid_codes]
    if not (np.all(np.diff(key) >= 0)):
        raise ValueError("train_series.parquet is not grouped by sorted series_id")
    ids = np.array(list(names.keys()))[order]
    bounds = np.concatenate([[0], np.flatnonzero(np.diff(key)) + 1, [n]])
    steps.flush(); feats.flush()
    np.save(out / "ids.npy", ids)
    np.save(out / "bounds.npy", bounds)

    # sleep windows: an onset immediately followed by an offset
    events = _load_events(data_dir)
    targets = {}
    for series_id, ev in events.groupby("series_id", sort=False):
        e, st = ev.event.values, ev.step.values
        onset = [st[j] for j in range(len(e) - 1) if e[j] == "onset" and e[j + 1] == "offset"]
        offset = [st[j + 1] for j in range(len(e) - 1) if e[j] == "onset" and e[j + 1] == "offset"]
        targets[series_id] = (np.array(onset), np.array(offset))
    pd.to_pickle(targets, out / "targets.pkl")
    (out / "done").touch()
    return out


class _SeriesStore:
    """Read-only view of one series: memory-mapped step and feature columns."""

    def __init__(self, steps, feats):
        self.steps, self.feats = steps, feats

    def __getitem__(self, cols):
        if cols == ["step"]:
            return pd.DataFrame({"step": np.asarray(self.steps)})
        return _Values(np.asarray(self.feats[:, [FEATS.index(c) for c in cols]]))


class _Values:
    def __init__(self, values):
        self.values = values


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

        out = process_sleep_dataset(data_dir, use_cat)
        self.ids = np.load(out / "ids.npy", allow_pickle=True)
        bounds = np.load(out / "bounds.npy")
        steps = np.load(out / "step.npy", mmap_mode="r")
        feats = np.load(out / "feats.npy", mmap_mode="r")
        all_targets = pd.read_pickle(out / "targets.pkl")
        self.events = _load_events(Path(data_dir))
        span = {sid: (bounds[j], bounds[j + 1]) for j, sid in enumerate(self.ids)}

        if fold != -1:
            idxs = list(kfold.split(self.ids))[fold][0 if training else 1]
            self.ids = self.ids[idxs]
            del idxs

        self.events = self.events.loc[self.events.series_id.isin(self.ids)]
        self.data = {
            id: _SeriesStore(steps[span[id][0] : span[id][1]], feats[span[id][0] : span[id][1]])
            for id in self.ids
        }
        self.targets = {id: all_targets.get(id, ([], [])) for id in self.ids}

    def get_step(self, series_id, idx):
        return pd.DataFrame({"step": np.asarray(self.data[series_id].steps[idx], dtype=np.int32)})

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
