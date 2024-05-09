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


def downsample_sequence(x, downsample_factor, method = 'standard', ):
    if method == 'standard':
        return x[::downsample_factor]
    else:
        # cut out end of array so that downsample divides the length of x
        end =  downsample_factor * int(len(x) / downsample_factor)
        x = x[:end] 

        # reshape x for downsampling and feature aggregation
        x = x.T
        x = np.reshape(x, (x.shape[0],x.shape[1]//downsample_factor,downsample_factor))
    
    if method == 'average':
        return np.mean(x, -1).T
    elif method == 'max':
        return np.max(x, -1).T
    else:
        raise ValueError('method not available')

def downsample_feats(x, downsample_factor, cat_feat = 2, agg_feats = True):
    # cut out end of array so that downsample divides the length of x
    length =  downsample_factor * (x.shape[0] // downsample_factor)
    feats = x.shape[1]
    x = x[:length] 
    
    # reshape x for downsampling and feature aggregation
    x = x.T
    x = np.reshape(x, (feats, length // downsample_factor, downsample_factor))
    
    if agg_feats:
        # aggregating features
        x = np.concatenate([np.max(x[:-cat_feat], -1),
                            np.min(x[:-cat_feat], -1),
                            np.mean(x[:-cat_feat], -1),
                            np.std(x[:-cat_feat], -1),
                            x[-cat_feat:,...,downsample_factor//2],
                           ],axis=0).T
    else:
        x = np.concatenate([np.mean(x[:-cat_feat], -1),
                            x[-cat_feat:,...,downsample_factor//2],
                           ],axis=0).T
        
    return x

def normalize_error(ttype = 'gau', sigma = 50, day_length = 24*60*12):
    
    if ttype == 'hard':
        distribution = np.ones(1)
    elif ttype == 'gau':
        r = range(-int(sigma*3),int(sigma*3)+1)
        distribution = np.array([exp((-float(x/sigma)**2)/2) for x in r])
    elif ttype == 'custom':
        distribution = np.zeros(360*2+1)
        for w in [12, 36, 60, 90, 120, 150, 180, 240, 300, 360]:
            i1,i2 = 360-w,360+w+1
            distribution[i1:i2] += 1/10
    
    return np.sqrt(np.sum(distribution**2) / day_length)

def get_targets(length, locations, ttype = 'gau', sigma = 50, normalize = True):
    if ttype[:3] == 'seg':
        target = np.zeros((length,1))
        for start,end in zip(*locations):
            target[int(start):int(end), 0] = 1
        return target
    
    if ttype == 'hard':
        target = np.zeros((length,2))
        for c, loc in enumerate(locations):
            target[loc,c] = 1
        target_variance = 0.007607257743127307 # normalize_error(ttype
    elif ttype == 'gau':
        r = range(-int(sigma*3),int(sigma*3)+1)
        gauss = np.array([exp((-float(x/sigma)**2)/2) for x in r])
    
        target = np.zeros((length,2))
        for c, loc in enumerate(locations):
            for i in loc:
                i1,i2 = max(0,i-sigma*3),min(length,i+sigma*3+1)
                target[i1:i2, c] = gauss[(i1-(i-sigma*3)):(sigma*6+1-((i+sigma*3+1)-i2))]
        target_variance = 0.071613698032125380 # normalize_error(ttype)
    elif ttype == 'custom':
        custom = np.zeros(360*2)
        for w in [12, 36, 60, 90, 120, 150, 180, 240, 300, 360]:
            i1,i2 = max(0,360-w),min(length,360+w+1)
            custom[i1:i2] += 1/10
            
        target = np.zeros((length,2))
        for c, loc in enumerate(locations):
            for i in loc:
                i1,i2 = max(0,i-360),min(length,i+360+1)
                target[i1:i2, c] = gauss[(i1-(i-360)):(720+1-((i+360+1)-i2))]
        target_variance = 0.104027685061522240 # normalize_error(ttype)
    if normalize:
        target = target / target_variance
    return target

def maskpad_to_sequence_length(X, y, mask, sequence_length = 7*(24*60*12), train = True):
    if train:
        if len(X)>sequence_length:
            st = int(random.random()*(len(X)-sequence_length))
            ed = st+sequence_length
            X = X[st:ed,:]
            y = y[st:ed,:]
            mask = np.array([0, sequence_length])
        else:
            ed = sequence_length-len(X)

            mask = np.array([0, len(X)])
            X = np.concatenate([X,np.zeros((ed,X.shape[1]))])
            y = np.concatenate([y,np.zeros((ed,y.shape[1]))])
            
    else:
        Xs = []
        ys = []
        masks = []
        if len(X)<sequence_length:
            ed = sequence_length-len(X)
            
            mask_ = np.array([0, len(X)])
            X_ = np.concatenate([X,np.zeros((ed,X.shape[1]))])
            y_ = np.concatenate([y,np.zeros((ed,y.shape[1]))])
            
            Xs.append(X_)
            ys.append(y_)
            masks.append(mask_)
        else:
            for i in range((len(X)+sequence_length-1)//sequence_length):
                if (i+1)*sequence_length<=len(X):
                    mask_ = np.array([0,sequence_length])
                    X_ = X[i*sequence_length:(i+1)*sequence_length]
                    y_ = y[i*sequence_length:(i+1)*sequence_length]
                elif len(X)<(i+1)*sequence_length:
                    start = sequence_length - (len(X)%sequence_length)
                    
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

class data_reader:
    def __init__(self, base_path : Path):
        super().__init__()
        self.names_mapping = {
            "submission" : {"path" : base_path / "sample_submission.csv", "is_parquet" : False, "has_timestamp" : False}, 
            "train_events" : {"path" : base_path / "train_events.csv", "is_parquet" : False, "has_timestamp" : True},
            "train_series" : {"path" : base_path / "train_series.parquet", "is_parquet" : True, "has_timestamp" : True},
            "test_series" : {"path" : base_path / "train_series.parquet", "is_parquet" : True, "has_timestamp" : True}
        }
        self.valid_names = ["submission", "train_events", "train_series", "test_series"]
    
    def verify(self, data_name):
        "function for data name verification"
        if data_name not in self.valid_names:
            print("PLEASE ENTER A VALID DATASET NAME, VALID NAMES ARE : ", valid_names)
        return
    
    def cleaning(self, data):
        "cleaning function : drop na values"
        before_cleaning = len(data)
        data = pl.DataFrame(data.to_pandas().dropna(subset = ["timestamp"]))
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
        
        data = data.sort(['series_id','step'])
        data = data.with_columns(pl.col('step').cast(pl.Int32)) # ensure datatypes match
        return data

def process_sleep_dataset(data_dir : str, 
                          use_time_cat : bool = False
                         ):
    data_dir = Path(data_dir)
    processed_filepath = data_dir / 'processed_data.npy'
    
    if not os.path.isfile(processed_filepath):
        
        reader = data_reader(data_dir)
        series = reader.load_data(data_name="train_series")
        events = reader.load_data(data_name="train_events")
        del reader
        
        # merge series with events to get the relevant event times
        series = series.join(events.select(["series_id", "step", "event"]), on=["series_id", "step"], how="left")
        
        # add time categorical features
        if use_time_cat:
            series = series.with_columns(pl.col('timestamp').str.to_datetime(format = '%Y-%m-%dT%H:%M:%S%z'),)
            series = series.with_columns(
                                        pl.col('timestamp').dt.hour().alias('hour'),
                                        pl.col('timestamp').dt.weekday().alias('wd')-1,
                                        )
            gc.collect()
        
        # pandas dataframe can iterate throught groups
        series = series.to_pandas().sort_values(['series_id','step']).reset_index(drop=True)
        events = events.to_pandas().sort_values(['series_id','step']).reset_index(drop=True)
        ids = series.series_id.unique()
        
        # store the data and targets into dictionaries
        data = {}
        targets = {}
        
        for series_id, viz_series in tqdm(series[['series_id', 'anglez', 'enmo', 'hour', 'wd', 'step']].groupby('series_id', sort=False)):
            data[series_id] = viz_series
            targets[series_id] = ([],[])
#         display(viz_series)
        for (series_id, event) in tqdm(series.loc[series.event.isin(['onset','wakeup']),['series_id','event','step',]].groupby('series_id', sort=False)):
            onset,wakeup = [],[]

            for i in range(len(event)-1):
                if event.iloc[i].event=='onset' and event.iloc[i+1].event=='wakeup':
                    onset.append(event.iloc[i].step)
                    wakeup.append(event.iloc[i+1].step)
            targets[series_id] = (np.array(onset),np.array(wakeup))
#         display(event)

        np.save(processed_filepath,
                (ids, events, data, targets)
               )
    return processed_filepath

class SleepDataset(Dataset):
    def __init__(
        self,
        data_dir : Path,
        fold : int,
        kfold : object = KFold(n_splits=5,shuffle=True,random_state=0),
        target_type : str = 'gau',
        training : bool = True,
        downsample : int = 5,
        agg_feats : bool = True,
        sequence_length : int = None,
        normalize : bool = True,
        use_time_cat : bool = True
    ):
        self.downsample = downsample
        self.agg_feats = agg_feats
        self.sequence_length = sequence_length
        self.training = training
        self.target_type = target_type
        self.normalize = normalize
        
        if use_time_cat:
            self.categorical_feats = 2
        else:
            self.categorical_feats = 0
        
        data_path = process_sleep_dataset(data_dir, use_time_cat)
        self.ids, self.events, self.data, self.targets = np.load(data_path, allow_pickle=True)
        
        if fold != -1:
            idxs = list(kfold.split(self.ids))[fold][0 if training else 1]
            self.ids = self.ids[idxs]
            del idxs
        
        self.events = self.events.loc[self.events.series_id.isin(self.ids)]
        self.data    = {id : self.data[id]    for id in self.ids}
        self.targets = {id : self.targets[id] for id in self.ids}
        
    def __len__(self):
        return len(self.ids)
    def __getitem__(self, index):
        series_id = self.ids[index]
        
        feats = ['anglez','enmo']
        if self.categorical_feats:
            feats += ['hour','wd']
            
        X = self.data[series_id][feats].values
        
        y = get_targets(X.shape[0], self.targets[series_id], self.target_type, normalize = self.normalize)
        
#         X, y = downsample_sequence(X,self.downsample), downsample_sequence(y,self.downsample)
        mask = np.array([0, self.sequence_length])
        
        if self.training:
            if self.sequence_length:
                X, y, mask = maskpad_to_sequence_length(X, y, mask, sequence_length = self.sequence_length, train = True)
            X = downsample_feats(X, self.downsample, self.categorical_feats, self.agg_feats)
            y = downsample_sequence(y,self.downsample, 'max')
            mask = mask // self.downsample
            
            return torch.from_numpy(X), torch.from_numpy(y), torch.from_numpy(mask), series_id
        else:
            if self.sequence_length:
                Xs, ys, masks = maskpad_to_sequence_length(X, y, mask, sequence_length = self.sequence_length, train = False)
                
            for i in range(len(Xs)):
                Xs[i] = downsample_feats(Xs[i], self.downsample, self.categorical_feats, self.agg_feats)
                ys[i] = downsample_sequence(ys[i], self.downsample, 'max')
                masks[i] = masks[i] // self.downsample
            
            return Xs, ys, masks, series_id