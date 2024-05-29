import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

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
        if cat_feat > 0:
            x = np.concatenate([np.max(x[:-cat_feat], -1),
                                np.min(x[:-cat_feat], -1),
                                np.mean(x[:-cat_feat], -1),
                                np.std(x[:-cat_feat], -1),
                                x[-cat_feat:,...,downsample_factor//2],
                               ],axis=0).T
        else:
            x = np.concatenate([np.max(x, -1),
                                np.min(x, -1),
                                np.mean(x, -1),
                                np.std(x, -1),
                               ],axis=0).T
    else:
        if cat_feat > 0:
            x = np.concatenate([np.mean(x[:-cat_feat], -1),
                                x[-cat_feat:,...,downsample_factor//2],
                               ],axis=0).T
        else:
            x = np.mean(x, -1).T
    return x

def normalize_error(dataclass, ttype = 'gau'):
    
    if ttype == 'hard':
        distribution = np.ones(1)
    elif ttype == 'gau':
        sigma = dataclass.gaussian_sigma
        dlength = int(sigma*3)
        r = range(-dlength ,dlength +1)
        distribution = np.array([exp((-float(x/sigma)**2)/2) for x in r])
    elif ttype == 'custom':
        dlength = max(dataclass.tolerances)
        distribution = np.zeros(dlength*2+1)
        for w in dataclass.tolerances:
            i1,i2 = dlength-w,dlength+w+1
            distribution[i1:i2] += 1/len(dataclass.tolerances)
    
    return np.sqrt(np.sum(distribution**2) / dataclass.day_length)


def normalize_error(dataclass, ttype = 'gau'):
    
    if ttype == 'hard':
        distribution = np.ones(1)
    elif ttype == 'gau':
        sigma = dataclass.gaussian_sigma
        dlength = int(sigma*3)
        r = range(-dlength ,dlength +1)
        distribution = np.array([exp((-float(x/sigma)**2)/2) for x in r])
    elif ttype == 'custom':
        dlength = max(dataclass.tolerances)
        distribution = np.zeros(dlength*2+1)
        for w in dataclass.tolerances:
            i1,i2 = dlength-w,dlength+w+1
            distribution[i1:i2] += 1/len(dataclass.tolerances)
    
    return np.sqrt(np.sum(distribution**2) / dataclass.day_length)

def get_targets(dataclass, length, locations, ttype = 'gau', normalize = True):
    if dataclass.event_type == 'interval':
        target = np.zeros((length,2))
    elif dataclass.event_type == 'point':
        target = np.zeros((length,1))
    else:
        raise ValueError(f'{dataclass.event_type} is not implemented')
    
    if ttype[:3] == 'seg':
        target = np.zeros((length,1))
        
        if dataclass.event_type == 'interval':
            for start,end in zip(*locations):
                target[int(start):int(end), 0] = 1
        elif dataclass.event_type == 'point':
            for loc in locations:
                target[loc,0] = 1
        
        return target
    
    if ttype == 'hard':
        
        if dataclass.event_type == 'interval':
            for c, loc in enumerate(locations):
                if len(loc)>0:
                    target[loc,c] = 1
        elif dataclass.event_type == 'point':
            for loc in locations:
                target[loc,0] = 1
        
    else:
        if ttype == 'gau':
            sigma = dataclass.gaussian_sigma
            dlength = int(sigma*3)
            r = range(-dlength ,dlength +1)
            distribution = np.array([exp((-float(x/sigma)**2)/2) for x in r])
        elif ttype == 'custom':
            dlength = max(dataclass.tolerances)
            distribution = np.zeros(dlength*2+1)
            for w in dataclass.tolerances:
                i1,i2 = dlength-w,dlength+w+1
                distribution[i1:i2] += 1/len(dataclass.tolerances)
        
    
        if dataclass.event_type == 'interval':
            for c, loc in enumerate(locations):
                for i in loc:
                    i1,i2 = max(0,i-dlength),min(length,i+dlength+1)
                    target[i1:i2, c] = np.max([target[i1:i2, 0],distribution[(i1-(i-dlength)):(2*dlength+1-((i+dlength+1)-i2))]],axis=0)
        elif dataclass.event_type == 'point':
            for i in locations:
                i1,i2 = max(0,i-dlength),min(length,i+dlength+1)
                target[i1:i2, 0] = np.max([target[i1:i2, 0],distribution[(i1-(i-dlength)):(2*dlength+1-((i+dlength+1)-i2))]],axis=0)
                
    if normalize:
        target_variance = normalize_error(dataclass, ttype)
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

def get_loss(objective):
    if objective[:3] == "seg":
        return nn.BCEWithLogitsLoss(reduction="none")
    return nn.MSELoss(reduction="none")

class DataClass():
    def __init__(self, 
                 name, 
                 combine_series_id,
                 event_type,
                 num_feats, 
                 cat_feats, 
                 cat_uniq, 
                 tolerances,
                 column_names,
                 max_distance,
                 gaussian_sigma,
                 day_length,
                 default_sequence_length,
                 dataset_construct,
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