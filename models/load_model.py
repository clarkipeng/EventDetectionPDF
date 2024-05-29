from models.BiRNN import MultiBiRNN
from models.PrecTime import PrecTime
from models.UNet1D import UNet1D

from src.utils import DataClass

def get_model(
    dataclass : DataClass,
    model_name: str,
    objective: str,
    sequence_length: int,
    agg_feats: bool = True,
    use_cat: bool = False,
):
    inputsize, outsize = 0, 0
    cat_feats, cat_unique = 0, 0
    
    # model output dimensions
    if objective[:3] == "seg" or dataclass.event_type == 'point':
        outsize = 1
    else:
        outsize = 2
    
    # dataset parameters
    inputsize = dataclass.num_feats
    if agg_feats:
        inputsize *= 4
    if use_cat:
        cat_feats = dataclass.cat_feats
        cat_unique = dataclass.cat_uniq
        

    if model_name == "rnn":
        return MultiBiRNN(
            input_channels=inputsize,
            cat_feats = cat_feats,
            cat_unique = cat_unique,
            n_layers=2,
            num_classes=outsize,
        )
    elif model_name == "unet":
        return UNet1D(
            input_channels=inputsize,
            sequence_length=sequence_length,
            cat_feats = cat_feats,
            cat_unique = cat_unique,
            num_classes=outsize,
            ks=7,
        )
    elif model_name == "unet_t":
        return UNet1D(
            input_channels=inputsize,
            sequence_length=sequence_length,
            cat_feats = cat_feats,
            cat_unique = cat_unique,
            num_classes=outsize,
            ks=7,
            use_attention=True,
        )
    elif model_name == "prectime":
        return PrecTime(
            input_channels=inputsize,
            sequence_length=sequence_length,
            cat_feats = cat_feats,
            cat_unique = cat_unique,
            num_classes=outsize,
        )
    raise ValueError("model not listed")