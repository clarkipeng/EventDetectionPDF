from models.BiRNN import MultiBiRNN
from models.PrecTime import PrecTime
from models.UNet1D import UNet1D

from src.utils import DataClass


def get_model(
    dataclass: DataClass,
    model_name: str,
    objective: str,
    sequence_length: int,
    downsample: int,
    agg_feats: str = "stat",
    use_cat: bool = False,
):
    inputsize, outsize = 0, 0
    cat_feats, cat_unique = 0, 0

    # model output dimensions
    if objective[:3] == "seg" or dataclass.event_type == "point":
        outsize = 1
    else:
        outsize = 2

    # dataset parameters
    inputsize = dataclass.num_feats
    if agg_feats == "stat":
        inputsize *= 4
    if agg_feats == "all":
        inputsize *= downsample
    if use_cat:
        cat_feats = dataclass.cat_feats
        cat_unique = dataclass.cat_uniq

    if model_name == "rnn":
        return MultiBiRNN(
            input_channels=inputsize,
            cat_feats=cat_feats,
            cat_unique=cat_unique,
            n_layers=2,
            num_classes=outsize,
        )
    elif model_name == "prectime":
        return PrecTime(
            input_channels=inputsize,
            sequence_length=sequence_length,
            cat_feats=cat_feats,
            cat_unique=cat_unique,
            num_classes=outsize,
        )
    elif model_name[:4] == "unet":

        use_attention = False
        layers = 3

        if "t" in model_name.split("_"):
            use_attention = True
        for arg in model_name.split("_"):
            if arg[:-1].isdigit() and arg[-1] == "l":
                layers = int(arg[:-1])
        assert layers >= 3

        return UNet1D(
            channels=[
                64,
                128,
                256,
            ]
            + [256] * (layers - 1),
            input_channels=inputsize,
            sequence_length=sequence_length,
            cat_feats=cat_feats,
            cat_unique=cat_unique,
            num_classes=outsize,
            ks=7,
            use_attention=use_attention,
        )
    raise ValueError("model not listed")
