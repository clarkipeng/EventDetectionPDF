from models.BiRNN import MultiBiRNN
from models.OnlineModels import CausalTransformer
from models.PrecTime import PrecTime
from models.UNet1D import UNet1D

from src.utils import DataClass, is_segmentation_objective
from torch import Tensor, nn


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
    if is_segmentation_objective(objective) or dataclass.event_type == "point":
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

    model_parts = model_name.split("_")
    model_prefix = model_parts[0]

    if model_prefix in ["rnn", "gru", "lstm", "frnn", "fgru", "flstm"]:
        layers = 2
        hidden_size = 32
        bidir = not (
            model_prefix.startswith("f")
            or any(arg in ["forward", "fw", "online"] for arg in model_parts)
        )

        for arg in model_parts:
            if arg[:-1].isdigit() and arg[-1] == "l":
                layers = int(arg[:-1])
            if arg[:-1].isdigit() and arg[-1] == "h":
                hidden_size = int(arg[:-1])

        rnn_name = model_prefix[1:] if model_prefix.startswith("f") else model_prefix
        return MultiBiRNN(
            input_channels=inputsize,
            cat_feats=cat_feats,
            cat_unique=cat_unique,
            n_layers=layers,
            rnn_unit={"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}[rnn_name],
            hidden_size=hidden_size,
            num_classes=outsize,
            bidir=bidir,
        )
    elif model_prefix in ["causal", "decoder", "ct", "transformer", "tf"]:
        layers = 4
        hidden_size = 64
        heads = 4
        dropout = 0.1
        causal = model_prefix in ["causal", "decoder", "ct"]

        for arg in model_parts:
            if arg[:-1].isdigit() and arg[-1] == "l":
                layers = int(arg[:-1])
            if arg[:-1].isdigit() and arg[-1] == "h":
                hidden_size = int(arg[:-1])
            if arg[:-1].isdigit() and arg[-1] == "a":
                heads = int(arg[:-1])
            if arg[:-1].isdigit() and arg[-1] == "d":
                dropout = int(arg[:-1]) / 100.0

        return CausalTransformer(
            input_channels=inputsize,
            cat_feats=cat_feats,
            cat_unique=cat_unique,
            n_layers=layers,
            hidden_size=hidden_size,
            num_heads=heads,
            dropout=dropout,
            num_classes=outsize,
            causal=causal,
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
        ks = 7

        if "t" in model_parts:
            use_attention = True
        for arg in model_parts:
            if arg[:-1].isdigit() and arg[-1] == "l":
                layers = int(arg[:-1])
            if arg[:-2].isdigit() and arg[-2:] == "ks":
                ks = int(arg[:-2])

        channels = [
            64,
            128,
            256,
        ] + [int(256 * 1.5**i) for i in range(1, layers - 3 + 1)]
        return UNet1D(
            channels=channels[:layers],
            input_channels=inputsize,
            sequence_length=sequence_length,
            cat_feats=cat_feats,
            cat_unique=cat_unique,
            num_classes=outsize,
            ks=ks,
            use_attention=use_attention,
        )
    raise ValueError("model not listed")
