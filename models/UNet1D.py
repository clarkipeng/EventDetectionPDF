import torch
import torch.nn.functional as F
from torch import Tensor, nn

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, x, mask=None):
        x, _ = self.attention(x, x, x, attn_mask=mask)
        return x

def conv_block(input_channels, out_ch, ks, st, pd=None):
    pd = 0 if pd == 0 else (ks - 1) // 2

    return nn.Sequential(
        nn.Conv1d(input_channels, out_ch, ks, st, pd),
        nn.BatchNorm1d(out_ch),
        nn.ReLU(inplace=True),
    )

class DoubleConv(nn.Module):
    def __init__(self, input_channels, out_ch, ks=3, st=1, pd=None):
        super(DoubleConv, self).__init__()

        self.conv = nn.Sequential(
            conv_block(input_channels, out_ch, ks, st, pd),
            conv_block(out_ch, out_ch, ks, st, pd),
        )

    def forward(self, x):
        return self.conv(x)

class UNet1D(nn.Module):
    """Temporal Event Regressor

    Args:
        input_channels: Input channel (Number of features).
        num_classes (int): Number of output classes.
        channels (list): Output channels of the blocks in the Encoder.
        ks (int): Kernel size.
        st (int): Stride.
        pd (int): Padding.

    Example:
        >>> X = torch.randn(5, 3, 1440)  # Dims: (batch, features, length)
        >>> model = TemporalEventRegressor(input_channels=2, npts=2)
        >>> y = model(X) # Output Dims: (batch, output, length)
    """
    def __init__(
        self,
        input_channels : int = 2,
        num_classes : int = 2,
        use_time_cat : bool = True,
        channels : list = [64, 128, 256],
        ks : int = 3,
        st : int = 1,
        pd : int = None,
        use_attention : bool = False,
        sequence_length : int = None,
        num_heads : int = 3,
    ):
        super(UNet1D, self).__init__()
        
        if use_time_cat:
            # add categorical embeddings
            self.hour_encoder = torch.nn.Embedding(24, 4)
            self.day_encoder = torch.nn.Embedding(7, 4)
            input_channels += 8

        self.use_time_cat = use_time_cat
        self.num_classes = num_classes
        self.use_attention = use_attention
        self.input_channels = input_channels
        
        channels = [input_channels] + channels

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # Encoder
        self.encoder = nn.ModuleList()
        for i in range(len(channels) - 2):
            self.encoder.append(DoubleConv(channels[i], channels[i + 1], ks, st, pd))

        # Bottleneck
        self.bottleneck = DoubleConv(channels[i + 1], channels[i + 2], 1, st, pd)

        # Multi-Head Attention Layer
        if use_attention:
            attention_embed_dim = sequence_length // int(2**(len(channels)-2))
            self.attention = MultiHeadAttention(embed_dim=attention_embed_dim, num_heads=num_heads)

        self.decoder = nn.ModuleList()
        for i in range(len(channels) - 2):
            self.decoder.append(
                DoubleConv(channels[-i - 1] + channels[-i - 2], channels[-i - 2])
            )

        self.head = nn.Sequential(nn.Conv1d(channels[1], channels[1], 1, st, 0),
                                  
                                 )
        self.lin = nn.Sequential(nn.Linear(channels[1], num_classes),)

    def forward(self, x):
        if self.use_time_cat:
            #use categorical embeddings
            x = torch.concat([x[...,:-2],self.hour_encoder(x[...,-2].int()),self.day_encoder(x[...,-1].int())],dim=-1)
        
        encoder_features = []
        
        x = x.permute(0,2,1)
        
        # Downward pass
        for module in self.encoder:
            x = module(x)
            encoder_features.append(x)
            x = self.pool(x)

        # Horizontal pass
        x = self.bottleneck(x)
        
        if self.use_attention:
            # Apply the multi-head attention layer
            x = self.attention(x)

        # # Upward pass
        for idx, module in enumerate(self.decoder):
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            x = torch.concat([x, encoder_features[-1 - idx]], 1)
            x = module(x)

        x = self.head(x)
        x = x.permute(0,2,1)
        x = x.relu()
        x = self.lin(x)
        return x