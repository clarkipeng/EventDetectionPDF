import math

import torch
import torch.nn.functional as F
from torch import nn

try:
    from flash_attn import flash_attn_func
except ImportError:  # pragma: no cover - optional GPU dependency
    flash_attn_func = None


class FeatureProjector(nn.Module):
    def __init__(
        self,
        input_channels,
        hidden_size,
        cat_feats=0,
        cat_unique=24,
        categorical_enc_dim=4,
    ):
        super().__init__()
        self.cat_feats = cat_feats
        if cat_feats:
            self.cat_encoders = nn.ModuleList(
                [
                    nn.Embedding(cat_unique, categorical_enc_dim)
                    for _ in range(cat_feats)
                ]
            )
            input_channels += categorical_enc_dim * cat_feats
        self.fc_in = nn.Linear(input_channels, hidden_size)
        self.ln = nn.LayerNorm(hidden_size)

    def forward(self, x):
        if self.cat_feats:
            x = torch.concat(
                [x[..., : -self.cat_feats]]
                + [
                    self.cat_encoders[i](x[..., -(i + 1)].int())
                    for i in range(self.cat_feats)
                ],
                dim=-1,
            )
        return F.relu(self.ln(self.fc_in(x)))


class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, length, device, dtype):
        position = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.hidden_size, 2, device=device, dtype=dtype)
            * (-math.log(10000.0) / self.hidden_size)
        )
        encoding = torch.zeros(length, self.hidden_size, device=device, dtype=dtype)
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term[: encoding[:, 1::2].shape[1]])
        return encoding.unsqueeze(0)


class FlashCausalSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.0, causal=True):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.dropout = dropout
        self.causal = causal
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        batch, length, _ = x.shape
        qkv = self.qkv(x).view(
            batch,
            length,
            3,
            self.num_heads,
            self.head_dim,
        )
        q, k, v = qkv.unbind(dim=2)
        dropout_p = self.dropout if self.training else 0.0

        if (
            flash_attn_func is not None
            and x.is_cuda
            and x.dtype in (torch.float16, torch.bfloat16)
        ):
            attn = flash_attn_func(q, k, v, dropout_p=dropout_p, causal=self.causal)
        else:
            attn = F.scaled_dot_product_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                dropout_p=dropout_p,
                is_causal=self.causal,
            ).transpose(1, 2)

        return self.proj(attn.reshape(batch, length, self.hidden_size))


class CausalTransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4, dropout=0.1, causal=True):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size)
        self.attn = FlashCausalSelfAttention(
            hidden_size,
            num_heads,
            dropout=dropout,
            causal=causal,
        )
        self.ln2 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * mlp_ratio, hidden_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.attn(self.ln1(x)))
        x = x + self.dropout(self.mlp(self.ln2(x)))
        return x


class CausalTransformer(nn.Module):
    def __init__(
        self,
        input_channels,
        cat_feats=0,
        cat_unique=24,
        categorical_enc_dim=4,
        num_classes=2,
        hidden_size=64,
        num_heads=4,
        n_layers=4,
        dropout=0.1,
        causal=True,
    ):
        super().__init__()
        self.causal = causal
        self.project = FeatureProjector(
            input_channels=input_channels,
            hidden_size=hidden_size,
            cat_feats=cat_feats,
            cat_unique=cat_unique,
            categorical_enc_dim=categorical_enc_dim,
        )
        self.position = SinusoidalPositionEncoding(hidden_size)
        self.blocks = nn.ModuleList(
            [
                CausalTransformerBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    dropout=dropout,
                    causal=causal,
                )
                for _ in range(n_layers)
            ]
        )
        self.ln = nn.LayerNorm(hidden_size)
        self.fc_out = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.project(x)
        x = x + self.position(x.shape[1], x.device, x.dtype)
        for block in self.blocks:
            x = block(x)
        return self.fc_out(self.ln(x))
