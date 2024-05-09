import torch
import torch.nn.functional as F
from torch import Tensor, nn

class BiRNN(nn.Module):
    def __init__(self, 
                 hidden_size : int, 
                 rnn_unit = nn.GRU, # or nn.LSTM
                 n_layers : int = 1, 
                 bidir : bool = True):
        super(BiRNN, self).__init__()

        self.hidden_size = hidden_size
        self.n_layers = n_layers
        
        self.rnn = rnn_unit(
            hidden_size,
            hidden_size,
            n_layers,
            batch_first=True,
            bidirectional=bidir,
        )
        
        dir_factor = 2 if bidir else 1
        
        self.fc1 = nn.Linear(
            hidden_size * dir_factor, hidden_size * dir_factor * 2
        )
        self.ln1 = nn.LayerNorm(hidden_size * dir_factor * 2)
        self.fc2 = nn.Linear(hidden_size * dir_factor * 2, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)

    def forward(self, x, h=None):
        res, new_h = self.rnn(x, h)

        res = self.fc1(res)
        res = self.ln1(res)
        res = nn.functional.relu(res)

        res = self.fc2(res)
        res = self.ln2(res)
        res = nn.functional.relu(res)

        # skip connection
        res = res + x

        return res

class MultiBiRNN(nn.Module):
    def __init__(self, 
                 input_channels : int = 2, 
                 use_time_cat : bool = True,
                 num_classes   : int = 2, 
                 hidden_size : int = 32, 
                 rnn_unit = nn.GRU, # or nn.LSTM
                 n_layers : int = 1, 
                 bidir : bool = True):
        super(MultiBiRNN, self).__init__()
        
        if use_time_cat:
            # add categorical embeddings
            self.hour_encoder = torch.nn.Embedding(24, 4)
            self.day_encoder = torch.nn.Embedding(7, 4)
            input_channels += 8

        self.use_time_cat = use_time_cat
        self.input_channels = input_channels
        self.hidden_size = hidden_size
        self.num_classes = num_classes 
        self.n_layers = n_layers

        self.fc_in = nn.Linear(self.input_channels, hidden_size)
        self.ln = nn.LayerNorm(hidden_size)
        self.res_bigrus = nn.ModuleList(
            [
                BiRNN(hidden_size, rnn_unit, n_layers=1, bidir=bidir)
                for _ in range(n_layers)
            ]
        )
        self.fc_out = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        if self.use_time_cat:
            #use categorical embeddings
            x = torch.concat([x[...,:-2],self.hour_encoder(x[...,-2].int()),self.day_encoder(x[...,-1].int())],dim=-1)
        
        x = self.fc_in(x)
        
        x = self.ln(x)
        x = nn.functional.relu(x)

        for i, res_bigru in enumerate(self.res_bigrus):
            x = res_bigru(x)
        
        x = self.fc_out(x)
        return x