import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


def get_loss(objective):
    if objective[:3] == "seg":
        return nn.BCEWithLogitsLoss(reduction="none")
    return nn.MSELoss(reduction="none")
