__author__ = "Patrick Nicolas"

import torch
from torch import nn


class Lift(nn.Module):
    def __init__(self):
        super(Lift, self).__init__()

    def forward(self, input: torch.Tensor, sz: int) -> torch.Tensor:
        return input.view(input.size(0), -1)
