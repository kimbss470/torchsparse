from torch import nn

from torchsparse import SparseTensor
from torchsparse.nn.utils import fapply

__all__ = ['Dropout']


class Dropout(nn.Dropout):

    def forward(self, input: SparseTensor) -> SparseTensor:
        return fapply(input, super().forward)
