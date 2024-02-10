from typing import List

import torch

import torchsparse
from torchsparse.tensor import SparseTensor

__all__ = ['cat', 'add']


def cat(inputs: List[SparseTensor]) -> SparseTensor:
    feats = torch.cat([input.feats for input in inputs], dim=1)
    output = SparseTensor(coords=inputs[0].coords,
                          feats=feats,
                          stride=inputs[0].stride)
    output.cmaps = inputs[0].cmaps
    output.kmaps = inputs[0].kmaps
    return output

def add(input1:SparseTensor, input2:SparseTensor) -> SparseTensor:
    feats = torch.add(input1.feats, input2.feats)
    output = SparseTensor(coords=input1.coords,
                          feats=feats,
                          stride=input1.stride)
    output.cmaps = input1.cmaps
    output.kmaps = input1.kmaps
    return output
