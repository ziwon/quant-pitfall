"""Linear weight fake quantization; classifier logits remain floating point."""

from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from ..quant.fake_quant import FakeQuantize


class QuantLinear(nn.Module):
    def __init__(self, linear: nn.Linear, weight_fq: FakeQuantize) -> None:
        super().__init__()
        self.linear = deepcopy(linear)
        self.weight_fq = weight_fq.to(linear.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight_fq(self.linear.weight), self.linear.bias)
