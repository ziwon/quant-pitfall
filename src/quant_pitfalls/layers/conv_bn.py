"""Conv2d + BatchNorm2d for QAT, with BN folded before weight quantization.

    W_fold = gamma / sqrt(var + eps) * W
    b_fold = beta + gamma * (b - mean) / sqrt(var + eps)

Pitfall 4: quantize W_fold. Folding after quantization scales every output
channel by a different factor and can move weights off the deployment grid.
Pitfall 7: freeze_bn_stats() switches folding to the running statistics and
stops updating them, independent of module.train() / module.eval().
"""

from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn

from ..quant.fake_quant import FakeQuantize


class QuantConvBn2d(nn.Module):
    """Fold before quantization, with batch correction during training.

    Following Krishnamoorthi (2018), section 3.2.2, equations 22-31:
      W_R = gamma * W / sigma_R
      y_train = conv(x, Q(W_R)) * sigma_R/sigma_B + beta - gamma*mean_B/sigma_B
    mean_B is measured on the convolution without bias. Running statistics are
    updated by PyTorch BatchNorm on the float convolution including its bias.
    After freezing, y = conv(x, Q(W_R), beta + gamma*(bias-mean_R)/sigma_R).
    This two-convolution path avoids division by gamma, including gamma=0.

    Reference: https://arxiv.org/abs/1806.08342
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        *,
        weight_fq: FakeQuantize | None = None,
        eps: float = 1e-5,
        momentum: float = 0.1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum)
        self.weight_fq = weight_fq
        self.weight_quant_order = "fold"
        self.deployment_weight_fq = None
        self.register_buffer("_bn_frozen", torch.tensor(False))

    def train(self, mode: bool = True) -> QuantConvBn2d:
        super().train(mode)
        self.bn.train(self.training and not bool(self._bn_frozen))
        return self

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        self.bn.train(self.training and not bool(self._bn_frozen))

    @classmethod
    def from_float(cls, conv: nn.Conv2d, bn: nn.BatchNorm2d, *, weight_fq: FakeQuantize | None = None) -> QuantConvBn2d:
        """Copy weights and BN state from a trained float pair."""
        if conv.out_channels != bn.num_features or not bn.track_running_stats:
            raise ValueError("Conv and BN must have matching channels and tracked running statistics")
        layer = cls(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding, conv.groups,
                    weight_fq=weight_fq, eps=bn.eps, momentum=bn.momentum)
        layer.conv = deepcopy(conv)
        layer.bn = deepcopy(bn)
        layer.to(device=conv.weight.device, dtype=conv.weight.dtype)
        layer.train(bn.training)
        return layer

    def _affine(self) -> tuple[torch.Tensor, torch.Tensor]:
        reference = self.conv.weight[:, 0, 0, 0]
        gamma = self.bn.weight if self.bn.affine else torch.ones_like(reference)
        beta = self.bn.bias if self.bn.affine else torch.zeros_like(reference)
        return gamma, beta

    def _fold(self, mean: torch.Tensor, var: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gamma, beta = self._affine()
        factor = gamma * torch.rsqrt(var + self.bn.eps)
        weight = (self.conv.weight * factor.reshape(-1, 1, 1, 1)).to(self.conv.weight.dtype)
        bias = self.conv.bias if self.conv.bias is not None else torch.zeros_like(mean)
        return weight, (beta + (bias - mean) * factor).to(self.conv.weight.dtype)

    def folded_weight_and_bias(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return float deployment weights and bias using running statistics in every mode."""
        return self._fold(self.bn.running_mean, self.bn.running_var)

    def freeze_bn_stats(self, frozen: bool = True) -> None:
        self._bn_frozen.fill_(frozen)
        self.bn.train(self.training and not frozen)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or bool(self._bn_frozen):
            weight, bias = self._fold(self.bn.running_mean, self.bn.running_var)
            if self.weight_fq is not None:
                if self.weight_quant_order == "raw":
                    gamma, _ = self._affine()
                    factor = gamma * torch.rsqrt(self.bn.running_var + self.bn.eps)
                    weight = self.weight_fq(self.conv.weight) * factor.reshape(-1, 1, 1, 1)
                else:
                    weight = self.weight_fq(weight)
            if self.deployment_weight_fq is not None:
                weight = self.deployment_weight_fq(weight)
            return self.conv._conv_forward(x, weight, bias)

        raw = self.conv._conv_forward(x, self.conv.weight, None)
        with torch.no_grad():
            biased = raw if self.conv.bias is None else raw + self.conv.bias.reshape(1, -1, 1, 1)
            self.bn(biased)
        values = raw.float() if raw.dtype in (torch.float16, torch.bfloat16) else raw
        mean = values.mean((0, 2, 3))
        var = values.var((0, 2, 3), unbiased=False)

        gamma, beta = self._affine()
        running_std = torch.sqrt(self.bn.running_var.detach().clone() + self.bn.eps)
        batch_std = torch.sqrt(var + self.bn.eps)
        weight = (self.conv.weight * (gamma / running_std).reshape(-1, 1, 1, 1)).to(self.conv.weight.dtype)
        if self.weight_fq is not None:
            if self.weight_quant_order == "raw":
                weight = self.weight_fq(self.conv.weight) * (gamma / running_std).reshape(-1, 1, 1, 1)
            else:
                weight = self.weight_fq(weight)
        result = self.conv._conv_forward(x, weight, None)
        result = result * (running_std / batch_std).reshape(1, -1, 1, 1)
        result = result + (beta - gamma * mean / batch_std).reshape(1, -1, 1, 1)
        if self.conv.bias is not None:
            result = result + (self.conv.bias - self.conv.bias).reshape(1, -1, 1, 1)
        return result.to(raw.dtype)
