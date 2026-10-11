"""Float model -> QAT model, and the staged QAT schedule.

prepare_qat replaces each Conv2d + BatchNorm2d pair with QuantConvBn2d and
inserts an activation FakeQuantize after every ReLU6 and at block outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from collections import OrderedDict

import torch
from torch import nn

from .layers.conv_bn import QuantConvBn2d
from .layers.linear import QuantLinear
from .quant.fake_quant import FakeQuantize
from .quant.apot import APoTSpec
from .quant.uniform import UniformSpec


@dataclass(frozen=True)
class QATSchedule:
    """Stage boundaries as fractions of the QAT epochs (pitfalls 5, 7, 8).

    Before weight_quant_from only the observers run. A one-shot schedule sets
    weight_quant_from = act_quant_from = 0.
    """

    weight_quant_from: float = 0.05
    act_quant_from: float = 0.10
    freeze_bn_from: float = 0.50
    freeze_observer_from: float = 0.75

    def __post_init__(self) -> None:
        if not all(0 <= value <= 1 for value in self.__dict__.values()):
            raise ValueError("Schedule boundaries must be fractions in [0, 1]")


@dataclass(frozen=True)
class QATStage:
    observe: bool
    weight_quant: bool
    act_quant: bool
    freeze_bn: bool


class QuantResidual(nn.Module):
    """Quantize the projection branch and the result of a residual addition."""

    def __init__(self, conv: nn.Module, use_res_connect: bool, output_quant: FakeQuantize) -> None:
        super().__init__()
        self.conv = conv
        self.use_res_connect = use_res_connect
        self.output_quant = output_quant

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch = self.conv(x)
        return self.output_quant(x + branch if self.use_res_connect else branch)


class QuantMobileNetV2(nn.Module):
    """Signed input, quantized features/pooling, 8-bit classifier weights."""

    def __init__(self, model: nn.Module, input_quant: FakeQuantize, pool_quant: FakeQuantize) -> None:
        super().__init__()
        self.input_quant = input_quant
        self.features = model.features
        self.pool_quant = pool_quant
        self.classifier = model.classifier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(self.input_quant(x))
        pooled = features.mean((2, 3))
        return self.classifier(self.pool_quant(pooled))


def prepare_qat(model: nn.Module, *, w_spec: UniformSpec | APoTSpec, a_spec: UniformSpec, per_channel: bool = True) -> nn.Module:
    """Copy and convert Conv-BN pairs, activations, residuals and Linear layers.

    MobileNetV2 uses 8-bit uniform weights at the stem and classifier. ReLU6
    outputs are unsigned; projections, residual outputs and inputs are signed.
    Addition and global averaging operate on dequantized values, followed by
    fake quantization. Biases and classifier logits remain floating point.
    """
    from torchvision.models.mobilenetv2 import InvertedResidual, MobileNetV2

    converted = deepcopy(model)
    first_conv = True

    def quantizer(spec, role, axis=None):
        fq = FakeQuantize(spec, axis=axis)
        fq.quant_role = role
        return fq

    def activation(signed=True):
        spec = UniformSpec(a_spec.bits, signed=signed, narrow_range=a_spec.narrow_range, rounding=a_spec.rounding)
        return quantizer(spec, "activation")

    def convert(module):
        nonlocal first_conv
        if isinstance(module, InvertedResidual):
            return QuantResidual(convert(module.conv), module.use_res_connect, activation())
        if isinstance(module, nn.Linear):
            return QuantLinear(module, quantizer(UniformSpec(8), "weight", 0 if per_channel else None))
        if isinstance(module, nn.Sequential):
            children = list(module.named_children())
            result = OrderedDict()
            index = 0
            while index < len(children):
                name, child = children[index]
                if isinstance(child, nn.Conv2d) and index + 1 < len(children) and isinstance(children[index + 1][1], nn.BatchNorm2d):
                    spec = UniformSpec(8) if first_conv else w_spec
                    first_conv = False
                    fq = quantizer(spec, "weight", 0 if per_channel else None)
                    result[name] = QuantConvBn2d.from_float(child, children[index + 1][1], weight_fq=fq)
                    following_relu = index + 2 < len(children) and isinstance(children[index + 2][1], nn.ReLU6)
                    if not following_relu:
                        result[name + "_output_quant"] = activation()
                    index += 2
                    continue
                result[name] = convert(child)
                if isinstance(child, nn.ReLU6):
                    result[name + "_output_quant"] = activation(signed=False)
                index += 1
            return nn.Sequential(result)
        for name, child in list(module.named_children()):
            setattr(module, name, convert(child))
        return module

    converted = convert(converted)
    if isinstance(model, MobileNetV2):
        result = QuantMobileNetV2(converted, activation(), activation(signed=False))
    else:
        result = nn.Sequential(OrderedDict(input_quant=activation(), model=converted))
    reference = next(model.parameters())
    result.to(device=reference.device, dtype=reference.dtype)
    result.train(model.training)
    return result


def stage_at(epoch: int, total_epochs: int, schedule: QATSchedule) -> QATStage:
    """Evaluate boundaries at epoch/total_epochs, including exact boundaries."""
    if total_epochs <= 0 or epoch < 0:
        raise ValueError("total_epochs must be positive and epoch nonnegative")
    progress = epoch / total_epochs
    return QATStage(progress < schedule.freeze_observer_from,
                    progress >= schedule.weight_quant_from,
                    progress >= schedule.act_quant_from,
                    progress >= schedule.freeze_bn_from)


def apply_stage(model: nn.Module, stage: QATStage) -> None:
    """Set observer, fake-quant and BN-freeze switches on every quantized module."""
    for module in model.modules():
        if isinstance(module, QuantConvBn2d):
            module.freeze_bn_stats(stage.freeze_bn)
        elif isinstance(module, FakeQuantize):
            module.enable_observer(stage.observe)
            module.enable_fake_quant(stage.weight_quant if getattr(module, "quant_role", "activation") == "weight" else stage.act_quant)
