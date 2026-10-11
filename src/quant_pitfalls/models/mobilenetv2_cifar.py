"""MobileNetV2 (Sandler et al., 2018) for 32x32 inputs.

Same inverted-residual layout as torchvision's mobilenet_v2. The stem and the
first stride-2 stage use stride 1, so the last feature map is 4x4 instead of 1x1.
"""

from __future__ import annotations

from torch import nn


def mobilenet_v2_cifar(num_classes: int = 10, width_mult: float = 1.0) -> nn.Module:
    """Build torchvision MobileNetV2 with two early downsamplings removed."""
    from torchvision.models import mobilenet_v2

    if num_classes <= 0 or width_mult <= 0:
        raise ValueError("num_classes and width_mult must be positive")
    model = mobilenet_v2(num_classes=num_classes, width_mult=width_mult)
    model.features[0][0].stride = (1, 1)
    first_downsample = model.features[2]
    first_downsample.stride = 1
    first_downsample._is_cn = False
    first_downsample.conv[1][0].stride = (1, 1)
    return model
