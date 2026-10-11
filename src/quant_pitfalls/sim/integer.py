"""Integer-only inference (Jacob et al., 2018).

The real multiplier M = s_w * s_in / s_out of each layer is applied as
M0 * 2^-n with an integer M0, so a layer runs on integer multiply, add and
shift only. Bit-exactness against the fake-quant path breaks on tie rounding
(half-to-even vs half-up), on the M0 approximation, and on accumulator overflow.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.utils import _pair

from ..quant.uniform import UniformSpec, quantize
from .accumulator import AccumulatorSpec, _integer_tensor, _ordered_saturation, accumulate, apply_overflow, required_bits


def quantize_multiplier(m: float, bits: int = 31) -> tuple[int, int]:
    """Return normalized M0, n for positive finite m ~= M0 * 2**-n.

    frexp normalizes m to [0.5, 1). The significand uses half-up rounding;
    a rounding carry renormalizes M0. n may be negative for large m.
    """
    if not math.isfinite(m) or m <= 0:
        raise ValueError("Multiplier must be finite and positive")
    if not isinstance(bits, int) or not 2 <= bits <= 31:
        raise ValueError("Multiplier bits must be in [2, 31]")
    fraction, exponent = math.frexp(m)
    m0 = math.floor(fraction * (1 << bits) + 0.5)
    if m0 == 1 << bits:
        m0 >>= 1
        exponent += 1
    return m0, bits - exponent


def _multiply_shift(acc: torch.Tensor, m0: int, n: int, *, rounding: str = "nearest") -> torch.Tensor:
    values = _integer_tensor(acc)
    if not isinstance(m0, int) or not 0 < m0 < 1 << 31 or not isinstance(n, int):
        raise ValueError("M0 must be a positive 31-bit integer and n an integer")
    if values.numel():
        if int(values.min()) < -((1 << 63) // m0) or int(values.max()) > ((1 << 63) - 1) // m0:
            raise OverflowError("acc * M0 exceeds int64")
    product = values * m0
    if n <= 0:
        if -n >= 63 or (product.numel() and max(abs(int(product.min())), abs(int(product.max()))) > ((1 << 63) - 1) >> -n):
            raise OverflowError("Left-shifted product exceeds int64")
        return product << -n
    if n >= 64:
        return torch.where(product < 0, -torch.ones_like(product), torch.zeros_like(product)) if rounding == "floor" else torch.zeros_like(product)
    # Quotient/remainder equals (product + half) >> n without overflowing
    # when the rounding offset itself would exceed the int64 headroom.
    quotient = product >> n
    if rounding == "nearest":
        remainder = torch.bitwise_and(product, (1 << n) - 1)
        quotient = quotient + (remainder >= (1 << (n - 1))).to(torch.int64)
    return quotient


def requantize(acc: torch.Tensor, m0: int, n: int, out_spec: UniformSpec) -> torch.Tensor:
    """Apply M0 * 2**-n, half-up (or floor), then clamp to output codes.

    Outputs are int64. Negative ties round toward positive infinity, just
    like floor(x + 0.5). The checked multiply rejects int64 overflow.
    """
    if n < 0:
        # Clamp before a potentially huge left shift, preserving the final
        # output clamp even when the unbounded shifted result cannot fit.
        values = _multiply_shift(acc, m0, 0)
        shift = -n
        if shift >= 63:
            return torch.where(values > 0, out_spec.qmax, torch.where(values < 0, out_spec.qmin, 0))
        lower = -((-out_spec.qmin) >> shift)
        upper = out_spec.qmax >> shift
        inside = values.clamp(lower, upper) << shift
        return torch.where(values < lower, out_spec.qmin, torch.where(values > upper, out_spec.qmax, inside))
    return _multiply_shift(acc, m0, n, rounding=out_spec.rounding).clamp(out_spec.qmin, out_spec.qmax)


def int_conv2d(
    x_q: torch.Tensor,
    w_q: torch.Tensor,
    bias_q: torch.Tensor | None = None,
    *,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    groups: int = 1,
    acc: AccumulatorSpec = AccumulatorSpec(),
    dilation: int | tuple[int, int] = 1,
    stats: dict | None = None,
) -> torch.Tensor:
    """Convolve codes, returning int64 accumulators with fixed-width overflow.

    The exact no-overflow convolution uses float64 only when the absolute
    sum bound is below 2**53. Otherwise unfold supplies integer products.
    Saturation follows input-channel, kernel-row, kernel-column order;
    bias is added last and is subject to the same overflow rule. Products
    are chunked to bound memory. groups includes depthwise convolution.

    Optional stats counts outputs whose *any* accumulation prefix exceeds
    the range, rather than just counting final sums outside the range.
    """
    x, weight = _integer_tensor(x_q), _integer_tensor(w_q)
    if x.ndim != 4 or weight.ndim != 4 or groups <= 0:
        raise ValueError("Expected NCHW input, OIHW weights and positive groups")
    if x.shape[1] != weight.shape[1] * groups or weight.shape[0] % groups:
        raise ValueError("Convolution channels do not match groups")
    if x.device != weight.device:
        raise ValueError("Input and weight devices must match")
    bias = None if bias_q is None else _integer_tensor(bias_q)
    if bias is not None and (bias.shape != (weight.shape[0],) or bias.device != x.device):
        raise ValueError("Bias must match output channels and device")
    fan_in = weight[0].numel()
    x_max = max(abs(int(x.min())), abs(int(x.max()))) if x.numel() else 0
    w_max = max(abs(int(weight.min())), abs(int(weight.max())))
    b_max = max(abs(int(bias.min())), abs(int(bias.max()))) if bias is not None else 0
    product_bound = fan_in * x_max * w_max
    bound = product_bound + b_max
    if bound >= 1 << 63 or max(x_max, w_max) >= 1 << 53:
        raise OverflowError("Convolution intermediates exceed exact representation")
    safe = product_bound <= acc.maximum
    exact = None
    if product_bound < 1 << 53:
        exact = F.conv2d(x.double(), weight.double(), None,
                         stride=stride, padding=padding, dilation=dilation, groups=groups).to(torch.int64)
        product_maximum = int(exact.abs().max()) if exact.numel() else 0
        if bias is not None:
            exact = exact + bias.reshape(1, -1, 1, 1)
        if not safe:
            # A data-dependent absolute-sum bound can prove every product
            # prefix safe even when the global code bound is pessimistic.
            absolute = F.conv2d(x.abs().double(), weight.abs().double(), stride=stride,
                                padding=padding, dilation=dilation, groups=groups)
            safe = int(absolute.max()) <= acc.maximum
        if safe or (acc.overflow == "wrap" and stats is None):
            if stats is not None:
                stats.update(outputs=exact.numel(), overflow_outputs=int(((exact < acc.minimum) | (exact > acc.maximum)).sum()),
                             max_abs_sum=int(exact.abs().max()) if exact.numel() else 0,
                             max_abs_product_sum=product_maximum)
            return apply_overflow(exact, acc)
    patches = F.unfold(x.double(), weight.shape[-2:], dilation=dilation, padding=padding, stride=stride).to(torch.int64)
    patches = patches.reshape(x.shape[0], groups, fan_in, -1)
    sh, sw = _pair(stride)
    ph, pw = _pair(padding)
    dh, dw = _pair(dilation)
    height = (x.shape[2] + 2 * ph - dh * (weight.shape[2] - 1) - 1) // sh + 1
    width = (x.shape[3] + 2 * pw - dw * (weight.shape[3] - 1) - 1) // sw + 1
    output = torch.empty((x.shape[0], weight.shape[0], height * width), dtype=torch.int64, device=x.device)
    overflow_outputs, maximum, product_maximum = 0, 0, 0
    out_per_group = weight.shape[0] // groups
    out_chunk = max(8, 2_000_000 // max(1, x.shape[0] * fan_in * min(height * width, 32)))
    for start in range(0, weight.shape[0], out_chunk):
        stop = min(start + out_chunk, weight.shape[0])
        group_index = torch.arange(start, stop, device=x.device) // out_per_group
        filters = weight[start:stop].reshape(stop-start, fan_in)
        chunk = max(1, 2_000_000 // max(1, x.shape[0] * (stop-start) * fan_in))
        for position in range(0, height * width, chunk):
            end = min(position + chunk, height * width)
            products = patches[:, group_index, :, position:end].transpose(-1, -2) * filters[None, :, None, :]
            if bias is not None:
                products = torch.cat((products, bias[start:stop][None, :, None, None].expand(*products.shape[:-1], 1)), -1)
            if stats is not None:
                prefix = products.cumsum(-1)
                overflow_outputs += int(((prefix < acc.minimum) | (prefix > acc.maximum)).any(-1).sum())
                maximum = max(maximum, int(prefix[..., -1].abs().max()))
                product_maximum = max(product_maximum, int(prefix[..., -2 if bias is not None else -1].abs().max()))
            # The convolution bound already checked all exact sums. Avoid
            # repeating host synchronizations for each GPU product chunk.
            if acc.overflow == "wrap":
                result = apply_overflow(products.sum(-1), acc)
            elif bound + (1 << (acc.bits - 1)) < 1 << 63:
                result = _ordered_saturation(products, acc)
            else:
                result = accumulate(products, acc)
            output[:, start:stop, position:end] = result
    if stats is not None:
        stats.update(outputs=output.numel(), overflow_outputs=overflow_outputs, max_abs_sum=maximum,
                     max_abs_product_sum=product_maximum)
    return output.reshape(x.shape[0], weight.shape[0], height, width)


@dataclass
class _Value:
    codes: torch.Tensor
    scale: torch.Tensor
    spec: UniformSpec | None = None


def _broadcast_scale(scale: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
    if scale.numel() == 1:
        return scale.reshape(())
    return scale.reshape(1, -1, *([1] * (codes.ndim - 2)))


def _rescale(codes: torch.Tensor, ratio: torch.Tensor, spec: UniformSpec | None = None) -> torch.Tensor:
    """Apply scalar or per-output-channel normalized multipliers."""
    ratios = ratio.flatten().tolist()
    transform = (lambda values, m0, n: requantize(values, m0, n, spec)) if spec is not None else _multiply_shift
    if len(ratios) == 1:
        return transform(codes, *quantize_multiplier(ratios[0]))
    if len(ratios) != codes.shape[1]:
        raise ValueError("Scale vector must match output channels")
    parameters = [quantize_multiplier(r) for r in ratios]
    if any(n <= 0 for _, n in parameters):
        return torch.stack([transform(codes[:, index], *parameters[index]) for index in range(len(ratios))], 1)
    shape = (1, len(ratios), *([1] * (codes.ndim - 2)))
    multipliers = torch.tensor([m0 for m0, _ in parameters], device=codes.device, dtype=torch.int64).reshape(shape)
    shifts = torch.tensor([n for _, n in parameters], device=codes.device, dtype=torch.int64).reshape(shape)
    if codes.numel() and max(abs(int(codes.min())), abs(int(codes.max()))) > ((1 << 63) - 1) // max(m0 for m0, _ in parameters):
        raise OverflowError("Per-channel acc * M0 exceeds int64")
    product = codes * multipliers
    safe_shifts = shifts.clamp_max(63)
    result = product >> safe_shifts
    rounding = "nearest" if spec is None else spec.rounding
    if rounding == "nearest":
        mask = (torch.ones_like(safe_shifts) << safe_shifts) - 1
        remainder = torch.bitwise_and(product, mask)
        result = result + (remainder >= (torch.ones_like(safe_shifts) << (safe_shifts - 1))).to(torch.int64)
        result = torch.where(shifts >= 64, 0, result)
    else:
        result = torch.where(shifts >= 64, torch.where(product < 0, -1, 0), result)
    return result if spec is None else result.clamp(spec.qmin, spec.qmax)


class IntegerModel:
    """Prepared integer simulator for a frozen, eval-mode uniform QAT model.

    Float parameters/scales are converted once at preparation. Bias uses
    half-up integer codes in s_input*s_weight units and is added last. ReLU6
    clips accumulator codes; residual branches align on the output scale
    with 16 fractional guard bits before one final requantization. Pooling
    sums int64 codes and folds division by H*W into its multiplier. The
    classifier uses an integer dot product with per-channel logit scales;
    only input encoding and final logit dequantization use real values.

    The numerical conv backend uses exact float64 sums below 2**53; this is
    a simulator, not an integer hardware kernel. reference=True substitutes
    independent float64 convolutions and fixed-point fake quantization for
    layer-by-layer code comparisons at a non-overflowing 32-bit width.
    Prepare after moving the source model to the desired device.
    """

    def __init__(self, model: nn.Module, *, acc: AccumulatorSpec = AccumulatorSpec(),
                 reference: bool = False, capture: bool = False, collect_stats: bool = False) -> None:
        from ..layers.conv_bn import QuantConvBn2d
        from ..layers.linear import QuantLinear
        from ..quant.fake_quant import FakeQuantize

        if model.training:
            raise ValueError("Integer inference requires model.eval()")
        self.model, self.acc, self.reference = model, acc, reference
        self.capture, self.collect_stats = capture, collect_stats
        self.trace, self.stats, self.layers, self.quantizers = {}, {}, {}, {}
        for name, module in model.named_modules():
            if isinstance(module, FakeQuantize):
                if not isinstance(module.spec, UniformSpec):
                    raise ValueError("Integer inference supports uniform quantization only")
                if bool(module.observer_enabled) or not bool(module.fake_quant_enabled):
                    raise ValueError("Observers must be frozen and fake quantization enabled")
                if not torch.isfinite(module.alpha).all() or not (module.alpha > 0).all():
                    raise ValueError("Quantizer clips must be finite and positive")
                # Preserve the actual FP32/FP64 step used during QAT.
                self.quantizers[name] = ((module.alpha / module.spec.qmax).double().detach().clone(), module.spec)
            if isinstance(module, (QuantConvBn2d, QuantLinear)):
                if isinstance(module, QuantConvBn2d):
                    if module.weight_quant_order != "fold" or module.deployment_weight_fq is not None:
                        raise ValueError("Integer inference requires fold-before-quantization weights")
                    weight, bias = module.folded_weight_and_bias()
                else:
                    weight, bias = module.linear.weight, module.linear.bias
                fq = module.weight_fq
                if fq is None or not isinstance(fq.spec, UniformSpec) or fq.axis not in (None, 0):
                    raise ValueError("Weights require a uniform scalar or output-channel quantizer")
                codes = quantize(weight.detach(), fq.alpha, fq.spec, axis=fq.axis).to(torch.int64)
                scale = (fq.alpha / fq.spec.qmax).double().detach().clone()
                self.layers[name] = (codes, scale, None if bias is None else bias.detach().double().clone())

    def _record(self, name: str, value: _Value) -> _Value:
        if self.capture:
            self.trace[name] = value.codes.detach().clone()
        return value

    def _quantize(self, name: str, value: _Value) -> _Value:
        scale, spec = self.quantizers[name]
        return self._record(name, _Value(_rescale(value.codes, value.scale / scale, spec), scale, spec))

    def _affine(self, name: str, module: nn.Module, value: _Value) -> _Value:
        from ..layers.linear import QuantLinear

        weight, weight_scale, bias = self.layers[name]
        if value.scale.numel() != 1:
            raise ValueError("A convolution input must have a per-tensor activation scale")
        scale = value.scale * weight_scale
        if bias is None:
            bias_q = None
        else:
            bias_codes = torch.floor(bias / scale + 0.5)
            if not torch.isfinite(bias_codes).all() or not (bias_codes.abs() < 1 << 63).all():
                raise OverflowError("Bias encoding exceeds int64")
            bias_q = bias_codes.to(torch.int64)
        linear = isinstance(module, QuantLinear)
        x_q = value.codes[:, :, None, None] if linear else value.codes
        w_q = weight[:, :, None, None] if linear else weight
        conv = None if linear else module.conv
        options = dict(stride=1 if linear else conv.stride, padding=0 if linear else conv.padding,
                       groups=1 if linear else conv.groups, dilation=1 if linear else conv.dilation)
        layer_stats = {} if self.collect_stats else None
        if self.reference:
            fan_in = w_q[0].numel()
            bound = fan_in * int(x_q.abs().max()) * int(w_q.abs().max())
            if bound > self.acc.maximum or bound >= 1 << 53:
                raise ValueError("Fixed-point fake-quant reference requires a non-overflowing exact sum")
            output = F.conv2d(x_q.double(), w_q.double(), **options).to(torch.int64)
            if bias_q is not None:
                output = output + bias_q.reshape(1, -1, 1, 1)
            output = apply_overflow(output, self.acc)
        else:
            output = int_conv2d(x_q, w_q, bias_q, acc=self.acc, stats=layer_stats, **options)
        if layer_stats is not None and not self.reference:
            input_spec_max = max(abs(value.spec.qmin), value.spec.qmax) if value.spec is not None else int(value.codes.abs().max())
            weight_spec_max = max(abs(module.weight_fq.spec.qmin), module.weight_fq.spec.qmax)
            layer_stats.update(required_bits=required_bits(weight_spec_max, input_spec_max, w_q[0].numel()),
                               fan_in=w_q[0].numel(), bias_max=int(bias_q.abs().max()) if bias_q is not None else 0,
                               required_bits_with_bias=(weight_spec_max*input_spec_max*w_q[0].numel() + (int(bias_q.abs().max()) if bias_q is not None else 0)).bit_length() + 1)
            self.stats[name] = layer_stats
        if linear:
            output = output[:, :, 0, 0]
        return self._record(name, _Value(output, scale))

    def _walk(self, module: nn.Module, name: str, value: _Value) -> _Value:
        from ..convert import QuantMobileNetV2, QuantResidual
        from ..layers.conv_bn import QuantConvBn2d
        from ..layers.linear import QuantLinear
        from ..quant.fake_quant import FakeQuantize

        def child_name(child):
            return f"{name}.{child}" if name else child

        if isinstance(module, QuantMobileNetV2):
            value = self._walk(module.features, child_name("features"), value)
            count = value.codes.shape[2] * value.codes.shape[3]
            value = _Value(value.codes.sum((2, 3)), value.scale / count)
            value = self._quantize(child_name("pool_quant"), value)
            return self._walk(module.classifier, child_name("classifier"), value)
        if isinstance(module, QuantResidual):
            branch = self._walk(module.conv, child_name("conv"), value)
            if module.use_res_connect:
                scale, _ = self.quantizers[child_name("output_quant")]
                guard_scale = scale / (1 << 16)
                codes = _rescale(value.codes, value.scale / guard_scale) + _rescale(branch.codes, branch.scale / guard_scale)
                branch = _Value(codes, guard_scale)
            return self._quantize(child_name("output_quant"), branch)
        if isinstance(module, nn.Sequential):
            for child, operation in module.named_children():
                value = self._walk(operation, child_name(child), value)
            return value
        if isinstance(module, (QuantConvBn2d, QuantLinear)):
            return self._affine(name, module, value)
        if isinstance(module, FakeQuantize):
            return self._quantize(name, value)
        if isinstance(module, nn.ReLU6):
            upper = torch.floor(6 / _broadcast_scale(value.scale, value.codes) + 0.5).clamp_max(1 << 61).to(torch.int64)
            return _Value(torch.minimum(value.codes.clamp_min(0), upper), value.scale)
        if isinstance(module, (nn.Dropout, nn.Identity)) and not module.training:
            return value
        raise TypeError(f"Unsupported integer inference module: {type(module).__name__}")

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        from ..convert import QuantMobileNetV2
        from ..quant.fake_quant import FakeQuantize

        self.trace, self.stats = {}, {}
        if isinstance(self.model, QuantMobileNetV2):
            fq, name = self.model.input_quant, "input_quant"
        elif isinstance(self.model, nn.Sequential) and isinstance(self.model[0], FakeQuantize):
            name, fq = next(iter(self.model.named_children()))
        else:
            raise TypeError("Expected a model produced by prepare_qat")
        scale, _ = self.quantizers[name]
        value = self._record(name, _Value(quantize(x, fq.alpha, fq.spec, axis=fq.axis).to(torch.int64), scale, fq.spec))
        if isinstance(self.model, QuantMobileNetV2):
            value = self._walk(self.model, "", value)
        else:
            for child, module in list(self.model.named_children())[1:]:
                value = self._walk(module, child, value)
        return (value.codes.double() * _broadcast_scale(value.scale, value.codes)).to(x.dtype)


def run_integer(model: nn.Module, x: torch.Tensor, *, acc: AccumulatorSpec = AccumulatorSpec()) -> torch.Tensor:
    """Run frozen uniform QAT using integer codes and return dequantized logits.

    See IntegerModel for bias rounding, residual guard bits, pooling, final
    per-channel logit scales, and the exact float64 convolution backend.
    Reuse IntegerModel directly when evaluating multiple batches.
    """
    return IntegerModel(model, acc=acc)(x)
