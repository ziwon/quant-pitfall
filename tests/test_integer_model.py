"""Independent scalar references for rounding, overflow and integer deployment."""

import math

import pytest
import torch
from torch import nn

from quant_pitfalls.convert import QATStage, apply_stage, prepare_qat
from quant_pitfalls.models.mobilenetv2_cifar import mobilenet_v2_cifar
from quant_pitfalls.quant.fake_quant import FakeQuantize
from quant_pitfalls.quant.uniform import UniformSpec
from quant_pitfalls.sim.accumulator import AccumulatorSpec, accumulate, required_bits
from quant_pitfalls.sim.integer import IntegerModel, int_conv2d, quantize_multiplier, requantize, run_integer


DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))]


def scalar_sum(values, bits, mode):
    lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    total = 0
    for value in values:
        total += int(value)
        total = max(lo, min(hi, total)) if mode == "saturate" else (total-lo) % (1 << bits) + lo
    return total


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("mode", ["saturate", "wrap"])
def test_accumulate_matches_ordered_scalar_reference(device, mode):
    generator = torch.Generator().manual_seed(912)
    values = torch.randint(-300, 301, (3, 19, 5), generator=generator)
    expected = torch.tensor([[scalar_sum(values[i, :, j], 8, mode) for j in range(5)] for i in range(3)])
    result = accumulate(values.to(device), AccumulatorSpec(8, mode), dim=1)
    assert torch.equal(result.cpu(), expected)
    assert result.dtype == torch.int64


def test_accumulator_empty_dimension_and_exact_power_bound():
    assert required_bits(0, 255, 9) == 1
    assert required_bits(1, 1, 2**52) == 54
    assert accumulate(torch.empty(2, 0, dtype=torch.int64), AccumulatorSpec()).tolist() == [0, 0]
    with pytest.raises(OverflowError):
        accumulate(torch.tensor([2**62, 2**62]), AccumulatorSpec())
    with pytest.raises(TypeError):
        accumulate(torch.tensor([1.0]), AccumulatorSpec())
    with pytest.raises(ValueError):
        AccumulatorSpec(0)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("length", [2, 3, 17, 129])
def test_parallel_clipping_composition_preserves_sequential_saturation(device, length):
    generator = torch.Generator().manual_seed(711)
    values = torch.randint(-400, 401, (11, length), generator=generator)
    expected = torch.tensor([scalar_sum(row, 8, "saturate") for row in values])
    actual = accumulate(values.to(device), AccumulatorSpec(8), parallel=True)
    assert torch.equal(actual.cpu(), expected)


@pytest.mark.parametrize("m", [2**-200, 0.5, 1.0, 2.0, 16.0, 2**50, math.nextafter(1.0, 0.0)])
def test_multiplier_handles_exponents_and_normalization_carry(m):
    m0, n = quantize_multiplier(m)
    assert 2**30 <= m0 < 2**31
    assert math.ldexp(m0, -n) == pytest.approx(m, rel=2**-30)


@pytest.mark.parametrize("m", [0, -1, float("nan"), float("inf")])
def test_multiplier_rejects_invalid_values(m):
    with pytest.raises(ValueError):
        quantize_multiplier(m)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("n", [-3, 0, 1, 31, 63, 64, 100])
def test_requantization_matches_python_unbounded_integer_reference(device, n):
    codes = [-2**32, -3, -1, 0, 1, 3, 2**32]
    m0 = 2**30
    spec = UniformSpec(8, narrow_range=False)
    expected = []
    for value in codes:
        product = value * m0
        result = (product + (1 << (n - 1))) >> n if n > 0 else product << -n
        expected.append(max(spec.qmin, min(spec.qmax, result)))
    assert requantize(torch.tensor(codes, device=device), m0, n, spec).cpu().tolist() == expected
    with pytest.raises(OverflowError):
        requantize(torch.tensor([2**34], device=device), m0, n, spec)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("mode", ["saturate", "wrap"])
def test_grouped_convolution_overflow_bias_and_prefix_statistics(device, mode):
    # Each depthwise window has a transient overflow even though the final
    # exact sum including bias is in range. Bias must be added last.
    x = torch.tensor([[[[100, 100, -100]], [[-100, -100, 100]]]], device=device)
    weight = torch.ones(2, 1, 1, 3, dtype=torch.int64, device=device)
    bias = torch.tensor([10, -10], device=device)
    stats = {}
    actual = int_conv2d(x, weight, bias, groups=2, acc=AccumulatorSpec(8, mode), stats=stats)
    expected = [scalar_sum([100, 100, -100, 10], 8, mode), scalar_sum([-100, -100, 100, -10], 8, mode)]
    assert actual.flatten().cpu().tolist() == expected
    assert stats == {"outputs": 2, "overflow_outputs": 2, "max_abs_sum": 110, "max_abs_product_sum": 100}


@pytest.mark.parametrize("device", DEVICES)
def test_grouped_convolution_stride_padding_dilation(device):
    g = torch.Generator().manual_seed(31)
    x = torch.randint(-8, 9, (2, 4, 7, 9), generator=g).to(device)
    w = torch.randint(-4, 5, (6, 2, 2, 3), generator=g).to(device)
    bias = torch.tensor([1, 2, -3, 4, 0, -6], device=device)
    expected = torch.nn.functional.conv2d(x.double(), w.double(), bias.double(), stride=(2, 1), padding=(1, 2), dilation=(2, 1), groups=2)
    actual = int_conv2d(x, w, bias, stride=(2, 1), padding=(1, 2), dilation=(2, 1), groups=2)
    assert torch.equal(actual.double(), expected)


@pytest.mark.parametrize("mode", ["saturate", "wrap"])
def test_large_bias_is_added_as_an_exact_integer_after_safe_product_sums(mode):
    x = torch.tensor([[[[1, 2, 3]]]])
    w = torch.ones(1, 1, 1, 3, dtype=torch.int64)
    bias = torch.tensor([2**54 + 3])
    stats = {}
    actual = int_conv2d(x, w, bias, acc=AccumulatorSpec(32, mode), stats=stats)
    assert actual.item() == scalar_sum([1, 2, 3, 2**54+3], 32, mode)
    assert stats["max_abs_product_sum"] == 6
    assert stats["max_abs_sum"] == 2**54 + 9
    assert stats["overflow_outputs"] == 1


def frozen(model):
    model.eval()
    apply_stage(model, QATStage(False, True, True, True))
    return model


@pytest.mark.parametrize("device", DEVICES)
def test_integer_model_exact_grid_and_relu6_matches_fake_quant(device):
    source = nn.Sequential(nn.Conv2d(1, 2, 1, bias=False), nn.BatchNorm2d(2), nn.ReLU6()).double().to(device)
    with torch.no_grad():
        source[0].weight[:] = torch.tensor([0.5, -1.0], device=device).reshape(2, 1, 1, 1)
        source[1].weight.fill_(math.sqrt(1 + source[1].eps))
        source[1].bias[:] = torch.tensor([0.5, 1.0], device=device)
    model = frozen(prepare_qat(source, w_spec=UniformSpec(4), a_spec=UniformSpec(8)))
    layer = model[1][0]
    layer.weight_fq.alpha = torch.tensor([63.5, 127.0], dtype=torch.float64, device=device)
    x = torch.tensor([[[[-20.0, -3, -1, 0, 1, 3, 20]]]], dtype=torch.float64, device=device)
    assert torch.equal(run_integer(model, x), model(x))
    assert run_integer(model, x).flatten().tolist() == [0, 0, 0, 1, 1, 2, 6, 6, 4, 2, 1, 0, 0, 0]


@pytest.mark.parametrize("device", DEVICES)
def test_mobilenet_fixed_point_reference_matches_every_layer_without_mutation(device):
    torch.manual_seed(118)
    model = prepare_qat(mobilenet_v2_cifar(width_mult=0.1), w_spec=UniformSpec(4), a_spec=UniformSpec(8)).to(device).eval()
    x = torch.randn(1, 3, 32, 32, device=device)
    with torch.no_grad():
        model(x)  # Observe nontrivial, per-layer grids.
    frozen(model)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    integer = IntegerModel(model, capture=True)
    reference = IntegerModel(model, reference=True, capture=True)
    assert torch.equal(integer(x), reference(x))
    assert integer.trace.keys() == reference.trace.keys()
    assert len(integer.trace) > 100
    assert all(torch.equal(value, reference.trace[name]) for name, value in integer.trace.items())
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in before.items())


def test_integer_model_requires_frozen_observers_and_eval():
    model = prepare_qat(nn.Sequential(nn.Conv2d(1, 1, 1), nn.BatchNorm2d(1)), w_spec=UniformSpec(4), a_spec=UniformSpec(8))
    with pytest.raises(ValueError, match="eval"):
        IntegerModel(model)
    model.eval()
    with pytest.raises(ValueError, match="Observers"):
        IntegerModel(model)
