import pytest
import torch

from quant_pitfalls.analysis import expected_mse, ulp
from quant_pitfalls.quant.apot import apot_fake_quant, apot_quantize
from quant_pitfalls.quant.uniform import UniformSpec, dequantize, fake_quant, quantize


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_per_channel_quantization_preserves_dtype_and_device(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    x = torch.tensor([[1, 2, 3], [8, -16, 24]], dtype=torch.float64, device=device)
    alpha = torch.tensor([7, 14, 21], dtype=torch.float32)
    spec = UniformSpec(4)
    codes = quantize(x, alpha, spec, axis=-1)
    assert codes.dtype == x.dtype and codes.device == x.device
    assert codes.tolist() == [[1, 1, 1], [7, -7, 7]]
    restored = dequantize(codes, alpha, spec, axis=1)
    assert restored.tolist() == [[1, 2, 3], [7, -14, 21]]
    torch.testing.assert_close(fake_quant(x, alpha, spec, axis=1), restored)


@pytest.mark.parametrize("rounding", ["nearest", "floor"])
def test_uniform_ste_clips_before_rounding(rounding):
    x = torch.tensor([-7.1, -7, -6.8, 0, 6.8, 7, 7.1], dtype=torch.float64, requires_grad=True)
    fake_quant(x, 7, UniformSpec(4, rounding=rounding)).sum().backward()
    assert x.grad.tolist() == [0, 1, 1, 1, 1, 1, 0]


def test_full_range_keeps_extra_negative_code():
    x = torch.tensor([-8.0, 8.0])
    assert quantize(x, 7, UniformSpec(4, narrow_range=False)).tolist() == [-8, 7]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_apot_midpoint_ties_and_unsigned_clipping(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    levels = torch.tensor([0, 0.5, 1], dtype=torch.float32)
    x = torch.tensor([-0.75, -0.25, 0.25, 0.75], dtype=torch.float64, device=device)
    assert apot_quantize(x, 1, levels).tolist() == [-1, -0.5, 0.5, 1]
    assert apot_quantize(x, 1, levels, signed=False).tolist() == [0, 0, 0.5, 1]
    x = torch.tensor([-1.1, -1, 0, 1, 1.1], device=device, requires_grad=True)
    apot_fake_quant(x, 1, levels).sum().backward()
    assert x.grad.tolist() == [0, 1, 1, 1, 0]


@pytest.mark.parametrize("dist", ["gaussian", "laplace", "student_t"])
def test_tail_integration_preserves_unit_variance(dist):
    calls = []

    def zero_quantizer(x):
        calls.append(x.numel())
        return torch.zeros_like(x)

    # An always-zero quantizer has MSE equal to the full variance, including tails.
    assert expected_mse(zero_quantizer, dist, limit=2) == pytest.approx(1, abs=1e-8)
    assert calls == [24001]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_ulp_at_zero_and_subnormal_values(dtype):
    smallest = torch.nextafter(torch.tensor(0, dtype=dtype), torch.tensor(1, dtype=dtype))
    values = torch.stack([torch.zeros_like(smallest), smallest])
    assert torch.equal(ulp(values), smallest.expand_as(values))
