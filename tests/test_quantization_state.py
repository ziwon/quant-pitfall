from copy import deepcopy

import pytest
import torch
from torch import nn

from quant_pitfalls.layers.conv_bn import QuantConvBn2d
from quant_pitfalls.quant.apot import APoTSpec, apot_fake_quant
from quant_pitfalls.quant.fake_quant import FakeQuantize
from quant_pitfalls.quant.observer import EMAMinMaxObserver, mse_clip_search
from quant_pitfalls.quant.uniform import UniformSpec, fake_quant


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    return request.param


def test_observer_channel_ema_and_checkpoint(device):
    x = torch.tensor([[-1, 2, -3], [1, 4, 6]], dtype=torch.float64, device=device, requires_grad=True)
    obs = EMAMinMaxObserver(0.25, axis=-1).to(device)
    assert obs(x) is x
    obs(x * 2)
    torch.testing.assert_close(obs.clip(), x.new_tensor([1.25, 5, 7.5]))
    assert not obs.min_val.requires_grad
    restored = EMAMinMaxObserver(0.25, axis=-1).to(device)
    restored.load_state_dict(deepcopy(obs.state_dict()))
    torch.testing.assert_close(restored.clip(), obs.clip())
    restored(x * 3)
    obs(x * 3)
    torch.testing.assert_close(restored.clip(), obs.clip())


def test_independent_fake_quant_switches(device):
    x = torch.tensor([[-1, 0.4, 1], [-2, 0.8, 2]], dtype=torch.float64, device=device)
    fq = FakeQuantize(UniformSpec(4), axis=0).to(device, dtype=torch.float64)
    fq.enable_fake_quant(False)
    assert fq(x) is x
    alpha = fq.alpha.clone()
    fq.enable_observer(False)
    assert fq(x * 10).equal(x * 10)
    torch.testing.assert_close(fq.alpha, alpha)
    fq.enable_fake_quant()
    torch.testing.assert_close(fq(x * 10), fake_quant(x * 10, alpha, fq.spec, axis=0))
    torch.testing.assert_close(fq.alpha, alpha)
    fq.enable_fake_quant(False)
    fq.enable_observer()
    fq(x * 2)
    assert (fq.alpha > alpha).all()


@pytest.mark.parametrize("spec", [UniformSpec(4), APoTSpec(4)])
def test_fake_quant_gradient_and_channel_state_roundtrip(device, spec):
    x = torch.tensor([[-1, 0.1, 0.4, 1], [-2, 0.2, 0.8, 2]], dtype=torch.float64, device=device, requires_grad=True)
    fq = FakeQuantize(spec, axis=0).to(device, dtype=torch.float64)
    actual = fq(x)
    expected = fake_quant(x, fq.alpha, spec, axis=0) if isinstance(spec, UniformSpec) else apot_fake_quant(x, fq.alpha[:, None], fq.levels)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x))
    fq.enable_observer(False)
    restored = FakeQuantize(spec, axis=0).to(device, dtype=torch.float64)
    restored.load_state_dict(deepcopy(fq.state_dict()))
    assert not bool(restored.observer_enabled)
    torch.testing.assert_close(restored(x * 2), fq(x * 2))


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_zero_channel_calibration_stays_finite(device, dtype):
    x = torch.zeros(2, 8, device=device, dtype=dtype, requires_grad=True)
    fq = FakeQuantize(UniformSpec(4), axis=0).to(device)
    y = fq(x)
    assert torch.equal(y, x) and torch.isfinite(y).all()
    assert (fq.alpha > 0).all()
    y.sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x))


def test_mse_calibration_channels_match_independent_search(device):
    gen = torch.Generator(device=device).manual_seed(0)
    x = torch.randn(2, 2000, generator=gen, device=device, dtype=torch.float64)
    x[1] *= 4
    x[0, -1], x[1, -1] = 8, -40
    spec = UniformSpec(4)
    alpha = mse_clip_search(x, spec, num_candidates=40, axis=0)
    independent = torch.stack([mse_clip_search(row, spec, num_candidates=40) for row in x])
    torch.testing.assert_close(alpha, independent)
    minmax = x.abs().amax(dim=1)
    mse = (fake_quant(x, alpha, spec, axis=0) - x).square().mean(dim=1)
    baseline = (fake_quant(x, minmax, spec, axis=0) - x).square().mean(dim=1)
    assert (mse < baseline).all()
    scalar_channels = x.new_tensor([0, 2, -3])
    scalar_alpha = mse_clip_search(scalar_channels, spec, axis=0)
    assert (scalar_alpha > 0).all()
    torch.testing.assert_close(fake_quant(scalar_channels, scalar_alpha, spec, axis=0), scalar_channels)


def test_training_float_fold_matches_bn_outputs_gradients_and_updates(device):
    torch.manual_seed(2)
    conv = nn.Conv2d(4, 6, 3, padding=2, dilation=2, groups=2, padding_mode="reflect", bias=True).to(device, dtype=torch.float64)
    bn = nn.BatchNorm2d(6).to(device, dtype=torch.float64)
    with torch.no_grad():
        bn.weight.copy_(bn.weight.new_tensor([0, -1, 0.5, 2, 0.1, 1.5]))
    layer = QuantConvBn2d.from_float(conv, bn)
    x = torch.randn(3, 4, 8, 8, device=device, dtype=torch.float64, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    expected = bn(conv(reference_x))
    actual = layer(x)
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-9)
    actual.square().mean().backward()
    expected.square().mean().backward()
    torch.testing.assert_close(x.grad, reference_x.grad, atol=1e-10, rtol=1e-9)
    for result, reference in [(layer.conv.weight, conv.weight), (layer.conv.bias, conv.bias), (layer.bn.weight, bn.weight), (layer.bn.bias, bn.bias)]:
        torch.testing.assert_close(result.grad, reference.grad, atol=1e-10, rtol=1e-9)
    torch.testing.assert_close(layer.bn.running_mean, bn.running_mean)
    torch.testing.assert_close(layer.bn.running_var, bn.running_var)
    assert layer.bn.num_batches_tracked == bn.num_batches_tracked
    deployment_weight, deployment_bias = layer.folded_weight_and_bias()
    deployment = layer.conv._conv_forward(x, deployment_weight, deployment_bias)
    torch.testing.assert_close(deployment, bn.eval()(conv(x)), atol=1e-10, rtol=1e-9)


def test_frozen_qat_export_checkpoint_and_unfreeze(device):
    torch.manual_seed(3)
    conv = nn.Conv2d(3, 4, 3, padding=1, bias=False).to(device, dtype=torch.float64)
    bn = nn.BatchNorm2d(4).to(device, dtype=torch.float64)
    with torch.no_grad():
        bn.weight[0] = 0
    fq = FakeQuantize(UniformSpec(4), axis=0).to(device, dtype=torch.float64)
    layer = QuantConvBn2d.from_float(conv, bn, weight_fq=fq)
    x = torch.randn(2, 3, 8, 8, device=device, dtype=torch.float64, requires_grad=True)
    layer(x).square().mean().backward()
    assert torch.isfinite(layer.conv.weight.grad).all() and torch.isfinite(layer.bn.weight.grad).all()
    layer.weight_fq.enable_observer(False)
    layer.freeze_bn_stats()
    layer.train()
    assert not layer.bn.training
    mean, var = layer.bn.running_mean.clone(), layer.bn.running_var.clone()
    weight, bias = layer.folded_weight_and_bias()
    expected = layer.conv._conv_forward(x, layer.weight_fq(weight), bias)
    actual = layer(x)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(layer.bn.running_mean, mean)
    torch.testing.assert_close(layer.bn.running_var, var)
    restored = QuantConvBn2d.from_float(conv, bn, weight_fq=FakeQuantize(UniformSpec(4), axis=0))
    restored.load_state_dict(deepcopy(layer.state_dict()))
    restored.train()
    assert not restored.bn.training
    torch.testing.assert_close(restored(x), actual)
    before = restored.bn.num_batches_tracked.clone()
    restored.freeze_bn_stats(False)
    restored(x)
    assert restored.bn.training and restored.bn.num_batches_tracked == before + 1
    unfrozen_state = deepcopy(restored.state_dict())
    restored.freeze_bn_stats()
    restored.load_state_dict(unfrozen_state)
    assert restored.bn.training and not bool(restored._bn_frozen)
