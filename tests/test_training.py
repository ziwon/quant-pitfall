from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import runpy

import pytest
import torch
from torch import nn

from quant_pitfalls.convert import QATSchedule, QATStage, apply_stage, prepare_qat, stage_at
from quant_pitfalls.layers.conv_bn import QuantConvBn2d
from quant_pitfalls.layers.linear import QuantLinear
from quant_pitfalls.models.mobilenetv2_cifar import mobilenet_v2_cifar
from quant_pitfalls.quant.fake_quant import FakeQuantize
from quant_pitfalls.quant.uniform import UniformSpec
from quant_pitfalls.training import WeightStorageOptimizer, calibrate, evaluate, kd_loss, model_weights
from quant_pitfalls import training


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    return request.param


def test_cifar_layout_and_parameter_count(device):
    model = mobilenet_v2_cifar().to(device).eval()
    x = torch.randn(2, 3, 32, 32, device=device)
    with torch.no_grad():
        assert model.features(x).shape == (2, 1280, 4, 4)
        assert model(x).shape == (2, 10)
    assert sum(p.numel() for p in model.parameters()) == 2236682


def test_large_activation_sample_indices_stay_inside_bounds(device):
    length = 2**25
    indices = training._sample_indices(length, 8192, device)
    assert indices.dtype == torch.int64
    assert indices[0] == 0 and indices[-1] == length-1
    assert ((indices >= 0) & (indices < length)).all()
    assert (indices[1:] > indices[:-1]).all()


def test_conversion_disabled_quant_matches_float_and_copies_state(device):
    torch.manual_seed(0)
    model = mobilenet_v2_cifar(width_mult=0.25).to(device).eval()
    quantized = prepare_qat(model, w_spec=UniformSpec(4), a_spec=UniformSpec(8))
    apply_stage(quantized, QATStage(False, False, False, True))
    x = torch.randn(2, 3, 32, 32, device=device)
    with torch.no_grad():
        torch.testing.assert_close(quantized(x), model(x), atol=1e-6, rtol=1e-4)
    layers = [m for m in quantized.modules() if isinstance(m, QuantConvBn2d)]
    assert len(layers) == 52
    assert layers[0].weight_fq.spec.bits == 8
    assert all(layer.weight_fq.spec.bits == 4 for layer in layers[1:])
    classifier = next(m for m in quantized.modules() if isinstance(m, QuantLinear))
    assert classifier.weight_fq.spec.bits == 8
    assert layers[0].conv.weight.data_ptr() != model.features[0][0].weight.data_ptr()


def test_schedule_boundaries_and_train_does_not_unfreeze(device):
    schedule = QATSchedule()
    assert stage_at(0, 20, schedule) == QATStage(True, False, False, False)
    assert stage_at(1, 20, schedule) == QATStage(True, True, False, False)
    assert stage_at(2, 20, schedule) == QATStage(True, True, True, False)
    assert stage_at(10, 20, schedule) == QATStage(True, True, True, True)
    assert stage_at(15, 20, schedule) == QATStage(False, True, True, True)
    model = prepare_qat(nn.Sequential(nn.Conv2d(3, 4, 1), nn.BatchNorm2d(4), nn.ReLU6()).to(device),
                        w_spec=UniformSpec(4), a_spec=UniformSpec(8))
    apply_stage(model, stage_at(15, 20, schedule))
    model.train()
    assert not next(m for m in model.modules() if isinstance(m, QuantConvBn2d)).bn.training
    assert all(not bool(m.observer_enabled) for m in model.modules() if isinstance(m, FakeQuantize))


def test_raw_quantization_order_control_and_deployment_requantization(device):
    conv = nn.Conv2d(2, 3, 1, bias=False).to(device)
    bn = nn.BatchNorm2d(3).to(device).eval()
    with torch.no_grad():
        bn.weight.copy_(torch.tensor([0.2, 1.3, -2], device=device))
    fq = FakeQuantize(UniformSpec(4)).to(device)
    layer = QuantConvBn2d.from_float(conv, bn, weight_fq=fq).eval()
    layer.weight_quant_order = "raw"
    fq(conv.weight)
    fq.enable_observer(False)
    x = torch.randn(2, 2, 4, 4, device=device)
    factor = bn.weight * torch.rsqrt(bn.running_var+bn.eps)
    weight = fq(conv.weight) * factor[:,None,None,None]
    _, bias = layer.folded_weight_and_bias()
    expected = layer.conv._conv_forward(x, weight, bias)
    torch.testing.assert_close(layer(x), expected)
    deployment = FakeQuantize(UniformSpec(4)).to(device)
    deployment(weight)
    deployment.enable_observer(False)
    layer.deployment_weight_fq = deployment
    torch.testing.assert_close(layer(x), layer.conv._conv_forward(x, deployment(weight), bias))


@pytest.mark.parametrize("optimizer_class", [torch.optim.SGD, torch.optim.Adam])
def test_fp16_storage_only_rounds_final_update_and_restores_state(device, optimizer_class):
    fp32 = nn.Parameter(torch.tensor([0.01], device=device))
    fp16 = nn.Parameter(fp32.detach().clone())
    normal = WeightStorageOptimizer(optimizer_class([fp32], lr=1e-6), [fp32])
    reduced = WeightStorageOptimizer(optimizer_class([fp16], lr=1e-6), [fp16], "fp16")
    before_normal, before_reduced = fp32.detach().clone(), fp16.detach().clone()
    fp32.grad, fp16.grad = torch.ones_like(fp32), torch.ones_like(fp16)
    normal.step()
    reduced.step()
    assert (fp32 != before_normal).all() and torch.equal(fp16, before_reduced)
    assert fp16.dtype == torch.float32 and reduced.storage[0].dtype == torch.float16
    assert torch.isfinite(fp16).all()
    if optimizer_class is torch.optim.Adam:
        assert reduced.optimizer.state[fp16]["exp_avg"].dtype == torch.float32
    restored_parameter = nn.Parameter(torch.zeros_like(fp16))
    restored = WeightStorageOptimizer(optimizer_class([restored_parameter], lr=1e-6), [restored_parameter], "fp16")
    restored.load_state_dict(deepcopy(reduced.state_dict()))
    torch.testing.assert_close(restored_parameter, fp16)


def test_calibration_evaluation_and_qat_checkpoint(device):
    torch.manual_seed(3)
    float_model = nn.Sequential(nn.Conv2d(3, 4, 1, bias=False), nn.BatchNorm2d(4), nn.ReLU6(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 2)).to(device)
    model = prepare_qat(float_model, w_spec=UniformSpec(4), a_spec=UniformSpec(8))
    data = torch.randn(8, 3, 8, 8)
    labels = torch.arange(8) % 2
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(data, labels), batch_size=4)
    report = calibrate(model, loader, device, batches=2, clip="mse", candidates=8)
    assert report["batches"] == 2
    restored = prepare_qat(float_model, w_spec=UniformSpec(4), a_spec=UniformSpec(8))
    restored.load_state_dict(deepcopy(model.state_dict()))
    apply_stage(restored, QATStage(False, True, True, True))
    model.eval()
    restored.eval()
    x = data.to(device)
    torch.testing.assert_close(restored(x), model(x))
    model.train()
    apply_stage(model, QATStage(True, True, True, True))
    before = [m.alpha.clone() for m in model.modules() if isinstance(m, FakeQuantize)]
    result = evaluate(model, loader, device)
    assert result["samples"] == 8 and model.training
    for m, alpha in zip([m for m in model.modules() if isinstance(m, FakeQuantize)], before):
        assert bool(m.observer_enabled)
        torch.testing.assert_close(m.alpha, alpha)
    optimizer = WeightStorageOptimizer(torch.optim.Adam(model.parameters(), lr=1e-4), model_weights(model))
    optimizer.zero_grad(set_to_none=True)
    output = model(x)
    loss = kd_loss(output, torch.randn_like(output, requires_grad=True), labels.to(device))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()


def test_epoch_resume_restores_parameters_optimizer_and_random_state(device, tmp_path, monkeypatch):
    def loaders(args, **kwargs):
        generator = torch.Generator().manual_seed(args.seed)
        images = torch.randn(16, 3, 4, 4, generator=torch.Generator().manual_seed(123))
        labels = torch.arange(16) % 2
        dataset = torch.utils.data.TensorDataset(images, labels)
        return (torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, generator=generator),
                torch.utils.data.DataLoader(dataset, batch_size=8), generator)

    def model(**kwargs):
        return nn.Sequential(nn.Conv2d(3, 4, 1), nn.BatchNorm2d(4), nn.ReLU6(),
                             nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.2), nn.Linear(4, 2))

    monkeypatch.setattr(training, "data_loaders", loaders)
    monkeypatch.setattr(training, "mobilenet_v2_cifar", model)
    common = dict(stage="float", seed=0, data=Path("data"), device=device, batch_size=4, workers=0,
                  epochs=2, lr=0.01, optimizer="sgd", weight_decay=0, amp="off", master_dtype="fp32",
                  width_mult=1, schedule="staged", weight_quant_from=0.05, act_quant_from=0.1,
                  freeze_bn_from=0.5, freeze_observer_from=0.75, track_updates=False, track_diagnostics=False,
                  max_steps=0, subset=0, eval_subset=0, name=None, resume=None, stop_after_epoch=0)
    complete_args = SimpleNamespace(**common, out=tmp_path/"complete")
    complete = training.train(complete_args)
    resumed_args = SimpleNamespace(**common, out=tmp_path/"resumed")
    resumed_args.stop_after_epoch = 1
    resumed = training.train(resumed_args)
    resumed_args.stop_after_epoch = 0
    resumed_args.resume = resumed/"last.pt"
    training.train(resumed_args)
    reference = torch.load(complete/"last.pt", map_location="cpu", weights_only=False)
    restored = torch.load(resumed/"last.pt", map_location="cpu", weights_only=False)
    for name, parameter in reference["model"].items():
        torch.testing.assert_close(restored["model"][name], parameter, atol=0, rtol=0)
    assert restored["global_step"] == reference["global_step"] == 8
    assert [h["train_loss"] for h in restored["history"]] == [h["train_loss"] for h in reference["history"]]
    resumed_args.lr = 0.02
    with pytest.raises(ValueError, match="hyperparameters"):
        training.train(resumed_args)


def test_kd_trainer_updates_student_and_keeps_teacher_frozen(device, tmp_path, monkeypatch):
    teacher = nn.Sequential(nn.Conv2d(3, 4, 1, bias=False), nn.BatchNorm2d(4), nn.ReLU6(),
                            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 2)).to(device)
    original_weight = teacher[0].weight.detach().clone()
    original_mean = teacher[1].running_mean.clone()

    def loaders(args, **kwargs):
        generator = torch.Generator().manual_seed(args.seed)
        images = torch.randn(8, 3, 4, 4, generator=torch.Generator().manual_seed(123))
        labels = torch.arange(8) % 2
        dataset = torch.utils.data.TensorDataset(images, labels)
        return (torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, generator=generator),
                torch.utils.data.DataLoader(dataset, batch_size=4), generator)

    monkeypatch.setattr(training, "data_loaders", loaders)
    monkeypatch.setattr(training, "load_float", lambda *args, **kwargs: (teacher, {"config": {"seed": 0}}))
    parser = runpy.run_path(str(Path(__file__).resolve().parents[1]/"train.py"))["build_parser"]()
    args = parser.parse_args(["qat", "--ckpt", "unused.pt", "--device", device, "--out", str(tmp_path),
                              "--name", "kd-regression", "--epochs", "1", "--calib-batches", "1",
                              "--clip", "minmax", "--per-channel", "--kd"])
    directory = training.train(args)
    checkpoint = torch.load(directory/"last.pt", map_location="cpu", weights_only=False)
    student_weight = checkpoint["model"]["model.0.conv.weight"]
    assert not torch.equal(student_weight, original_weight.cpu())
    torch.testing.assert_close(teacher[0].weight, original_weight, atol=0, rtol=0)
    torch.testing.assert_close(teacher[1].running_mean, original_mean, atol=0, rtol=0)
    assert not teacher.training
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in teacher.parameters())
