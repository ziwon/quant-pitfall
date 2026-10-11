"""Checkpoint-dependent analyses for the M3 notebooks."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from .convert import QATStage, apply_stage, prepare_qat
from .layers.conv_bn import QuantConvBn2d
from .models.mobilenetv2_cifar import mobilenet_v2_cifar
from .quant.apot import apot_fake_quant, apot_levels
from .quant.fake_quant import FakeQuantize
from .quant.observer import mse_clip_search
from .quant.uniform import UniformSpec, fake_quant, quantize
from .training import data_loaders, evaluate, load_float, quant_specs, seed_everything


def dataset_options(seed, device, data=Path("data")):
    return SimpleNamespace(data=data, seed=seed, device=device, batch_size=128, workers=4, subset=0, eval_subset=0)


def load_qat(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    w_spec, a_spec = quant_specs(SimpleNamespace(**config))
    model = prepare_qat(mobilenet_v2_cifar(width_mult=config["width_mult"]), w_spec=w_spec, a_spec=a_spec, per_channel=config["per_channel"])
    for module in model.modules():
        if isinstance(module, QuantConvBn2d):
            module.weight_quant_order = config.get("fold_order", "fold")
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    apply_stage(model, QATStage(False, True, True, True))
    return model.eval()


class InputQuantized(nn.Module):
    def __init__(self, model, alpha, mode):
        super().__init__()
        self.model = model
        self.register_buffer("alpha", alpha)
        self.mode = mode

    def forward(self, x):
        spec = UniformSpec(8, narrow_range=False, rounding="nearest" if self.mode == "nearest" else "floor")
        if self.mode == "half-lsb":
            x = x + self.alpha / spec.qmax / 2
        return self.model(fake_quant(x, self.alpha, spec))


@torch.no_grad()
def floor_bias_comparison(root=Path("."), device="cuda", seeds=(0, 1, 2)):
    """Change only the quantization/correction at the first Conv's input.

    Bias correction is estimated on an unaugmented training batch and evaluated
    on the separate test set. This is a single-layer controlled experiment;
    the full floor-activation QAT result is reported in the main result table.
    """
    rows = []
    for seed in seeds:
        seed_everything(seed)
        model, _ = load_float(root/f"runs/float-baseline-s{seed}/best.pt", device)
        model.eval()
        calibration, test, _ = data_loaders(dataset_options(seed, device, root/"data"), calibration=True)
        images, _ = next(iter(calibration))
        images = images.to(device)
        alpha = images.abs().max()
        nearest = fake_quant(images, alpha, UniformSpec(8, narrow_range=False))
        floor = fake_quant(images, alpha, UniformSpec(8, narrow_range=False, rounding="floor"))
        conv = model.features[0][0]
        reference = conv(nearest)
        floored = conv(floor)
        shift = (floored-reference).mean((0, 2, 3))
        for method in ("nearest", "floor", "half-lsb", "bias-corrected"):
            base = deepcopy(model)
            mode = "floor" if method == "bias-corrected" else method
            if method == "bias-corrected":
                bias = base.features[0][0].bias
                corrected = -shift if bias is None else bias.detach()-shift
                base.features[0][0].bias = nn.Parameter(corrected, requires_grad=False)
            candidate = InputQuantized(base, alpha, mode).to(device).eval()
            result = evaluate(candidate, test, device)
            probe = images + alpha/127/2 if method == "half-lsb" else images
            probe_spec = UniformSpec(8, narrow_range=False, rounding="nearest" if method == "nearest" else "floor")
            output = base.features[0][0](fake_quant(probe, alpha, probe_spec))
            difference = (output-reference).mean((0, 2, 3))
            rows.append({"seed": seed, "method": method, "top1": result["top1"],
                         "output_channel_mean_shift_rms": float(difference.square().mean().sqrt()),
                         "input_alpha": float(alpha)})
    return {"seeds": len(seeds), "rows": rows, "scope": "First-convolution input only; separate calibration and test data"}


@torch.no_grad()
def weight_distributions(root=Path("."), device="cuda", seeds=(0, 1, 2), candidates=100):
    """Compare 31-level grids on centered unit-variance trained layer weights."""
    rows = []
    levels = apot_levels(4, 2).to(device)
    for seed in seeds:
        model, _ = load_float(root/f"runs/float-baseline-s{seed}/best.pt", device)
        for name, module in model.named_modules():
            if not isinstance(module, (nn.Conv2d, nn.Linear)):
                continue
            values = module.weight.detach().double().flatten()
            values -= values.mean()
            variance = values.square().mean()
            if variance == 0:
                continue
            values /= variance.sqrt()
            kurtosis = float(values.pow(4).mean())
            uniform_alpha = mse_clip_search(values, UniformSpec(5), num_candidates=candidates)
            uniform_mse = float((fake_quant(values, uniform_alpha, UniformSpec(5))-values).square().mean())
            apot_mse, apot_alpha = float("inf"), None
            maximum = values.abs().max()
            for ratio in torch.linspace(0.05, 1, candidates).tolist():
                alpha = maximum*ratio
                error = float((apot_fake_quant(values, alpha, levels)-values).square().mean())
                if error < apot_mse:
                    apot_mse, apot_alpha = error, float(alpha)
            rows.append({"seed": seed, "layer": name, "kurtosis": kurtosis, "uniform_mse": uniform_mse,
                         "apot_mse": apot_mse, "mse_ratio": apot_mse/uniform_mse,
                         "uniform_alpha": float(uniform_alpha), "apot_alpha": apot_alpha})
    return {"seeds": len(seeds), "levels": 31, "candidates": candidates, "rows": rows}


@torch.no_grad()
def fold_order_comparison(root=Path("."), device="cuda", seeds=(0, 1, 2)):
    rows = []
    for seed in seeds:
        _, test, _ = data_loaders(dataset_options(seed, device, root/"data"), calibration=True)
        for granularity in ("pt", "pc"):
            for order in ("fold", "raw"):
                prefix = "qat-w4a8-pc-mse-staged-uniform-nearest-fp32" if granularity == "pc" and order == "fold" else f"nb04-{granularity}-{order}"
                model = load_qat(root/f"runs/{prefix}-s{seed}/best.pt", device)
                before = evaluate(model, test, device)
                mismatches, count = 0, 0
                if order == "raw":
                    for layer in model.modules():
                        if not isinstance(layer, QuantConvBn2d):
                            continue
                        fq = layer.weight_fq
                        folded, _ = layer.folded_weight_and_bias()
                        alpha = folded.abs().flatten(1).amax(1) if fq.axis == 0 else folded.abs().max()
                        alpha = alpha.clamp_min(torch.finfo(folded.dtype).tiny*fq.spec.qmax)
                        gamma, _ = layer._affine()
                        factor = gamma * torch.rsqrt(layer.bn.running_var + layer.bn.eps)
                        raw_quant = fake_quant(layer.conv.weight, fq.alpha, fq.spec, axis=fq.axis)
                        virtual_weight = raw_quant * factor.reshape(-1, 1, 1, 1)
                        naive = quantize(virtual_weight, alpha, fq.spec, axis=fq.axis)
                        correct = quantize(folded, alpha, fq.spec, axis=fq.axis)
                        mismatches += int((naive != correct).sum())
                        count += folded.numel()
                        deployment = FakeQuantize(fq.spec, axis=fq.axis).to(device)
                        deployment.quant_role = "weight"
                        deployment.alpha = alpha
                        deployment.enable_observer(False)
                        layer.deployment_weight_fq = deployment
                after = evaluate(model, test, device)
                rows.append({"seed": seed, "granularity": granularity, "order": order,
                             "virtual_top1": before["top1"], "deployment_top1": after["top1"],
                             "deployment_change_pp": after["top1"]-before["top1"],
                             "common_grid_order_mismatch_pct": 100*mismatches/count if count else 0})
    return {"seeds": len(seeds), "rows": rows,
            "scope": "Raw-Q folding is requantized on a folded min/max deployment grid; fold-first keeps its learned grid"}
