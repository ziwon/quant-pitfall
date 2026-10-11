"""Reproducible CIFAR-10 training and bounded post-training calibration."""

from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .convert import QATSchedule, QATStage, apply_stage, prepare_qat, stage_at
from .layers.conv_bn import QuantConvBn2d
from .layers.linear import QuantLinear
from .models.mobilenetv2_cifar import mobilenet_v2_cifar
from .quant.apot import APoTSpec, apot_fake_quant
from .quant.fake_quant import FakeQuantize
from .quant.observer import mse_clip_search, _minimum_alpha
from .quant.uniform import UniformSpec, quantize


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def data_loaders(args, *, calibration=False):
    from torchvision import datasets, transforms

    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    plain = transforms.Compose([transforms.ToTensor(), normalize])
    augment = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
    train = datasets.CIFAR10(args.data, train=True, transform=plain if calibration else augment, download=False)
    test = datasets.CIFAR10(args.data, train=False, transform=plain, download=False)
    if args.subset:
        indices = torch.randperm(len(train), generator=torch.Generator().manual_seed(args.seed))[:args.subset].tolist()
        train = torch.utils.data.Subset(train, indices)
    if args.eval_subset:
        test = torch.utils.data.Subset(test, range(min(args.eval_subset, len(test))))
    generator = torch.Generator().manual_seed(args.seed)
    options = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=str(args.device).startswith("cuda"), worker_init_fn=seed_worker)
    train_loader = torch.utils.data.DataLoader(train, shuffle=True, generator=generator, **options)
    test_loader = torch.utils.data.DataLoader(test, shuffle=False, **options)
    return train_loader, test_loader, generator


def quant_specs(args):
    weight = APoTSpec(args.w_bits) if args.weight_quantizer == "apot" else UniformSpec(args.w_bits)
    return weight, UniformSpec(args.a_bits, rounding=args.rounding)


def load_float(path: Path, device, *, width_mult=1.0):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    width_mult = checkpoint.get("config", {}).get("width_mult", width_mult)
    model = mobilenet_v2_cifar(width_mult=width_mult)
    model.load_state_dict(checkpoint["model"])
    return model.to(device), checkpoint


def _sample_indices(length: int, count: int, device):
    """Evenly spaced int64 indices; float32 linspace can round length-1 up."""
    return torch.arange(count, device=device, dtype=torch.int64) * (length-1) // max(count-1, 1)


@torch.no_grad()
def calibrate(model, loader, device, *, batches=32, clip="minmax", candidates=100):
    """Aggregate calibration extrema; search bounded samples for MSE clips.

    Activations retain up to 16384 evenly selected values over all batches.
    Weight tensors are retained once, with up to 2048 values per channel.
    Extrema always use all observed values. Data augmentation is disabled by
    the caller. The returned quantized model has observers and BN frozen.
    """
    if batches <= 0:
        raise ValueError("Calibration requires at least one batch")
    model.eval()
    apply_stage(model, QATStage(True, False, False, True))
    records = {}
    handles = []
    quantizers = [m for m in model.modules() if isinstance(m, FakeQuantize)]
    per_batch = max(1, 16384 // batches)

    def collect(module, inputs):
        values = inputs[0].detach().float()
        key = id(module)
        if getattr(module, "quant_role", "activation") == "weight" and key in records:
            return
        if module.axis is None:
            rows = values.reshape(1, -1)
        else:
            moved = values.movedim(module.axis, 0)
            rows = moved.reshape(moved.shape[0], -1)
        low, high = rows.amin(1), rows.amax(1)
        if key not in records:
            records[key] = [low, high, []]
        record = records[key]
        record[0] = torch.minimum(record[0], low)
        record[1] = torch.maximum(record[1], high)
        if clip == "mse":
            count = min(rows.shape[1], 2048 if getattr(module, "quant_role", "activation") == "weight" else per_batch)
            indices = _sample_indices(rows.shape[1], count, values.device)
            record[2].append(rows[:, indices].cpu())

    for fq in quantizers:
        handles.append(fq.register_forward_pre_hook(collect))
    started = time.perf_counter()
    observed = 0
    try:
        for index, (images, _) in enumerate(loader):
            if index >= batches:
                break
            model(images.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last))
            observed += 1
    finally:
        for handle in handles:
            handle.remove()
    if observed == 0:
        raise ValueError("Calibration loader was empty")
    for fq in quantizers:
        low, high, samples = records[id(fq)]
        alpha = torch.maximum(low.abs(), high.abs()) if fq.spec.signed else high.clamp_min(0)
        minimum = _minimum_alpha(torch.float32, fq.spec if isinstance(fq.spec, UniformSpec) else None)
        alpha = alpha.clamp_min(minimum)
        if clip == "mse":
            values = torch.cat(samples, dim=1)
            if isinstance(fq.spec, UniformSpec):
                alpha = mse_clip_search(values, fq.spec, num_candidates=candidates, axis=0)
            else:
                maximum = alpha.cpu()
                best = torch.full_like(maximum, float("inf"))
                for ratio in torch.linspace(0.05, 1, candidates):
                    candidate = (maximum * ratio).clamp_min(minimum)
                    restored = apot_fake_quant(values, candidate[:, None], fq.levels.cpu(), signed=fq.spec.signed)
                    error = (restored - values).square().mean(1)
                    alpha = torch.where(error < best, candidate, alpha.cpu())
                    best = torch.minimum(best, error)
        alpha = alpha.to(device)
        fq.alpha = alpha[0] if fq.axis is None else alpha
        fq.observer.min_val = low[0] if fq.axis is None else low
        fq.observer.max_val = high[0] if fq.axis is None else high
    apply_stage(model, QATStage(False, True, True, True))
    return {"batches": observed, "clip": clip, "seconds": time.perf_counter() - started}


class WeightStorageOptimizer:
    """FP32 optimizer/forward, with optional FP16 weight storage after addition.

    Only Conv/Linear weights use reduced storage precision. BN affine values,
    biases, gradients, optimizer state and computed deltas remain FP32. After
    each step, storage becomes round_fp16(storage_fp32 - delta_fp32), and the
    FP32 working parameter is refreshed from storage. There is no FP32 master
    copy accumulating sub-ulp updates in the FP16 experiment.
    """
    def __init__(self, optimizer, weights, dtype="fp32"):
        self.optimizer = optimizer
        self.weights = list(weights)
        self.dtype = dtype
        self.storage = [p.detach().half().clone() for p in self.weights] if dtype == "fp16" else []
        if dtype not in ("fp16", "fp32"):
            raise ValueError("Unknown storage dtype")
        with torch.no_grad():
            for parameter, stored in zip(self.weights, self.storage):
                parameter.copy_(stored.float())

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def zero_grad(self, **kwargs):
        self.optimizer.zero_grad(**kwargs)

    @torch.no_grad()
    def step(self):
        before = [p.clone() for p in self.weights] if self.storage else []
        self.optimizer.step()
        for parameter, previous, stored in zip(self.weights, before, self.storage):
            delta = previous - parameter
            stored.copy_((stored.float() - delta).half())
            parameter.copy_(stored.float())

    def state_dict(self):
        return {"optimizer": self.optimizer.state_dict(), "dtype": self.dtype, "storage": self.storage}

    def load_state_dict(self, state):
        if state["dtype"] != self.dtype:
            raise ValueError("Resume storage dtype must match")
        self.optimizer.load_state_dict(state["optimizer"])
        with torch.no_grad():
            for parameter, stored, loaded in zip(self.weights, self.storage, state["storage"]):
                stored.copy_(loaded)
                parameter.copy_(stored.float())


def model_weights(model):
    return [m.weight for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]


@torch.no_grad()
def weight_codes(model):
    codes = []
    for module in model.modules():
        if isinstance(module, QuantConvBn2d):
            weight = module.conv.weight if module.weight_quant_order == "raw" else module.folded_weight_and_bias()[0]
        elif isinstance(module, QuantLinear):
            weight = module.linear.weight
        else:
            continue
        fq = module.weight_fq
        if isinstance(fq.spec, UniformSpec):
            codes.append(quantize(weight, fq.alpha, fq.spec, axis=fq.axis))
        else:
            alpha = fq.alpha.reshape(-1, *([1]*(weight.ndim-1))) if fq.axis is not None else fq.alpha
            codes.append(apot_fake_quant(weight, alpha, fq.levels, signed=fq.spec.signed))
    return codes


@torch.no_grad()
def evaluate(model, loader, device, amp="off"):
    """Evaluate with observers disabled; restore their switches afterwards."""
    training = model.training
    quantizers = [m for m in model.modules() if isinstance(m, FakeQuantize)]
    switches = [bool(m.observer_enabled) for m in quantizers]
    for module in quantizers:
        module.enable_observer(False)
    model.eval()
    totals = torch.zeros(3, device=device, dtype=torch.float64)
    try:
        for images, labels in loader:
            images = images.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16, enabled=amp == "bf16"):
                output = model(images)
            totals[0] += F.cross_entropy(output.float(), labels, reduction="sum")
            totals[1] += (output.argmax(1) == labels).sum()
            totals[2] += labels.numel()
    finally:
        model.train(training)
        for module, switch in zip(quantizers, switches):
            module.enable_observer(switch)
    loss, correct, count = totals.cpu().tolist()
    return {"loss": loss / count, "top1": 100 * correct / count, "samples": int(count)}


def kd_loss(student, teacher, labels, weight=0.5, temperature=4.0):
    ce = F.cross_entropy(student, labels)
    kl = F.kl_div(F.log_softmax(student / temperature, dim=1), F.softmax(teacher.detach() / temperature, dim=1), reduction="batchmean")
    return (1-weight)*ce + weight*temperature**2*kl


def atomic_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def public_config(args):
    return {key: (str(value) if not value.is_absolute() else value.name) if isinstance(value, Path) else value
            for key, value in vars(args).items() if key != "run"}


def run_name(args):
    if args.name:
        return args.name
    if args.stage == "float":
        return f"float-baseline-s{args.seed}"
    channel = "pc" if args.per_channel else "pt"
    base = f"{args.stage}-w{args.w_bits}a{args.a_bits}-{channel}-{args.clip}"
    if args.stage == "qat":
        base += f"-{args.schedule}-{args.weight_quantizer}-{args.rounding}-{args.master_dtype}"
        if args.kd:
            base += "-kd"
    return base + f"-s{args.seed}"


def train(args):
    """Train/evaluate/checkpoint each epoch; resume restores optimizer and RNG."""
    seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    directory = args.out / run_name(args)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "last.pt").exists() and not args.resume:
        raise FileExistsError("Run already has a checkpoint; use --resume or a new --name")
    config = public_config(args)
    loader, test_loader, generator = data_loaders(args)
    teacher = None
    calibration = None
    if args.stage == "float":
        model = mobilenet_v2_cifar(width_mult=args.width_mult).to(device)
    else:
        model, source = load_float(args.ckpt, device, width_mult=args.width_mult)
        if source.get("config", {}).get("seed") != args.seed:
            raise ValueError("Each quantized run must start from the matching float seed")
        if args.kd:
            teacher = model.eval()
        w_spec, a_spec = quant_specs(args)
        model = prepare_qat(model, w_spec=w_spec, a_spec=a_spec, per_channel=args.per_channel)
        if teacher is not None:
            # Freeze only after copying; prepare_qat preserves requires_grad flags.
            teacher.requires_grad_(False)
        for module in model.modules():
            if isinstance(module, QuantConvBn2d):
                module.weight_quant_order = args.fold_order
        if not args.resume and not args.skip_calibration:
            calibration_loader, _, _ = data_loaders(args, calibration=True)
            calibration = calibrate(model, calibration_loader, device, batches=args.calib_batches, clip=args.clip, candidates=args.clip_candidates)
    model.to(memory_format=torch.channels_last)
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps=1e-8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    storage_optimizer = WeightStorageOptimizer(optimizer, model_weights(model), args.master_dtype)
    schedule = QATSchedule(args.weight_quant_from, args.act_quant_from, args.freeze_bn_from, args.freeze_observer_from)
    if args.schedule == "one-shot":
        schedule = QATSchedule(0, 0, schedule.freeze_bn_from, schedule.freeze_observer_from)
    history, diagnostics, start_epoch, global_step, best = [], [], 0, 0, -math.inf
    previous_runtime = 0.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if checkpoint["config"]["stage"] != args.stage or checkpoint["config"]["epochs"] != args.epochs:
            raise ValueError("Resume stage and total epochs must match the checkpoint")
        immutable = ("seed", "batch_size", "epochs", "lr", "width_mult", "optimizer", "weight_decay", "amp", "master_dtype", "schedule",
                     "w_bits", "a_bits", "per_channel", "weight_quantizer", "rounding", "clip", "kd", "kd_lambda", "kd_tau",
                     "weight_quant_from", "act_quant_from", "freeze_bn_from", "freeze_observer_from", "fold_order", "skip_calibration",
                     "subset", "eval_subset", "max_steps")
        if any(checkpoint["config"].get(key) != config.get(key) for key in immutable):
            raise ValueError("Resume hyperparameters must match the checkpoint")
        model.load_state_dict(checkpoint["model"])
        storage_optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        history = checkpoint["history"]
        diagnostics = checkpoint.get("diagnostics", [])
        start_epoch, global_step, best = checkpoint["epoch"]+1, checkpoint["global_step"], checkpoint["best_top1"]
        previous_runtime = checkpoint["runtime_seconds"]
        random.setstate(checkpoint["python_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
        torch.set_rng_state(checkpoint["torch_rng"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng"]])
        generator.set_state(checkpoint["loader_rng"].cpu())
        calibration = checkpoint.get("calibration")
    atomic_json(directory / "config.json", config)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for epoch in range(start_epoch, args.epochs):
        model.train()
        stage = stage_at(epoch, args.epochs, schedule)
        if args.stage == "qat":
            apply_stage(model, stage)
        sums = torch.zeros(4, device=device, dtype=torch.float64)
        epoch_started = time.perf_counter()
        step_times = []
        changes = []
        code_changes = []
        for images, labels in loader:
            if args.max_steps and global_step >= args.max_steps:
                break
            step_started = time.perf_counter()
            images = images.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            labels = labels.to(device, non_blocking=True)
            storage_optimizer.zero_grad(set_to_none=True)
            before = [p.detach().clone() for p in storage_optimizer.weights] if args.track_updates else []
            previous_codes = weight_codes(model) if args.track_updates else []
            diagnostic = args.track_diagnostics and global_step < 1000
            bn_means = [m.bn.running_mean.detach().clone() for m in model.modules() if isinstance(m, QuantConvBn2d)] if diagnostic else []
            clipped = torch.zeros(2, device=device, dtype=torch.float64)
            hooks = []
            if diagnostic:
                def clip_count(module, inputs, output):
                    values = inputs[0].detach()
                    alpha = module.alpha
                    low = -alpha if module.spec.signed else 0
                    clipped[0] += ((values < low) | (values > alpha)).sum()
                    clipped[1] += values.numel()
                for module in model.modules():
                    if isinstance(module, FakeQuantize) and module.quant_role == "activation":
                        hooks.append(module.register_forward_hook(clip_count))
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.amp == "bf16"):
                output = model(images)
                if teacher is None:
                    loss = F.cross_entropy(output.float(), labels)
                else:
                    with torch.no_grad():
                        targets = teacher(images)
                    loss = kd_loss(output.float(), targets.float(), labels, args.kd_lambda, args.kd_tau)
            for hook in hooks:
                hook.remove()
            if not torch.isfinite(loss):
                raise FloatingPointError("Training loss is nonfinite")
            loss.backward()
            storage_optimizer.step()
            if before:
                changed = sum((p != old).sum() for p, old in zip(storage_optimizer.weights, before))
                changes.append(float(changed) / sum(p.numel() for p in storage_optimizer.weights))
                current_codes = weight_codes(model)
                changed_codes = sum((now != old).sum() for now, old in zip(current_codes, previous_codes))
                code_changes.append(float(changed_codes) / sum(c.numel() for c in current_codes))
            if diagnostic:
                mean_change = sum((m.bn.running_mean-old).double().square().sum() for m, old in zip(
                    [m for m in model.modules() if isinstance(m, QuantConvBn2d)], bn_means))
                diagnostics.append({"step": global_step+1, "loss": float(loss.detach()),
                                    "activation_clipping_fraction": float(clipped[0]/clipped[1]),
                                    "bn_mean_change_l2": float(mean_change.sqrt())})
            sums[0] += loss.detach() * labels.numel()
            sums[1] += (output.detach().argmax(1) == labels).sum()
            sums[2] += labels.numel()
            sums[3] += F.cross_entropy(output.detach().float(), labels, reduction="sum")
            global_step += 1
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            step_times.append(time.perf_counter() - step_started)
        train_loss, correct, count, ce_sum = sums.cpu().tolist()
        if count == 0:
            break
        train_seconds = time.perf_counter() - epoch_started
        evaluation = evaluate(model, test_loader, device)
        metrics = {"epoch": epoch+1, "step": global_step, "train_loss": train_loss/count,
                   "train_top1": 100*correct/count, "test_loss": evaluation["loss"], "test_top1": evaluation["top1"],
                   "train_ce_loss": ce_sum/count,
                   "lr": optimizer.param_groups[0]["lr"], "train_seconds": train_seconds,
                   "seconds": time.perf_counter()-epoch_started, "step_seconds_mean": float(np.mean(step_times)),
                   "step_seconds_after_warmup": float(np.mean(step_times[5:] or step_times)),
                   "stage": asdict(stage) if args.stage == "qat" else None}
        if changes:
            metrics["weight_change_fractions"] = changes
            metrics["weight_code_change_fractions"] = code_changes
        history.append(metrics)
        scheduler.step()
        improved = evaluation["top1"] > best
        best = max(best, evaluation["top1"])
        runtime = previous_runtime + time.perf_counter()-started
        checkpoint = {"model": model.state_dict(), "optimizer": storage_optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                      "config": config, "epoch": epoch, "global_step": global_step, "best_top1": best, "history": history,
                      "runtime_seconds": runtime, "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                      "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                      "loader_rng": generator.get_state(), "calibration": calibration, "diagnostics": diagnostics}
        temporary = directory / "checkpoint.tmp"
        torch.save(checkpoint, temporary)
        temporary.replace(directory / "last.pt")
        if improved:
            torch.save(checkpoint, directory / "best.pt")
        report = {"config": config, "history": history, "best_top1": best, "final_top1": evaluation["top1"],
                  "runtime_seconds": runtime, "calibration": calibration,
                  "peak_memory_gib": torch.cuda.max_memory_allocated(device)/2**30 if device.type == "cuda" else 0,
                  "complete": epoch+1 == args.epochs and not args.max_steps and not args.subset and not args.eval_subset,
                  "environment": {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None}}
        atomic_json(directory / "metrics.json", report)
        if diagnostics:
            atomic_json(directory / "diagnostics.json", diagnostics)
        line = json.dumps(metrics, allow_nan=False)
        with (directory / "log.txt").open("a") as log:
            log.write(line + "\n")
        print(line, flush=True)
        if args.max_steps and global_step >= args.max_steps:
            break
        if args.stop_after_epoch and epoch+1 >= args.stop_after_epoch:
            break
    return directory


def ptq(args):
    seed_everything(args.seed)
    directory = args.out / run_name(args)
    if (directory / "metrics.json").exists():
        raise FileExistsError("PTQ run exists; use a new --name")
    directory.mkdir(parents=True, exist_ok=True)
    model, source = load_float(args.ckpt, args.device, width_mult=args.width_mult)
    if source.get("config", {}).get("seed") != args.seed:
        raise ValueError("Each quantized run must start from the matching float seed")
    w_spec, a_spec = quant_specs(args)
    model = prepare_qat(model, w_spec=w_spec, a_spec=a_spec, per_channel=args.per_channel)
    loader, test_loader, _ = data_loaders(args, calibration=True)
    started = time.perf_counter()
    calibration = calibrate(model, loader, args.device, batches=args.calib_batches, clip=args.clip, candidates=args.clip_candidates)
    result = evaluate(model, test_loader, args.device)
    config = public_config(args)
    atomic_json(directory / "config.json", config)
    report = {"config": config, "calibration": calibration, "best_top1": result["top1"], "final_top1": result["top1"],
              "test_loss": result["loss"], "runtime_seconds": time.perf_counter()-started,
              "complete": not args.subset and not args.eval_subset}
    atomic_json(directory / "metrics.json", report)
    torch.save({"model": model.state_dict(), "config": config}, directory / "best.pt")
    (directory / "log.txt").write_text(json.dumps(report, allow_nan=False) + "\n")
    print(json.dumps(report, allow_nan=False), flush=True)
    return directory
