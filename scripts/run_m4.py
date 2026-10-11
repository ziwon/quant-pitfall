"""Measure fixed-point parity and accumulator overflow on a frozen M3 checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from quant_pitfalls.experiments import dataset_options, load_qat
from quant_pitfalls.quant.fake_quant import FakeQuantize
from quant_pitfalls.sim.accumulator import AccumulatorSpec
from quant_pitfalls.sim.integer import IntegerModel
from quant_pitfalls.training import data_loaders, seed_everything


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


@torch.no_grad()
def compare_paths(model, images):
    """Compare every integer boundary and independently convolved reference."""
    runner = IntegerModel(model, capture=True)
    logits = runner(images)
    reference = IntegerModel(model, reference=True, capture=True)
    reference_logits = reference(images)
    parity = []
    for name, codes in runner.trace.items():
        target = reference.trace[name]
        parity.append({"layer": name, "elements": codes.numel(), "mismatches": int((codes != target).sum())})
    if any(row["mismatches"] for row in parity) or not torch.equal(logits, reference_logits):
        raise RuntimeError("Integer and fixed-point fake-quant reference disagree")
    original = {}
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, FakeQuantize) and getattr(module, "quant_role", "activation") == "activation":
            def capture(operation, inputs, output, name=name):
                scale = (operation.alpha / operation.spec.qmax).double()
                original[name] = torch.floor(output.double() / scale + 0.5).to(torch.int64)
            handles.append(module.register_forward_hook(capture))
    try:
        original_logits = model(images)
    finally:
        for handle in handles:
            handle.remove()
    differences = [{"layer": name, "elements": codes.numel(),
                    "mismatches": int((codes != runner.trace[name]).sum())} for name, codes in original.items()]
    return {"images": len(images), "fixed_point_layers": parity,
            "fixed_point_mismatches": sum(row["mismatches"] for row in parity),
            "original_fake_quant_layers": differences,
            "original_fake_quant_mismatches": sum(row["mismatches"] for row in differences),
            "original_logit_max_abs_difference": float((logits-original_logits).abs().max()),
            "original_prediction_disagreements": int((logits.argmax(1) != original_logits.argmax(1)).sum())}


@torch.no_grad()
def evaluate(runner, loader, device, *, integer=False, status=None):
    started = time.perf_counter()
    correct, count, layers = 0, 0, {}
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = runner(images)
        correct += int((logits.argmax(1) == labels).sum())
        count += len(labels)
        if integer:
            for name, row in runner.stats.items():
                total = layers.setdefault(name, {**row, "outputs": 0, "overflow_outputs": 0,
                                                "max_abs_sum": 0, "max_abs_product_sum": 0})
                total["outputs"] += row["outputs"]
                total["overflow_outputs"] += row["overflow_outputs"]
                for key in ("max_abs_sum", "max_abs_product_sum", "required_bits", "required_bits_with_bias"):
                    total[key] = max(total[key], row[key])
        if status is not None:
            status(count)
    sync(device)
    for row in layers.values():
        row["overflow_pct"] = 100 * row["overflow_outputs"] / row["outputs"]
    return {"images": count, "correct": correct, "top1": 100 * correct/count,
            "runtime_seconds": time.perf_counter()-started, "layers": layers}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, default=Path("runs/qat-w4a8-pc-mse-staged-uniform-nearest-fp32-s0/best.pt"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--subset", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--full-bits", type=int, nargs="*", default=[16, 32])
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument("--max-hours", type=float, default=48)
    args = parser.parse_args()
    if not 1 <= args.subset <= 10000 or args.batch_size < 1 or args.ckpt.is_absolute():
        parser.error("Use a relative checkpoint path, positive batch size and subset in [1, 10000]")
    if any(bits not in (12, 14, 16, 18, 24, 32) for bits in args.full_bits):
        parser.error("full-bits must be selected from the sweep widths")
    seed_everything(args.seed)
    torch.set_num_threads(4)
    if str(args.device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    directory = Path("runs/m4")
    directory.mkdir(parents=True, exist_ok=True)
    model = load_qat(args.ckpt, args.device)
    options = dataset_options(args.seed, args.device)
    options.batch_size, options.workers, options.eval_subset = args.batch_size, args.workers, args.subset
    _, subset_loader, _ = data_loaders(options, calibration=True)
    images, _ = next(iter(subset_loader))
    images = images.to(args.device)
    parity = compare_paths(model, images)
    write_json(directory/"parity.json", parity)
    smoke = {}
    for bits in (32, 12):
        runner = IntegerModel(model, acc=AccumulatorSpec(bits), collect_stats=True)
        started = time.perf_counter()
        runner(images)
        sync(args.device)
        smoke[str(bits)] = {"images": len(images), "runtime_seconds": time.perf_counter()-started}
    slow_per_image = max(row["runtime_seconds"]/row["images"] for row in smoke.values())
    estimate = {"smoke": smoke, "sweep_images": 12*args.subset,
                "full_images": 10000*len(args.full_bits),
                "estimated_hours_with_margin": slow_per_image*(12*args.subset + 10000*len(args.full_bits))*1.3/3600,
                "peak_memory_gib": torch.cuda.max_memory_allocated()/2**30 if str(args.device).startswith("cuda") else None}
    write_json(directory/"estimate.json", estimate)
    print(json.dumps({"fixed_point_mismatches": parity["fixed_point_mismatches"], "estimate": estimate}), flush=True)
    if args.estimate_only:
        return
    if estimate["estimated_hours_with_margin"] > args.max_hours:
        raise RuntimeError("Runtime estimate exceeds the execution limit; reduce the experiment first")
    sources = [Path(__file__), Path("src/quant_pitfalls/sim/integer.py"), Path("src/quant_pitfalls/sim/accumulator.py")]
    revision = hashlib.sha256(b"".join(path.read_bytes() for path in sources)).hexdigest()
    config = {"checkpoint": str(args.ckpt), "checkpoint_sha256": hashlib.sha256(args.ckpt.read_bytes()).hexdigest(),
              "seed": args.seed, "subset": args.subset, "batch_size": args.batch_size,
              "full_bits": args.full_bits, "implementation_sha256": revision}
    path = directory/"results.json"
    previous = json.loads(path.read_text()) if path.exists() else {}
    data = previous if previous.get("config") == config else {
        "state": "running", "config": config, "seeds": [args.seed], "parity": parity,
        "environment": {"torch": torch.__version__, "gpu": torch.cuda.get_device_name() if str(args.device).startswith("cuda") else None},
        "estimate": estimate, "sweep": {}, "full": {}}
    status = {"state": "running", "started_at": time.time(), "current": "fake-quant subset"}

    def progress(count):
        status["images_completed"] = count
        write_json(directory/"status.json", status)

    if "fake_quant_subset" not in data:
        data["fake_quant_subset"] = evaluate(model, subset_loader, args.device, status=progress)
        write_json(path, data)
    for bits in (12, 14, 16, 18, 24, 32):
        for mode in ("saturate", "wrap"):
            key = f"{bits}-{mode}"
            if key in data["sweep"]:
                continue
            status.update(current=key, images_completed=0)
            runner = IntegerModel(model, acc=AccumulatorSpec(bits, mode), collect_stats=True)
            data["sweep"][key] = evaluate(runner, subset_loader, args.device, integer=True, status=progress)
            write_json(path, data)
            print(key, "top1:", data["sweep"][key]["top1"], "seconds:", data["sweep"][key]["runtime_seconds"], flush=True)
    options.eval_subset = 0
    _, full_loader, _ = data_loaders(options, calibration=True)
    if "fake_quant_full" not in data:
        status.update(current="fake-quant full", images_completed=0)
        data["fake_quant_full"] = evaluate(model, full_loader, args.device, status=progress)
        write_json(path, data)
    for bits in args.full_bits:
        key = f"{bits}-saturate"
        if key in data["full"]:
            continue
        status.update(current=f"full-{key}", images_completed=0)
        runner = IntegerModel(model, acc=AccumulatorSpec(bits), collect_stats=False)
        data["full"][key] = evaluate(runner, full_loader, args.device, status=progress)
        write_json(path, data)
        print("full", key, "top1:", data["full"][key]["top1"], "seconds:", data["full"][key]["runtime_seconds"], flush=True)
    data.update(state="complete", runtime_seconds=time.time()-status["started_at"])
    write_json(path, data)
    status.update(state="complete", runtime_seconds=data["runtime_seconds"])
    write_json(directory/"status.json", status)


if __name__ == "__main__":
    main()
