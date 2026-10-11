"""Sequential, resumable training experiment queue; stops on errors or weak baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from collect_results import collect


def experiments(seeds=(0, 1, 2)):
    jobs = []

    def add(name, stage, seed, flags=(), epochs=20):
        args = [stage, "--seed", str(seed), "--name", f"{name}-s{seed}"]
        if stage != "ptq":
            args += ["--epochs", str(200 if stage == "float" else epochs)]
        if stage != "float":
            args += ["--ckpt", f"runs/float-baseline-s{seed}/best.pt"]
        jobs.append({"name": f"{name}-s{seed}", "stage": stage, "seed": seed, "args": args+list(flags)})

    for seed in seeds:
        add("float-baseline", "float", seed)
    for seed in seeds:
        for bits, channel, clip in [(8, False, "mse"), (8, True, "mse"), (4, True, "minmax"), (4, True, "mse")]:
            flags = ["--w-bits", str(bits), "--clip", clip] + (["--per-channel"] if channel else [])
            add(f"ptq-w{bits}a8-{'pc' if channel else 'pt'}-{clip}", "ptq", seed, flags)
        for schedule, quantizer, rounding, kd, bits in [
            ("one-shot", "uniform", "nearest", False, 4), ("staged", "uniform", "nearest", False, 4),
            ("staged", "uniform", "nearest", True, 4), ("staged", "apot", "nearest", False, 4),
            ("staged", "uniform", "nearest", False, 5), ("staged", "uniform", "floor", False, 4),
        ]:
            name = f"qat-w{bits}a8-pc-mse-{schedule}-{quantizer}-{rounding}-fp32" + ("-kd" if kd else "")
            flags = ["--per-channel", "--w-bits", str(bits), "--schedule", schedule,
                     "--weight-quantizer", quantizer, "--rounding", rounding]
            if kd:
                flags.append("--kd")
            if schedule == "staged" and quantizer == "uniform" and rounding == "nearest" and not kd and bits == 4:
                flags.append("--track-updates")
            add(name, "qat", seed, flags)
    for seed in seeds:
        for dtype in ("fp32", "fp16"):
            for optimizer in ("sgd", "adam"):
                for lr in ("1e-4", "1e-5", "1e-6"):
                    add(f"nb06-{dtype}-{optimizer}-lr{lr}", "qat", seed,
                        ["--per-channel", "--master-dtype", dtype, "--optimizer", optimizer, "--lr", lr,
                         "--schedule", "one-shot", "--freeze-bn-from", "0", "--freeze-observer-from", "0", "--track-updates"], epochs=1)
        for channel, order in [(False, "fold"), (False, "raw"), (True, "raw")]:
            add(f"nb04-{'pc' if channel else 'pt'}-{order}", "qat", seed,
                ["--fold-order", order] + (["--per-channel"] if channel else []))
        for method in ("warmup", "both", "quant-only"):
            flags = ["--per-channel", "--skip-calibration", "--track-diagnostics"]
            if method == "warmup":
                flags += ["--weight-quant-from", "0.05", "--act-quant-from", "0.05"]
            else:
                flags += ["--schedule", "one-shot"]
            if method == "quant-only":
                flags += ["--freeze-observer-from", "0"]
            add(f"nb05-{method}", "qat", seed, flags)
        for batch in (32, 128):
            for frozen in ("0", "0.5", "1"):
                if batch == 128 and frozen == "0.5":
                    continue  # Reuse the matching staged result from the main table.
                add(f"nb07-bs{batch}-freeze{frozen}", "qat", seed,
                    ["--per-channel", "--batch-size", str(batch), "--freeze-bn-from", frozen, "--track-updates"])
        add("nb08-delayed", "qat", seed, ["--per-channel", "--weight-quant-from", "0.1", "--act-quant-from", "0.2"])
    return jobs


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--min-float-top1", type=float, default=92)
    parser.add_argument("--max-hours", type=float, default=48)
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--wait-for-existing", action="store_true", help="wait for the current queue before starting")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    directory = root/"runs/m3"
    directory.mkdir(parents=True, exist_ok=True)
    jobs = experiments()
    if args.baseline_only:
        jobs = [job for job in jobs if job["stage"] == "float"]
    if args.dry_run:
        write_json(directory/"manifest.json", jobs)
        print(json.dumps({"runs": len(jobs), "float_epochs": 600,
                          "qat_epochs": sum(int(job["args"][job["args"].index("--epochs")+1]) for job in jobs if job["stage"] == "qat")}, indent=2))
        return
    if args.wait_for_existing:
        existing_status = directory/"status.json"
        while existing_status.exists():
            previous = json.loads(existing_status.read_text())
            if previous["state"] != "running":
                if previous["state"] != "complete":
                    raise RuntimeError("Previous queue needs review before continuing")
                break
            time.sleep(5)
    write_json(directory/"manifest.json", jobs)
    started = time.time()
    status = {"state": "running", "started_at": started, "total_jobs": len(jobs), "completed_jobs": 0}
    for index, job in enumerate(jobs):
        status.update(current=job["name"], job_index=index)
        write_json(directory/"status.json", status)
        metrics_path = root/"runs"/job["name"]/"metrics.json"
        existing = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
        if not existing.get("complete"):
            if time.time()-started > args.max_hours*3600:
                status.update(state="stopped", reason="Elapsed time limit exceeded")
                write_json(directory/"status.json", status)
                return
            command = job["args"].copy()
            checkpoint = root/"runs"/job["name"]/"last.pt"
            if checkpoint.exists():
                command += ["--resume", str(checkpoint.relative_to(root))]
            result = subprocess.run([sys.executable, "scripts/run_job.py", "--status-dir", f"runs/m3/jobs/{job['name']}", "--", *command], cwd=root)
            if result.returncode:
                status.update(state="failed", reason="Training job failed", exit_code=result.returncode)
                write_json(directory/"status.json", status)
                return
        report = json.loads(metrics_path.read_text())
        if not report.get("complete"):
            status.update(state="failed", reason="Run is incomplete")
            write_json(directory/"status.json", status)
            return
        if job["stage"] == "float" and report["best_top1"] < args.min_float_top1:
            status.update(state="needs_review", reason="Float accuracy below recipe review threshold", top1=report["best_top1"])
            write_json(directory/"status.json", status)
            return
        status["completed_jobs"] = index+1
        table, values = collect(root/"runs")
        (directory/"results.md").write_text(table+"\n")
        write_json(directory/"results.json", values)
    if not args.baseline_only:
        status.update(state="postprocessing", current="notebook analyses")
        write_json(directory/"status.json", status)
        result = subprocess.run([sys.executable, "scripts/finish_m3.py"], cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        (directory/"postprocessing.log").write_text(result.stdout.replace(str(root)+"/", ""))
        if result.returncode:
            status.update(state="failed", reason="Postprocessing failed", exit_code=result.returncode)
            write_json(directory/"status.json", status)
            return
    status.update(state="complete", runtime_seconds=time.time()-started)
    write_json(directory/"status.json", status)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        message = str(error).replace(str(Path(__file__).resolve().parents[1])+"/", "")
        print(f"Queue stopped: {type(error).__name__}: {message}", file=sys.stderr, flush=True)
        raise SystemExit(1)
