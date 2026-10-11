"""Run tests, all notebooks, figures, and result collection in one pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from collect_results import ROWS, collect_reports
from quant_pitfalls.results import load_results
from run_m3 import experiments


FIGURES = (
    "01_clipping-mse", "02_floor-rounding", "02_floor-layer-bias",
    "03_apot-vs-uniform", "03_weight-kurtosis", "04_bn-folding",
    "04_bn-folding-trained", "05_observe-before-quant", "05_observer-training",
    "06_fp16-master-weights", "07_bn-stats-in-qat", "07_bn-freeze-training",
    "08_ptq-vs-qat", "09_accumulator-saturation",
)


def checkpoint_preflight(root: Path) -> None:
    """Reject incomplete checkpoint mode before launching any evaluations."""
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Checkpoint analyses currently require CUDA; use --from-results for CPU rendering")
    missing = []
    for job in experiments():
        metrics = root / "runs" / job["name"] / "metrics.json"
        if not metrics.exists() or not json.loads(metrics.read_text()).get("complete"):
            missing.append(f"runs/{job['name']}/metrics.json (complete)")
        if job["name"].startswith("nb05-") and not (metrics.parent / "diagnostics.json").exists():
            missing.append(f"runs/{job['name']}/diagnostics.json")
    for seed in (0, 1, 2):
        for prefix in ("float-baseline", "nb04-pt-fold", "nb04-pt-raw", "nb04-pc-raw",
                       "qat-w4a8-pc-mse-staged-uniform-nearest-fp32"):
            relative = f"runs/{prefix}-s{seed}/best.pt"
            if not (root / relative).exists():
                missing.append(relative)
    if missing:
        raise RuntimeError(f"Checkpoint mode needs {len(missing)} missing inputs; first: {missing[0]}. Use --from-results for archived rendering")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--from-results", action="store_true", help="default: render recorded results without data/checkpoints/training")
    modes.add_argument("--skip-training", action="store_true", help="recompute analyses from complete runs and existing checkpoints")
    modes.add_argument("--train", action="store_true", help="run/resume training, then integer evaluation (may take many hours)")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    mode = "train" if args.train else "checkpoints" if args.skip_training else "archive"
    directory = root / "runs/reproduce"
    directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    environment = os.environ.copy()
    environment["QUANT_PITFALLS_RESULT_SOURCE"] = "archive" if mode == "archive" else "live"
    environment["MPLCONFIGDIR"] = str(directory / ".matplotlib")
    replacements = ((str(root), "<repo>"), (str(Path.home()), "<home>"))
    summary = {"state": "running", "mode": mode, "training_requested": args.train,
               "result_source": environment["QUANT_PITFALLS_RESULT_SOURCE"]}

    def save() -> None:
        (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    with (directory / "log.txt").open("w") as log:
        def write(line: str) -> None:
            for original, replacement in replacements:
                line = line.replace(original, replacement)
            print(line, end="", flush=True)
            log.write(line)
            log.flush()

        def run(label: str, arguments: list[str]) -> None:
            write(f"[{label}]\n")
            process = subprocess.Popen([sys.executable, *arguments], cwd=root, env=environment,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in process.stdout:
                write(line)
            if process.wait():
                raise RuntimeError(f"{label} failed; see runs/reproduce/log.txt")

        save()
        try:
            if mode == "archive":
                for milestone in ("m3", "m4"):
                    load_results(root, milestone, "archive")
                write("Rendering recorded training and integer measurements; this does not reevaluate accuracy.\n")
            elif mode == "checkpoints":
                checkpoint_preflight(root)
            run("tests", ["-m", "pytest", "-q", "--junitxml=runs/reproduce/pytest.xml"])
            if mode == "train":
                run("training and analyses", ["scripts/run_m3.py"])
                status = json.loads((root / "runs/m3/status.json").read_text())
                if status.get("state") != "complete":
                    raise RuntimeError("Training queue stopped before completion; review runs/m3/status.json")
            elif mode == "checkpoints":
                run("checkpoint analyses", ["scripts/finish_m3.py", "--skip-notebooks"])
            if mode != "archive":
                run("integer evaluation (matching cached settings may be reused)", ["scripts/run_m4.py", "--batch-size", "32"])
            report, m3_source = load_results(root, "m3", environment["QUANT_PITFALLS_RESULT_SOURCE"])
            m4, m4_source = load_results(root, "m4", environment["QUANT_PITFALLS_RESULT_SOURCE"])
            table, values = collect_reports(report["reports"])
            if report.get("state") != "complete" or m4.get("state") != "complete" or len(values) != len(ROWS):
                raise RuntimeError("Complete training/integer measurements and all three table seeds are required")
            notebooks = sorted(root.glob("notebooks/[0-9][0-9]_*.ipynb"))
            if len(notebooks) != 9:
                raise RuntimeError("Expected all nine experiment notebooks")
            run("notebooks and figures", ["-m", "nbconvert", "--to", "notebook", "--execute", "--inplace",
                                         "--ExecutePreprocessor.timeout=1800", *[str(p.relative_to(root)) for p in notebooks]])
            (directory / "results.md").write_text(table + "\n")
            write("[result table]\n" + table + "\n")
            figure_hashes = {}
            for name in FIGURES:
                relative = f"docs/figures/{name}.svg"
                path = root / relative
                if not path.exists() or path.stat().st_mtime < time.time() - (time.perf_counter() - started) - 2:
                    raise RuntimeError(f"Expected a newly generated figure: {relative}")
                figure_hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
            summary.update(state="complete", notebooks=len(notebooks), figures=figure_hashes,
                           sources={"m3": m3_source, "m4": m4_source}, table_rows=len(values))
        except Exception as error:
            summary.update(state="failed", error=type(error).__name__)
            write(f"Stopped: {error}\n")
            raise SystemExit(1)
        finally:
            summary["runtime_seconds"] = time.perf_counter() - started
            save()


if __name__ == "__main__":
    main()
