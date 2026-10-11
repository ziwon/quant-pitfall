"""Collect finished training runs, compute checkpoint analyses and execute notebooks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from collect_results import collect
from run_m3 import experiments

from quant_pitfalls.experiments import floor_bias_comparison, fold_order_comparison, weight_distributions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-notebooks", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    directory = root/"runs/m3"
    directory.mkdir(parents=True, exist_ok=True)
    reports = {}
    for job in experiments():
        path = root/"runs"/job["name"]/"metrics.json"
        if not path.exists():
            raise RuntimeError("Checkpoint postprocessing needs every planned run")
        report = json.loads(path.read_text())
        if not report.get("complete"):
            raise RuntimeError("Checkpoint postprocessing found an incomplete run")
        reports[job["name"]] = report
        diagnostics = path.parent/"diagnostics.json"
        if diagnostics.exists():
            report["diagnostics"] = json.loads(diagnostics.read_text())
    table, values = collect(root/"runs")
    (directory/"results.md").write_text(table+"\n")
    data = {"state": "complete", "seeds": [0, 1, 2], "reports": reports, "table": values,
            "floor_bias": floor_bias_comparison(root), "weight_distributions": weight_distributions(root),
            "fold_order": fold_order_comparison(root)}
    temporary = directory/"notebook-data.tmp"
    temporary.write_text(json.dumps(data, allow_nan=False) + "\n")
    temporary.replace(directory/"notebook-data.json")
    notebooks = [f"notebooks/{name}.ipynb" for name in (
        "02_floor-rounding", "03_apot-vs-uniform", "04_bn-folding", "05_observe-before-quant",
        "06_fp16-master-weights", "07_bn-stats-in-qat", "08_ptq-vs-qat")]
    if not args.skip_notebooks:
        subprocess.run([sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute", "--inplace",
                        "--ExecutePreprocessor.timeout=1800", *notebooks], cwd=root, check=True)


if __name__ == "__main__":
    main()
