"""Generate the README result table from complete matching-seed runs only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics


ROWS = [
    ("float", "float-baseline", "FP32", "FP32"),
    ("PTQ, per-tensor", "ptq-w8a8-pt-mse", "8", "8"),
    ("PTQ, per-channel", "ptq-w8a8-pc-mse", "8", "8"),
    ("PTQ, per-channel, min/max 클립", "ptq-w4a8-pc-minmax", "4", "8"),
    ("PTQ, per-channel, MSE 클립", "ptq-w4a8-pc-mse", "4", "8"),
    ("QAT, 한 번에", "qat-w4a8-pc-mse-one-shot-uniform-nearest-fp32", "4", "8"),
    ("QAT, 단계별", "qat-w4a8-pc-mse-staged-uniform-nearest-fp32", "4", "8"),
    ("QAT, 단계별 + KD", "qat-w4a8-pc-mse-staged-uniform-nearest-fp32-kd", "4", "8"),
    ("QAT, 단계별, APoT 가중치", "qat-w4a8-pc-mse-staged-apot-nearest-fp32", "4 + sign", "8"),
    ("QAT, 단계별, 균일 31레벨 (APoT 대조군)", "qat-w5a8-pc-mse-staged-uniform-nearest-fp32", "5", "8"),
    ("QAT, 단계별, floor 활성화", "qat-w4a8-pc-mse-staged-uniform-floor-fp32", "4", "8"),
]


def collect(root: Path, seeds=(0, 1, 2)):
    reports = {}
    for _, prefix, _, _ in ROWS:
        for seed in seeds:
            path = root / f"{prefix}-s{seed}" / "metrics.json"
            if path.exists():
                reports[f"{prefix}-s{seed}"] = json.loads(path.read_text())
    return collect_reports(reports, seeds)


def collect_reports(reports: dict, seeds=(0, 1, 2)):
    """Aggregate archived or live reports with the same completeness rules."""
    values = {}
    for label, prefix, _, _ in ROWS:
        scores = []
        for seed in seeds:
            report = reports.get(f"{prefix}-s{seed}")
            if report is None:
                break
            if not report.get("complete") or report["config"]["seed"] != seed:
                break
            expected_epochs = 200 if prefix.startswith("float") else 20 if prefix.startswith("qat") else None
            if expected_epochs is not None and report["config"].get("epochs") != expected_epochs:
                break
            scores.append(report["best_top1"])
        if len(scores) == len(seeds):
            values[label] = scores
    baseline = values.get("float")
    lines = ["| 설정 | 가중치 | 활성화 | top-1 (%) | float 대비 |", "|---|---|---|---|---|"]
    for label, _, weight, activation in ROWS:
        scores = values.get(label)
        if scores:
            deviation = statistics.stdev(scores) if len(scores) > 1 else 0
            result = f"{statistics.mean(scores):.2f} ± {deviation:.2f} (n={len(scores)})"
            delta = "—" if label == "float" else (f"{statistics.mean([s-b for s, b in zip(scores, baseline)]):+.2f} pp" if baseline else "")
        else:
            result, delta = "측정 예정", "—" if label == "float" else ""
        lines.append(f"| {label} | {weight} | {activation} | {result} | {delta} |")
    return "\n".join(lines), values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source", choices=("live", "archive"), default="live")
    args = parser.parse_args()
    if args.source == "archive":
        from quant_pitfalls.results import load_results
        report, _ = load_results(Path(__file__).resolve().parents[1], "m3", "archive")
        table, _ = collect_reports(report["reports"])
    else:
        table, _ = collect(args.runs)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(table + "\n")
    print(table)


if __name__ == "__main__":
    main()
