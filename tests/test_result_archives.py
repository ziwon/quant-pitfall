"""Protect recorded result provenance and avoid silent historical fallback."""

import json
from pathlib import Path
import runpy
import shutil

import pytest

from quant_pitfalls.results import load_results


ROOT = Path(__file__).resolve().parents[1]


def test_archived_measurements_are_complete_and_table_matches_reports():
    collect = runpy.run_path(str(ROOT / "scripts/collect_results.py"))["collect_reports"]
    m3, source = load_results(ROOT, "m3", "archive")
    assert source == "docs/results/m3.json.gz"
    assert len(m3["reports"]) == 105
    assert all(report["complete"] for report in m3["reports"].values())
    table, values = collect(m3["reports"])
    assert values == m3["table"]
    assert "92.40 ± 0.12 (n=3)" in table
    m4, _ = load_results(ROOT, "m4", "archive")
    assert len(m4["sweep"]) == 12
    assert m4["parity"]["fixed_point_mismatches"] == 0
    for result in (*m4["sweep"].values(), *m4["full"].values()):
        assert result["top1"] == pytest.approx(100 * result["correct"] / result["images"])


def test_archive_checksum_rejects_modified_measurements(tmp_path):
    directory = tmp_path / "docs/results"
    directory.mkdir(parents=True)
    shutil.copy(ROOT / "docs/results/manifest.json", directory)
    blob = bytearray((ROOT / "docs/results/m4.json.gz").read_bytes())
    blob[len(blob) // 2] ^= 1
    (directory / "m4.json.gz").write_bytes(blob)
    with pytest.raises(ValueError, match="Archive checksum mismatch"):
        load_results(tmp_path, "m4", "archive")


def test_explicit_live_source_does_not_substitute_archived_accuracy():
    with pytest.raises(FileNotFoundError, match="Missing live measurements"):
        load_results(ROOT / "tests", "m4", "live")


def test_auto_keeps_incomplete_local_measurements_visible(tmp_path, monkeypatch):
    directory = tmp_path / "runs/m3"
    directory.mkdir(parents=True)
    (directory / "notebook-data.json").write_text(json.dumps({"state": "running"}))
    monkeypatch.setenv("QUANT_PITFALLS_RESULT_SOURCE", "auto")
    report, source = load_results(tmp_path, "m3")
    assert report["state"] == "running"
    assert source == "runs/m3/notebook-data.json"



@pytest.mark.parametrize("invalid_config", [{"seed": 1, "epochs": 200}, {"seed": 0, "epochs": 1}])
def test_archived_table_rejects_wrong_seed_or_smoke_epochs(invalid_config):
    collect = runpy.run_path(str(ROOT / "scripts/collect_results.py"))["collect_reports"]
    reports = {f"float-baseline-s{seed}": {"complete": True, "config": {"seed": seed, "epochs": 200},
                                          "best_top1": 90 + seed} for seed in (0, 1, 2)}
    reports["float-baseline-s0"]["config"] = invalid_config
    table, values = collect(reports)
    assert "float" not in values and "측정 예정" in table
