import json
from pathlib import Path
import runpy


def test_experiment_queue_has_unique_valid_commands(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root/"scripts"))
    experiments = runpy.run_path(str(root/"scripts/run_m3.py"))["experiments"]
    parser = runpy.run_path(str(root/"train.py"))["build_parser"]()
    jobs = experiments()
    assert len(jobs) == len({job["name"] for job in jobs}) == 105
    arguments = [parser.parse_args(job["args"]) for job in jobs]
    assert sum(args.stage == "float" for args in arguments) == 3
    assert sum(args.stage == "ptq" for args in arguments) == 12
    assert sum(args.epochs for args in arguments if args.stage == "qat") == 1116


def test_result_collection_requires_complete_seed_set_and_paired_baseline(tmp_path):
    collect = runpy.run_path(str(Path(__file__).resolve().parents[1]/"scripts/collect_results.py"))["collect"]
    for seed in (0, 1, 2):
        directory = tmp_path/f"float-baseline-s{seed}"
        directory.mkdir()
        (directory/"metrics.json").write_text(json.dumps({"complete":True,"config":{"seed":seed,"epochs":200},"best_top1":93+seed}))
    prefix = "ptq-w8a8-pc-mse"
    for seed in (0, 1, 2):
        directory = tmp_path/f"{prefix}-s{seed}"
        directory.mkdir()
        (directory/"metrics.json").write_text(json.dumps({"complete":seed!=2,"config":{"seed":seed},"best_top1":92+seed}))
    table, values = collect(tmp_path)
    assert values["float"] == [93,94,95]
    assert "PTQ, per-channel" not in values
    path = tmp_path/f"{prefix}-s2"/"metrics.json"
    result = json.loads(path.read_text())
    result["complete"] = True
    path.write_text(json.dumps(result))
    table, values = collect(tmp_path)
    assert "94.00 ± 1.00 (n=3)" in table
    assert "93.00 ± 1.00 (n=3)" in table and "-1.00 pp" in table
