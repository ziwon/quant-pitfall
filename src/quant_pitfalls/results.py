"""Load live measurements or hash-verified, portable numerical archives."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
from typing import Any


LIVE_PATHS = {"m3": "runs/m3/notebook-data.json", "m4": "runs/m4/results.json"}


def load_results(root: Path, milestone: str, source: str | None = None) -> tuple[dict[str, Any], str]:
    """Return measurements and their relative source path.

    ``auto`` prefers a local run, then the published archive. ``live`` never
    falls back to historical measurements. Archives contain recorded results;
    loading them does not evaluate checkpoints or rerun training. Their SHA-256
    hashes detect accidental corruption, rather than authenticate a publisher.
    The environment variable QUANT_PITFALLS_RESULT_SOURCE selects the default.
    """
    source = source or os.environ.get("QUANT_PITFALLS_RESULT_SOURCE", "auto")
    if source not in {"auto", "archive", "live"} or milestone not in LIVE_PATHS:
        raise ValueError("Expected source auto/archive/live and milestone m3/m4")
    live = root / LIVE_PATHS[milestone]
    if source == "live" or (source == "auto" and live.exists()):
        if not live.exists():
            raise FileNotFoundError(f"Missing live measurements: {LIVE_PATHS[milestone]}")
        return json.loads(live.read_text()), LIVE_PATHS[milestone]
    relative = f"docs/results/{milestone}.json.gz"
    manifest = json.loads((root / "docs/results/manifest.json").read_text())
    data = (root / relative).read_bytes()
    expected = manifest["archives"][milestone]
    if hashlib.sha256(data).hexdigest() != expected["sha256"]:
        raise ValueError(f"Archive checksum mismatch: {relative}")
    decoded = gzip.decompress(data)
    if hashlib.sha256(decoded).hexdigest() != expected["json_sha256"]:
        raise ValueError(f"Decoded checksum mismatch: {relative}")
    report = json.loads(decoded)
    if report.get("state") != "complete":
        raise ValueError(f"Archive is incomplete: {relative}")
    return report, relative
