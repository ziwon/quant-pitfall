"""Run a training CLI job and save a sanitized process log and exit status."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status-dir", type=Path, required=True)
    parser.add_argument("training_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.training_args
    if command and command[0] == "--":
        command = command[1:]
    root = Path(__file__).resolve().parents[1]
    directory = args.status_dir
    directory.mkdir(parents=True, exist_ok=True)
    started = time.time()
    status = {"state": "running", "started_at": started, "command": ["train.py", *command]}
    status_path = directory / "status.json"
    status_path.write_text(json.dumps(status, indent=2) + "\n")
    process = subprocess.Popen([sys.executable, "-u", "train.py", *command], cwd=root,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    with (directory / "process.log").open("a") as log:
        for line in process.stdout:
            cleaned = line.replace(str(root) + "/", "").replace(str(root), ".")
            cleaned = re.sub(r'(?<![A-Za-z0-9_.])/(?:__w|usr|opt|lib|tmp|home|Users)/[^\s():"]+', '<system-path>', cleaned)
            log.write(cleaned)
            log.flush()
    code = process.wait()
    status.update(state="complete" if code == 0 else "failed", exit_code=code, runtime_seconds=time.time()-started)
    status_path.write_text(json.dumps(status, indent=2) + "\n")
    sys.exit(code)


if __name__ == "__main__":
    main()
