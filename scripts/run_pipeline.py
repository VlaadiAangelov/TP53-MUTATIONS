#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys


STEPS = [
    ("download", ["scripts/00_download_data.py"]),
    ("build", ["scripts/01_build_dataset.py"]),
    ("eda", ["scripts/02_eda.py"]),
    ("binary", ["scripts/03_train_binary.py"]),
    ("multiclass", ["scripts/04_train_multiclass.py"]),
    ("interpret", ["scripts/05_interpret.py"]),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full TP53 project pipeline.")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--smoke-test", action="store_true", help="Use a tiny sample for code testing, not final results.")
    args = parser.parse_args()

    for name, cmd in STEPS:
        if args.skip_download and name == "download":
            continue
        full_cmd = [sys.executable, *cmd]
        if args.smoke_test and name == "build":
            full_cmd += ["--max-samples", "120"]
        print(f"\n=== {name} ===")
        subprocess.run(full_cmd, check=True)


if __name__ == "__main__":
    main()

