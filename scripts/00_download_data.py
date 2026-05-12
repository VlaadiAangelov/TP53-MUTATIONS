#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tp53_ml.config import load_config
from tp53_ml.data import download_depmap_file, try_download_candidates


def main() -> None:
    parser = argparse.ArgumentParser(description="Download DepMap/CCLE files for the TP53 project.")
    parser.add_argument("--config", default="config/project.yaml")
    parser.add_argument("--release", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    release = args.release or cfg["depmap"]["release"]
    raw_dir = cfg["data"]["raw_dir"]

    expression = download_depmap_file(release, cfg["depmap"]["expression_file"], raw_dir)
    mutation = try_download_candidates(release, cfg["depmap"]["mutation_file_candidates"], raw_dir)
    metadata = try_download_candidates(release, cfg["depmap"]["metadata_file_candidates"], raw_dir)

    print(f"Downloaded expression: {expression}")
    print(f"Downloaded mutations:  {mutation}")
    print(f"Downloaded metadata:   {metadata}")


if __name__ == "__main__":
    main()
