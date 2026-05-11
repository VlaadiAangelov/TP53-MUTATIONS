#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from tp53_ml.config import load_config
from tp53_ml.data import align_metadata, make_tp53_labels, read_expression, read_metadata


def _find_existing(raw_dir: Path, candidates: list[str]) -> Path:
    for candidate in candidates:
        path = raw_dir / candidate
        if path.exists():
            return path
    raise FileNotFoundError(f"None of these files exist in {raw_dir}: {candidates}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create matched expression matrix and TP53 labels.")
    parser.add_argument("--config", default="config/project.yaml")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional small subset for smoke tests.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw_dir = Path(cfg["data"]["raw_dir"])
    expression_path = raw_dir / cfg["depmap"]["expression_file"]
    mutation_path = _find_existing(raw_dir, cfg["depmap"]["mutation_file_candidates"])
    metadata_path = _find_existing(raw_dir, cfg["depmap"]["metadata_file_candidates"])

    expression = read_expression(expression_path)
    if args.max_samples:
        expression = expression.head(args.max_samples)

    mutations = pd.read_csv(mutation_path, low_memory=False)
    labels = make_tp53_labels(mutations, expression.index)
    metadata = align_metadata(read_metadata(metadata_path), expression.index)

    expression.to_csv(cfg["data"]["expression_processed"], compression="gzip")
    labels.to_csv(cfg["data"]["labels_processed"], index=False)
    metadata.to_csv(cfg["data"]["metadata_processed"], index=False)

    print(f"Expression shape: {expression.shape}")
    print(labels["tp53_mutant"].value_counts().rename(index={0: 'WT', 1: 'Mutant'}))
    print(f"Wrote {cfg['data']['expression_processed']}")
    print(f"Wrote {cfg['data']['labels_processed']}")
    print(f"Wrote {cfg['data']['metadata_processed']}")


if __name__ == "__main__":
    main()
