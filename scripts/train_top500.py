from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tp53_baseline.model_training import train_top_features_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a first TP53 classifier using the baseline top-500 feature set."
    )
    parser.add_argument("--expression-csv", type=Path, default=os.environ.get("TP53_EXPRESSION_CSV"))
    parser.add_argument("--mutation-csv", type=Path, default=os.environ.get("TP53_MUTATION_CSV"))
    parser.add_argument(
        "--top-features-csv",
        type=Path,
        default=os.environ.get("TP53_TOP_FEATURES_CSV", "outputs/baseline/top_500_features.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=os.environ.get("TP53_MODEL_OUTPUT_DIR", "outputs/models/top500_logistic"),
    )
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.expression_csv is None or args.mutation_csv is None:
        print(
            "Provide --expression-csv and --mutation-csv, or set TP53_EXPRESSION_CSV and TP53_MUTATION_CSV.",
            file=sys.stderr,
        )
        return 2

    results = train_top_features_model(
        expression_csv=args.expression_csv.expanduser().resolve(),
        mutation_csv=args.mutation_csv.expanduser().resolve(),
        top_features_csv=args.top_features_csv.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        test_size=args.test_size,
        random_state=args.random_state,
    )
    print(json.dumps(results["metrics"], indent=2))
    print(f"Metrics written to: {results['metrics_path']}")
    print(f"Predictions written to: {results['predictions_path']}")
    print(f"Coefficients written to: {results['coefficients_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
