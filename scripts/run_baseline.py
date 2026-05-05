from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tp53_baseline import BaselineConfig, run_baseline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the TP53 mutation baseline feature-selection workflow."
    )
    parser.add_argument("--expression-csv", type=Path, help="Path to CCLE expression CSV.")
    parser.add_argument("--mutation-csv", type=Path, help="Path to CCLE mutation CSV.")
    parser.add_argument("--output-dir", type=Path, help="Directory for generated outputs.")
    parser.add_argument(
        "--top-n",
        type=int,
        help="Number of top-ranked features to save.",
    )
    parser.add_argument(
        "--exclude-prefix",
        type=str,
        help="Case-insensitive string used to drop TP53-related expression columns.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> BaselineConfig:
    env_config = None
    try:
        env_config = BaselineConfig.from_env()
    except ValueError:
        env_config = None

    expression_csv = args.expression_csv
    mutation_csv = args.mutation_csv
    output_dir = args.output_dir
    top_n = args.top_n
    exclude_prefix = args.exclude_prefix

    if env_config is not None:
        expression_csv = expression_csv or env_config.expression_csv
        mutation_csv = mutation_csv or env_config.mutation_csv
        output_dir = output_dir or env_config.output_dir
        top_n = top_n or env_config.top_n
        exclude_prefix = exclude_prefix or env_config.exclude_prefix

    if expression_csv is None or mutation_csv is None:
        raise ValueError(
            "Provide expression and mutation CSV paths with env vars or command-line arguments."
        )

    return BaselineConfig(
        expression_csv=expression_csv.expanduser().resolve(),
        mutation_csv=mutation_csv.expanduser().resolve(),
        output_dir=(output_dir or Path("outputs/baseline")).expanduser().resolve(),
        top_n=top_n or 500,
        exclude_prefix=exclude_prefix or "TP53",
    )


def main() -> int:
    args = parse_args()
    try:
        config = build_config(args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    results = run_baseline(config)
    print(json.dumps(results["summary"], indent=2))
    print(f"Summary written to: {results['summary_path']}")
    print(f"Ranking written to: {results['ranking_path']}")
    print(f"Top features written to: {results['top_features_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
