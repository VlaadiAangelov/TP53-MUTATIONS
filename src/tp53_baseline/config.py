from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class BaselineConfig:
    expression_csv: Path
    mutation_csv: Path
    output_dir: Path
    top_n: int = 500
    exclude_prefix: str = "TP53"

    @classmethod
    def from_env(cls) -> "BaselineConfig":
        expression_csv = os.environ.get("TP53_EXPRESSION_CSV")
        mutation_csv = os.environ.get("TP53_MUTATION_CSV")
        output_dir = os.environ.get("TP53_OUTPUT_DIR", "outputs/baseline")
        top_n = int(os.environ.get("TP53_TOP_N", "500"))
        exclude_prefix = os.environ.get("TP53_EXCLUDE_PREFIX", "TP53")

        if not expression_csv or not mutation_csv:
            raise ValueError(
                "Set TP53_EXPRESSION_CSV and TP53_MUTATION_CSV or pass explicit paths on the command line."
            )

        return cls(
            expression_csv=Path(expression_csv).expanduser().resolve(),
            mutation_csv=Path(mutation_csv).expanduser().resolve(),
            output_dir=Path(output_dir).expanduser().resolve(),
            top_n=top_n,
            exclude_prefix=exclude_prefix,
        )
