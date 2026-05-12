from __future__ import annotations

from pathlib import Path
import json


DEFAULT_CONFIG = {
    "project": {"title": "Predicting TP53 Mutation Status from Gene Expression Profiles", "random_state": 42},
    "depmap": {
        "release": "DepMap Public 26Q1",
        "expression_file": "OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv",
        "mutation_file_candidates": ["OmicsSomaticMutations.csv", "CCLE_mutations.csv"],
        "metadata_file_candidates": ["Model.csv", "Models.csv", "sample_info.csv"],
    },
    "data": {
        "raw_dir": "data/raw",
        "processed_dir": "data/processed",
        "expression_processed": "data/processed/expression_matched.csv.gz",
        "labels_processed": "data/processed/tp53_labels.csv",
        "metadata_processed": "data/processed/sample_metadata.csv",
    },
    "features": {"top_variable_genes": 3000, "min_mean_expression": 0.1},
    "split": {"test_size": 0.15, "validation_size": 0.15},
}


def load_config(path: str | Path = "config/project.yaml") -> dict:
    """Load a tiny YAML-like config without requiring PyYAML."""
    path = Path(path)
    if not path.exists():
        return DEFAULT_CONFIG
    try:
        import yaml  # type: ignore

        with path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
        return _deep_update(DEFAULT_CONFIG.copy(), loaded or {})
    except Exception:
        return DEFAULT_CONFIG


def save_json(obj: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, sort_keys=True)


def _deep_update(base: dict, update: dict) -> dict:
    out = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out
