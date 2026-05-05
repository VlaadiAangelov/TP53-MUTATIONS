from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Union

from .config import BaselineConfig
from .data import (
    TARGET_COLUMN,
    build_labeled_dataset,
    get_numeric_feature_matrix,
    get_tp53_related_columns,
    load_expression_data,
    load_mutation_data,
    summarize_dataset,
)
from .feature_selection import compute_feature_ranking, exclude_columns_by_name, select_top_features


def run_baseline(config: BaselineConfig) -> Dict[str, Union[Path, Dict[str, Any]]]:
    config.output_dir.mkdir(parents=True, exist_ok=True)

    expression_df = load_expression_data(config.expression_csv)
    mutation_df = load_mutation_data(config.mutation_csv)
    dataset = build_labeled_dataset(expression_df, mutation_df)
    numeric_features = get_numeric_feature_matrix(dataset)
    excluded_columns = get_tp53_related_columns(list(numeric_features.columns), config.exclude_prefix)
    filtered_features = exclude_columns_by_name(numeric_features, excluded_columns)
    ranking_df = compute_feature_ranking(filtered_features, dataset[TARGET_COLUMN])
    top_features_df = select_top_features(ranking_df, config.top_n)

    summary = summarize_dataset(
        expression_df=expression_df,
        mutation_df=mutation_df,
        dataset=dataset,
        numeric_features=numeric_features,
        filtered_features=filtered_features,
        excluded_columns=excluded_columns,
    )

    summary_path = config.output_dir / "dataset_summary.json"
    ranking_path = config.output_dir / "feature_ranking.csv"
    top_features_path = config.output_dir / f"top_{config.top_n}_features.csv"
    excluded_columns_path = config.output_dir / "excluded_tp53_related_features.txt"

    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    ranking_df.to_csv(ranking_path, index=False)
    top_features_df.to_csv(top_features_path, index=False)
    excluded_columns_path.write_text("\n".join(excluded_columns) + ("\n" if excluded_columns else ""), encoding="utf-8")

    return {
        "summary": summary,
        "summary_path": summary_path,
        "ranking_path": ranking_path,
        "top_features_path": top_features_path,
        "excluded_columns_path": excluded_columns_path,
    }
