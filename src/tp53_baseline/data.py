from __future__ import annotations

from pathlib import Path
from typing import Dict, Union

import pandas as pd

ID_COLUMN = "DepMap_ID"
TARGET_COLUMN = "TP53_status"


def load_expression_data(path: Path) -> pd.DataFrame:
    expression_df = pd.read_csv(path)
    if "Unnamed: 0" in expression_df.columns:
        expression_df = expression_df.rename(columns={"Unnamed: 0": ID_COLUMN})
    if ID_COLUMN not in expression_df.columns:
        raise ValueError(f"Expression data must include '{ID_COLUMN}'.")
    return expression_df


def load_mutation_data(path: Path) -> pd.DataFrame:
    use_columns = ["Hugo_Symbol", ID_COLUMN]
    mutation_df = pd.read_csv(path, usecols=use_columns)
    return mutation_df


def build_tp53_labels(mutation_df: pd.DataFrame) -> pd.DataFrame:
    tp53_positive = (
        mutation_df.loc[mutation_df["Hugo_Symbol"] == "TP53", [ID_COLUMN]]
        .drop_duplicates()
        .assign(**{TARGET_COLUMN: 1})
    )
    return tp53_positive


def build_labeled_dataset(expression_df: pd.DataFrame, mutation_df: pd.DataFrame) -> pd.DataFrame:
    labels = expression_df[[ID_COLUMN]].copy()
    tp53_labels = build_tp53_labels(mutation_df)
    labels = labels.merge(tp53_labels, on=ID_COLUMN, how="left")
    labels[TARGET_COLUMN] = labels[TARGET_COLUMN].fillna(0).astype(int)
    return expression_df.merge(labels, on=ID_COLUMN)


def get_numeric_feature_matrix(dataset: pd.DataFrame) -> pd.DataFrame:
    features = dataset.drop(columns=[ID_COLUMN, TARGET_COLUMN])
    return features.select_dtypes(include="number")


def get_tp53_related_columns(columns: list[str], exclude_prefix: str) -> list[str]:
    prefix = exclude_prefix.upper()
    return [column for column in columns if prefix in column.upper()]


def summarize_dataset(
    expression_df: pd.DataFrame,
    mutation_df: pd.DataFrame,
    dataset: pd.DataFrame,
    numeric_features: pd.DataFrame,
    filtered_features: pd.DataFrame,
    excluded_columns: list[str],
) -> Dict[str, Union[int, float]]:
    expression_ids = set(expression_df[ID_COLUMN])
    mutation_ids = set(mutation_df[ID_COLUMN])
    positive_count = int(dataset[TARGET_COLUMN].sum())
    negative_count = int((dataset[TARGET_COLUMN] == 0).sum())

    return {
        "expression_samples": int(expression_df[ID_COLUMN].nunique()),
        "mutation_samples": int(mutation_df[ID_COLUMN].nunique()),
        "overlap_samples": int(len(expression_ids.intersection(mutation_ids))),
        "merged_samples": int(len(dataset)),
        "tp53_positive_samples": positive_count,
        "tp53_negative_samples": negative_count,
        "positive_rate": round(positive_count / len(dataset), 6),
        "numeric_feature_count": int(numeric_features.shape[1]),
        "excluded_tp53_related_feature_count": int(len(excluded_columns)),
        "remaining_feature_count": int(filtered_features.shape[1]),
    }
