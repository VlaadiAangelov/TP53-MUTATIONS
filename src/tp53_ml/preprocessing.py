from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def clean_gene_name(column: str) -> str:
    """Convert DepMap columns like 'TP53 (7157)' to 'TP53'."""
    column = str(column)
    return re.sub(r"\s+\(\d+\)$", "", column).strip()


def clean_expression_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [clean_gene_name(col) for col in df.columns]
    if df.columns.duplicated().any():
        df = df.T.groupby(level=0).mean().T
    return df


class TopVarianceSelector(BaseEstimator, TransformerMixin):
    """Select the k most variable columns using only the fitted data."""

    def __init__(self, k: int = 3000, min_mean_expression: float = 0.1):
        self.k = k
        self.min_mean_expression = min_mean_expression

    def fit(self, X, y=None):
        frame = _as_frame(X)
        means = frame.mean(axis=0)
        variances = frame.var(axis=0)
        eligible = variances[means >= self.min_mean_expression]
        if eligible.empty:
            eligible = variances
        self.selected_features_ = eligible.sort_values(ascending=False).head(self.k).index.to_list()
        return self

    def transform(self, X):
        frame = _as_frame(X)
        return frame.loc[:, self.selected_features_]

    def get_feature_names_out(self, input_features: Iterable[str] | None = None):
        return np.asarray(self.selected_features_, dtype=object)


class GeneSetSelector(BaseEstimator, TransformerMixin):
    """Select the genes from a curated set that are present in the matrix."""

    def __init__(self, genes: Iterable[str]):
        self.genes = genes

    def fit(self, X, y=None):
        frame = _as_frame(X)
        requested = sorted(set(self.genes))
        self.selected_features_ = [gene for gene in requested if gene in frame.columns]
        if not self.selected_features_:
            raise ValueError("None of the requested gene-set features were found in the expression matrix.")
        return self

    def transform(self, X):
        frame = _as_frame(X)
        return frame.loc[:, self.selected_features_]

    def get_feature_names_out(self, input_features: Iterable[str] | None = None):
        return np.asarray(self.selected_features_, dtype=object)


def _as_frame(X) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X
    return pd.DataFrame(X)
