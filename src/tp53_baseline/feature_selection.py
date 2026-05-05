from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind


def exclude_columns_by_name(features: pd.DataFrame, excluded_columns: list[str]) -> pd.DataFrame:
    return features.drop(columns=excluded_columns, errors="ignore")


def compute_feature_ranking(features: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
    mutant_features = features.loc[target == 1]
    wild_type_features = features.loc[target == 0]

    mean_mutant = mutant_features.mean(axis=0)
    mean_wild_type = wild_type_features.mean(axis=0)
    mean_diff = mean_mutant - mean_wild_type

    std_mutant = mutant_features.std(axis=0)
    std_wild_type = wild_type_features.std(axis=0)
    pooled_scale = 0.5 * (std_mutant + std_wild_type)
    pooled_scale = pooled_scale.replace(0, np.nan)
    effect_size = mean_diff / pooled_scale

    test_result = ttest_ind(
        mutant_features.to_numpy(),
        wild_type_features.to_numpy(),
        axis=0,
        equal_var=False,
        nan_policy="omit",
    )
    p_values = pd.Series(test_result.pvalue, index=features.columns, name="p_value")

    results = pd.DataFrame(
        {
            "feature": features.columns,
            "mean_diff": mean_diff.abs().to_numpy(),
            "effect_size": effect_size.abs().fillna(0.0).to_numpy(),
            "p_value": p_values.to_numpy(),
        }
    )
    return results.sort_values(
        by=["effect_size", "mean_diff", "p_value"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def select_top_features(ranking_df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    return ranking_df.head(top_n).copy()
