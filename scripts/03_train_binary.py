#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split

from tp53_ml.config import load_config
from tp53_ml.evaluation import binary_metrics, save_binary_curves, save_confusion_matrix, save_per_cancer_metrics
from tp53_ml.models import binary_model_specs, feature_selectors, make_pipeline, positive_class_scores, selected_feature_names


SCALE_MODELS = {"logistic_l2", "elastic_net_logistic", "linear_svm", "mlp"}


HILLCLIMB_MAX_ROUNDS = 3
HILLCLIMB_MIN_DELTA = 1e-4


def main() -> None:
    parser = argparse.ArgumentParser(description="Train binary TP53 mutant vs WT classifiers.")
    parser.add_argument("--config", default="config/project.yaml")
    parser.add_argument("--data-dir", default=None,
                        help="Override processed data directory (e.g. data/processed/tcga)")
    parser.add_argument("--tag", default=None,
                        help="Prefix for output files, e.g. 'tcga' -> tcga_binary_test_performance.csv")
    args = parser.parse_args()

    cfg = load_config(args.config)
    random_state = cfg["project"]["random_state"]

    # Resolve data paths; CLI --data-dir overrides config.
    if args.data_dir:
        data_dir = Path(args.data_dir)
        expr_path = data_dir / "expression_matched.csv.gz"
        labels_path = data_dir / "tp53_labels.csv"
        meta_path = data_dir / "sample_metadata.csv"
    else:
        expr_path = Path(cfg["data"]["expression_processed"])
        labels_path = Path(cfg["data"]["labels_processed"])
        meta_path = Path(cfg["data"]["metadata_processed"])

    tag = (args.tag or (Path(args.data_dir).name if args.data_dir else "ccle")) + "_"

    Path("models").mkdir(exist_ok=True)
    Path("reports/tables").mkdir(parents=True, exist_ok=True)
    Path("reports/figures").mkdir(parents=True, exist_ok=True)

    X = pd.read_csv(expr_path, index_col=0)
    labels = pd.read_csv(labels_path)
    metadata = pd.read_csv(meta_path)
    metadata = _normalise_disease_col(metadata)

    split_frame = labels[["sample_id", "tp53_mutant", "mutation_type_collapsed"]].merge(
        metadata[["sample_id", "primary_disease"]],
        on="sample_id",
        how="left",
    )

    test_size = cfg["split"]["test_size"]
    validation_size = cfg["split"]["validation_size"]
    min_stratum_count = int(np.ceil(1 / min(test_size, validation_size)))

    disease_status_counts = (
        split_frame.groupby(["primary_disease", "tp53_mutant"])
        .size()
        .unstack(fill_value=0)
    )
    for status in [0, 1]:
        if status not in disease_status_counts.columns:
            disease_status_counts[status] = 0

    eligible_diseases = disease_status_counts.index[
        (disease_status_counts[0] >= min_stratum_count)
        & (disease_status_counts[1] >= min_stratum_count)
    ]
    disease_keep_mask = split_frame["primary_disease"].isin(eligible_diseases)

    strata = (
        split_frame["primary_disease"].fillna("Unknown disease")
        + " | TP53_"
        + split_frame["tp53_mutant"].map({0: "WT", 1: "mutant"}).astype(str)
        + " | "
        + split_frame["mutation_type_collapsed"].fillna("Unknown mutation type").astype(str)
    )
    strata_after_disease_filter = strata.loc[disease_keep_mask]
    stratum_counts = strata_after_disease_filter.value_counts()
    composite_keep_mask = strata.map(stratum_counts).fillna(0) >= min_stratum_count
    keep_mask = disease_keep_mask & composite_keep_mask

    X_split = X.loc[keep_mask.to_numpy()]
    labels_split = labels.loc[keep_mask.to_numpy()].reset_index(drop=True)
    strata_split = strata.loc[keep_mask].reset_index(drop=True)
    y = labels_split["tp53_mutant"].astype(int)

    print(f"Minimum samples required per stratum: {min_stratum_count}")
    print(f"Samples before filtering: {len(labels):,}")
    print(f"Samples after filtering:  {len(labels_split):,}")
    print(f"Samples removed:          {len(labels) - len(labels_split):,}")
    print(f"Cancer types retained:    {split_frame.loc[keep_mask, 'primary_disease'].nunique():,}")
    print(f"Strata retained:          {strata_split.nunique():,}")

    X_trainval, X_test, y_trainval, y_test, lab_trainval, lab_test, strata_trainval, _ = train_test_split(
        X_split,
        y,
        labels_split,
        strata_split,
        test_size=test_size,
        stratify=strata_split,
        random_state=random_state,
    )
    val_fraction = validation_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=val_fraction,
        stratify=strata_trainval,
        random_state=random_state,
    )

    selectors = feature_selectors(cfg["features"]["top_variable_genes"], cfg["features"]["min_mean_expression"])
    models = binary_model_specs(random_state)
    rows = []
    fitted = {}
    for feature_name, selector in selectors.items():
        for model_name, model in models.items():
            if model_name in {"hist_gradient_boosting", "mlp"} and feature_name != f"top_{cfg['features']['top_variable_genes']}_variable":
                continue
            pipe = make_pipeline(selector, model, scale=model_name in SCALE_MODELS)
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_val)
            y_score = positive_class_scores(pipe, X_val)
            metric_row = binary_metrics(y_val, y_pred, y_score)
            metric_row.update({"feature_set": feature_name, "model": model_name, "split": "validation"})
            rows.append(metric_row)
            fitted[(feature_name, model_name)] = pipe
            print(f"{feature_name} / {model_name}: val ROC-AUC={metric_row['roc_auc']:.3f}, F1={metric_row['f1']:.3f}")

    results = pd.DataFrame(rows).sort_values(["roc_auc", "balanced_accuracy", "f1"], ascending=False)
    results.to_csv(f"reports/tables/{tag}binary_validation_performance.csv", index=False)

    tuned_results, tuning_trace = _hillclimb_all_binary_models(
        selectors=selectors,
        models=models,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        top_variable_feature_set=f"top_{cfg['features']['top_variable_genes']}_variable",
    )
    tuned_results.to_csv(f"reports/tables/{tag}binary_tuned_validation_performance.csv", index=False)
    tuning_trace.to_csv(f"reports/tables/{tag}binary_tuning_trace.csv", index=False)

    best = results.iloc[0]
    best_key = (best["feature_set"], best["model"])
    best_pipe = make_pipeline(selectors[best_key[0]], models[best_key[1]], scale=best_key[1] in SCALE_MODELS)
    best_pipe.fit(X_trainval, y_trainval)

    y_test_pred = best_pipe.predict(X_test)
    y_test_score = positive_class_scores(best_pipe, X_test)
    test_metrics = binary_metrics(y_test, y_test_pred, y_test_score)
    pd.DataFrame([{**test_metrics, "feature_set": best_key[0], "model": best_key[1], "split": "test"}]).to_csv(
        f"reports/tables/{tag}binary_test_performance.csv", index=False
    )
    save_confusion_matrix(y_test, y_test_pred, labels=[0, 1], path=f"reports/figures/{tag}binary_confusion_matrix.png", title="TP53 mutant vs WT")
    save_binary_curves(y_test, y_test_score, f"reports/figures/{tag}binary_best_model")
    save_per_cancer_metrics(lab_test, metadata, y_test_pred, f"reports/tables/{tag}binary_per_lineage_performance.csv")
    joblib.dump(best_pipe, f"models/{tag}best_binary_model.joblib")

    best_tuned = tuned_results.iloc[0]
    best_tuned_key = (best_tuned["feature_set"], best_tuned["model"])
    best_tuned_params = _params_from_row(best_tuned)
    best_tuned_pipe = make_pipeline(
        selectors[best_tuned_key[0]],
        models[best_tuned_key[1]],
        scale=best_tuned_key[1] in SCALE_MODELS,
    )
    if best_tuned_params:
        best_tuned_pipe.set_params(**best_tuned_params)
    best_tuned_pipe.fit(X_trainval, y_trainval)

    y_tuned_test_pred = best_tuned_pipe.predict(X_test)
    y_tuned_test_score = positive_class_scores(best_tuned_pipe, X_test)
    tuned_test_metrics = binary_metrics(y_test, y_tuned_test_pred, y_tuned_test_score)
    pd.DataFrame(
        [
            {
                **tuned_test_metrics,
                "feature_set": best_tuned_key[0],
                "model": best_tuned_key[1],
                "split": "test",
                "best_params": best_tuned["best_params"],
            }
        ]
    ).to_csv(f"reports/tables/{tag}binary_tuned_test_performance.csv", index=False)
    save_confusion_matrix(
        y_test,
        y_tuned_test_pred,
        labels=[0, 1],
        path=f"reports/figures/{tag}binary_tuned_confusion_matrix.png",
        title="TP53 mutant vs WT - tuned",
    )
    save_binary_curves(y_test, y_tuned_test_score, f"reports/figures/{tag}binary_tuned_best_model")
    joblib.dump(best_tuned_pipe, f"models/{tag}best_binary_tuned_model.joblib")

    feature_names = selected_feature_names(best_pipe)
    importance = _feature_importance(best_pipe, feature_names)
    if importance is not None:
        importance.to_csv(f"reports/tables/{tag}binary_best_model_top_genes.csv", index=False)
        _plot_importance(importance.head(25), f"reports/figures/{tag}binary_top_genes.png")

    print("Best validation model:")
    print(best[["feature_set", "model", "roc_auc", "balanced_accuracy", "f1"]])
    print("Test metrics:")
    print(test_metrics)
    print("Best tuned validation model:")
    print(best_tuned[["feature_set", "model", "roc_auc", "balanced_accuracy", "f1", "best_params"]])
    print("Tuned test metrics:")
    print(tuned_test_metrics)


def _hillclimb_all_binary_models(
    *,
    selectors: dict,
    models: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    top_variable_feature_set: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    param_spaces = _binary_hillclimb_param_spaces()
    tuned_rows = []
    trace_rows = []

    for feature_name, selector in selectors.items():
        for model_name, model in models.items():
            if model_name in {"hist_gradient_boosting", "mlp"} and feature_name != top_variable_feature_set:
                continue
            pipe = make_pipeline(selector, model, scale=model_name in SCALE_MODELS)
            space = param_spaces.get(model_name, {})
            _best_pipe, best_metrics, best_params, model_trace = _hillclimb_pipeline(
                pipe=pipe,
                param_space=space,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
            )
            row = {
                **best_metrics,
                "feature_set": feature_name,
                "model": model_name,
                "split": "validation_tuned",
                "best_params": _format_params(best_params),
                "n_tuning_evaluations": len(model_trace),
            }
            tuned_rows.append(row)
            for trace_row in model_trace:
                trace_rows.append(
                    {
                        "feature_set": feature_name,
                        "model": model_name,
                        **trace_row,
                    }
                )
            print(
                f"Tuned {feature_name} / {model_name}: "
                f"val ROC-AUC={best_metrics['roc_auc']:.3f}, F1={best_metrics['f1']:.3f}, "
                f"params={_format_params(best_params)}"
            )

    tuned_results = pd.DataFrame(tuned_rows).sort_values(
        ["roc_auc", "balanced_accuracy", "f1"],
        ascending=False,
    )
    trace = pd.DataFrame(trace_rows)
    return tuned_results, trace


def _hillclimb_pipeline(
    *,
    pipe,
    param_space: dict[str, list],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> tuple[object, dict, dict, list[dict]]:
    current_params = {
        name: pipe.get_params()[name]
        for name in param_space
        if name in pipe.get_params()
    }
    current_pipe, current_metrics = _fit_score_pipeline(pipe, current_params, X_train, y_train, X_val, y_val)
    best_pipe = current_pipe
    trace = [
        {
            "round": 0,
            "changed_param": "baseline",
            "params": _format_params(current_params),
            **current_metrics,
        }
    ]
    seen = {_params_key(current_params)}

    if not param_space:
        return best_pipe, current_metrics, current_params, trace

    for round_idx in range(1, HILLCLIMB_MAX_ROUNDS + 1):
        round_best_pipe = best_pipe
        round_best_params = current_params
        round_best_metrics = current_metrics
        round_best_changed_param = None

        for param_name, values in param_space.items():
            if current_params.get(param_name) not in values:
                candidates = values
            else:
                current_index = values.index(current_params[param_name])
                neighbor_indices = [current_index - 1, current_index + 1]
                candidates = [values[i] for i in neighbor_indices if 0 <= i < len(values)]

            for value in candidates:
                candidate_params = {**current_params, param_name: value}
                candidate_key = _params_key(candidate_params)
                if candidate_key in seen:
                    continue
                seen.add(candidate_key)

                candidate_pipe, candidate_metrics = _fit_score_pipeline(
                    pipe,
                    candidate_params,
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                )
                trace.append(
                    {
                        "round": round_idx,
                        "changed_param": param_name,
                        "params": _format_params(candidate_params),
                        **candidate_metrics,
                    }
                )
                if _metric_tuple(candidate_metrics) > _metric_tuple(round_best_metrics):
                    round_best_pipe = candidate_pipe
                    round_best_params = candidate_params
                    round_best_metrics = candidate_metrics
                    round_best_changed_param = param_name

        improvement = round_best_metrics["roc_auc"] - current_metrics["roc_auc"]
        if round_best_changed_param is None or improvement < HILLCLIMB_MIN_DELTA:
            break

        current_params = round_best_params
        current_metrics = round_best_metrics
        best_pipe = round_best_pipe

    return best_pipe, current_metrics, current_params, trace


def _fit_score_pipeline(
    pipe,
    params: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> tuple[object, dict]:
    candidate = clone(pipe).set_params(**params)
    candidate.fit(X_train, y_train)
    y_pred = candidate.predict(X_val)
    y_score = positive_class_scores(candidate, X_val)
    return candidate, binary_metrics(y_val, y_pred, y_score)


def _binary_hillclimb_param_spaces() -> dict[str, dict[str, list]]:
    return {
        "majority": {},
        "logistic_l2": {
            "model__C": [0.01, 0.03, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0],
        },
        "elastic_net_logistic": {
            "model__C": [0.01, 0.03, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0],
            "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
        },
        "linear_svm": {
            "model__C": [0.01, 0.03, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0],
        },
        "random_forest": {
            "model__n_estimators": [200, 400, 700],
            "model__max_features": ["sqrt", 0.25, 0.5],
            "model__min_samples_leaf": [1, 2, 5, 10],
            "model__max_depth": [None, 8, 16, 32],
        },
        "extra_trees": {
            "model__n_estimators": [300, 500, 800],
            "model__max_features": ["sqrt", 0.25, 0.5],
            "model__min_samples_leaf": [1, 2, 5, 10],
            "model__max_depth": [None, 16, 32],
        },
        "hist_gradient_boosting": {
            "model__learning_rate": [0.02, 0.05, 0.1],
            "model__max_iter": [150, 250, 400],
            "model__max_leaf_nodes": [15, 31, 63],
            "model__l2_regularization": [0.0, 0.1, 1.0],
        },
        "mlp": {
            "model__hidden_layer_sizes": [(128,), (256, 64), (256, 128)],
            "model__alpha": [0.0001, 0.001, 0.01],
            "model__learning_rate_init": [0.0005, 0.001, 0.003],
        },
    }


def _metric_tuple(metrics: dict) -> tuple[float, float, float]:
    return (metrics["roc_auc"], metrics["balanced_accuracy"], metrics["f1"])


def _params_key(params: dict) -> tuple:
    return tuple(sorted((name, repr(value)) for name, value in params.items()))


def _format_params(params: dict) -> str:
    if not params:
        return "{}"
    clean_params = {name.replace("model__", ""): value for name, value in params.items()}
    return "; ".join(f"{name}={value!r}" for name, value in sorted(clean_params.items()))


def _params_from_row(row: pd.Series) -> dict:
    params_text = str(row.get("best_params", "{}"))
    if params_text == "{}":
        return {}

    parsed = {}
    for part in params_text.split("; "):
        if not part:
            continue
        name, value_text = part.split("=", 1)
        parsed[f"model__{name}"] = _parse_param_value(value_text)
    return parsed


def _parse_param_value(value_text: str):
    if value_text == "None":
        return None
    if value_text.startswith("'") and value_text.endswith("'"):
        return value_text[1:-1]
    if value_text.startswith("(") and value_text.endswith(")"):
        values = value_text.strip("()").split(",")
        return tuple(int(value.strip()) for value in values if value.strip())
    try:
        return int(value_text)
    except ValueError:
        return float(value_text)


def _feature_importance(pipe, feature_names: list[str]) -> pd.DataFrame | None:
    model = pipe.named_steps["model"]
    if hasattr(model, "coef_"):
        values = np.ravel(model.coef_)
        return (
            pd.DataFrame({"gene": feature_names, "importance": values, "abs_importance": np.abs(values)})
            .sort_values("abs_importance", ascending=False)
            .head(100)
        )
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
        return (
            pd.DataFrame({"gene": feature_names, "importance": values, "abs_importance": np.abs(values)})
            .sort_values("abs_importance", ascending=False)
            .head(100)
        )
    return None


def _plot_importance(df: pd.DataFrame, path: str) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    plot_df = df.iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    sns.barplot(plot_df, x="abs_importance", y="gene", ax=ax, color="#4C78A8")
    ax.set_xlabel("Absolute importance")
    ax.set_ylabel("")
    ax.set_title("Top genes in best binary model")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _normalise_disease_col(metadata: "pd.DataFrame") -> "pd.DataFrame":
    if "primary_disease" in metadata.columns:
        return metadata
    for col in metadata.columns:
        if col.lower().replace("_", "") == "oncotreeprimarydisease" or col.lower().replace("_", "") == "primarydisease":
            return metadata.rename(columns={col: "primary_disease"})
    metadata = metadata.copy()
    metadata["primary_disease"] = "Unknown"
    return metadata


if __name__ == "__main__":
    main()
