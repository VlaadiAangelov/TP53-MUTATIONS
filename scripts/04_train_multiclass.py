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
from tp53_ml.evaluation import multiclass_metrics, save_confusion_matrix
from tp53_ml.models import feature_selectors, make_pipeline, multiclass_model_specs


SCALE_MODELS = {"logistic_l2", "elastic_net_logistic"}

HILLCLIMB_MAX_ROUNDS = 3
HILLCLIMB_MIN_DELTA = 1e-4


def main() -> None:
    parser = argparse.ArgumentParser(description="Train mutation-type classifiers.")
    parser.add_argument("--config", default="config/project.yaml")
    parser.add_argument("--data-dir", default=None,
                        help="Override processed data directory (e.g. data/processed/tcga)")
    parser.add_argument("--tag", default=None,
                        help="Prefix for output files, e.g. 'tcga' -> tcga_multiclass_test_performance.csv")
    parser.add_argument("--merge-splice", action="store_true",
                        help="Collapse 'Splice' into 'Other' before training")
    parser.add_argument("--min-class-count", type=int, default=10,
                        help="Drop mutation type classes with fewer than this many samples (default: 10)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    random_state = cfg["project"]["random_state"]

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

    y_raw = labels["mutation_type_collapsed"].astype(str)
    if args.merge_splice:
        y_raw = y_raw.replace("Splice", "Other")

    # ------------------------------------------------------------------ #
    # Cancer type filtering — drop types with too few samples to stratify
    # (prevents model from exploiting lineage as a shortcut)
    # ------------------------------------------------------------------ #
    test_size = cfg["split"]["test_size"]
    validation_size = cfg["split"]["validation_size"]
    min_stratum_count = int(np.ceil(1 / min(test_size, validation_size)))

    split_frame = labels[["sample_id"]].assign(mutation_type_collapsed=y_raw.values).merge(
        metadata[["sample_id", "primary_disease"]], on="sample_id", how="left"
    )

    disease_counts = split_frame.groupby("primary_disease").size()
    eligible_diseases = disease_counts.index[disease_counts >= min_stratum_count]
    disease_keep_mask = split_frame["primary_disease"].isin(eligible_diseases).to_numpy()

    X = X.loc[disease_keep_mask]
    y_raw = y_raw.loc[disease_keep_mask].reset_index(drop=True)

    print(f"Minimum samples required per cancer type: {min_stratum_count}")
    print(f"Samples before cancer-type filter: {len(labels):,}")
    print(f"Samples after cancer-type filter:  {len(y_raw):,}")
    print(f"Cancer types retained: {split_frame.loc[disease_keep_mask, 'primary_disease'].nunique():,}")

    # Drop mutation type classes with too few samples to train on meaningfully
    counts = y_raw.value_counts()
    keep = y_raw.isin(counts[counts >= args.min_class_count].index)
    X = X.loc[keep.to_numpy()]
    y = y_raw.loc[keep].reset_index(drop=True)

    print(f"Samples after mutation-class filter: {len(y):,}")
    print(f"Classes retained: {sorted(y.unique())}")

    if y.nunique() < 3:
        raise RuntimeError("Not enough mutation-type classes after filtering.")

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    val_fraction = validation_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_fraction,
        stratify=y_trainval,
        random_state=random_state,
    )

    selectors = feature_selectors(cfg["features"]["top_variable_genes"], cfg["features"]["min_mean_expression"])
    models = multiclass_model_specs(random_state)

    # ------------------------------------------------------------------ #
    # Baseline pass — all model × feature combinations, no tuning
    # ------------------------------------------------------------------ #
    rows = []
    for feature_name, selector in selectors.items():
        for model_name, model in models.items():
            pipe = make_pipeline(selector, model, scale=model_name in SCALE_MODELS)
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_val)
            metric_row = multiclass_metrics(y_val, y_pred)
            metric_row.update({"feature_set": feature_name, "model": model_name, "split": "validation"})
            rows.append(metric_row)
            print(f"{feature_name} / {model_name}: val macro-F1={metric_row['macro_f1']:.3f}")

    results = pd.DataFrame(rows).sort_values(["macro_f1", "balanced_accuracy", "weighted_f1"], ascending=False)
    results.to_csv(f"reports/tables/{tag}multiclass_validation_performance.csv", index=False)

    # ------------------------------------------------------------------ #
    # Hill-climb hyperparameter tuning on validation set
    # ------------------------------------------------------------------ #
    param_spaces = _multiclass_hillclimb_param_spaces()
    tuned_rows = []
    for feature_name, selector in selectors.items():
        for model_name, model in models.items():
            pipe = make_pipeline(selector, model, scale=model_name in SCALE_MODELS)
            space = param_spaces.get(model_name, {})
            best_pipe, best_metrics, best_params, _ = _hillclimb_pipeline(
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
            }
            tuned_rows.append(row)
            print(f"[tuned] {feature_name} / {model_name}: val macro-F1={best_metrics['macro_f1']:.3f}  params={_format_params(best_params)}")

    tuned_results = pd.DataFrame(tuned_rows).sort_values(["macro_f1", "balanced_accuracy", "weighted_f1"], ascending=False)
    tuned_results.to_csv(f"reports/tables/{tag}multiclass_tuned_validation_performance.csv", index=False)

    # ------------------------------------------------------------------ #
    # Evaluate best tuned model on test set
    # ------------------------------------------------------------------ #
    best_tuned = tuned_results.iloc[0]
    best_key = (best_tuned["feature_set"], best_tuned["model"])
    best_params = _params_from_row(best_tuned)

    best_pipe = make_pipeline(selectors[best_key[0]], models[best_key[1]], scale=best_key[1] in SCALE_MODELS)
    if best_params:
        best_pipe.set_params(**best_params)
    best_pipe.fit(X_trainval, y_trainval)

    y_test_pred = best_pipe.predict(X_test)
    test_metrics = multiclass_metrics(y_test, y_test_pred)
    pd.DataFrame([{
        **test_metrics,
        "feature_set": best_key[0],
        "model": best_key[1],
        "split": "test",
        "best_params": best_tuned["best_params"],
    }]).to_csv(f"reports/tables/{tag}multiclass_tuned_test_performance.csv", index=False)

    ordered_labels = sorted(y.unique())
    save_confusion_matrix(
        y_test, y_test_pred,
        labels=ordered_labels,
        path=f"reports/figures/{tag}multiclass_tuned_confusion_matrix.png",
        title="TP53 mutation type — tuned",
    )
    joblib.dump(best_pipe, f"models/{tag}best_multiclass_tuned_model.joblib")

    print("\nBest tuned validation model:")
    print(best_tuned[["feature_set", "model", "macro_f1", "balanced_accuracy", "best_params"]])
    print("Test metrics:")
    print(test_metrics)


# ------------------------------------------------------------------ #
# Hill-climb helpers
# ------------------------------------------------------------------ #

def _hillclimb_pipeline(*, pipe, param_space, X_train, y_train, X_val, y_val):
    current_params = {
        name: pipe.get_params()[name]
        for name in param_space
        if name in pipe.get_params()
    }
    current_pipe, current_metrics = _fit_score(pipe, current_params, X_train, y_train, X_val, y_val)
    best_pipe = current_pipe
    seen = {_params_key(current_params)}

    if not param_space:
        return best_pipe, current_metrics, current_params, []

    for _ in range(1, HILLCLIMB_MAX_ROUNDS + 1):
        round_best_pipe = best_pipe
        round_best_params = current_params
        round_best_metrics = current_metrics
        improved = False

        for param_name, values in param_space.items():
            if current_params.get(param_name) not in values:
                candidates = values
            else:
                idx = values.index(current_params[param_name])
                candidates = [values[i] for i in [idx - 1, idx + 1] if 0 <= i < len(values)]

            for value in candidates:
                candidate_params = {**current_params, param_name: value}
                key = _params_key(candidate_params)
                if key in seen:
                    continue
                seen.add(key)
                candidate_pipe, candidate_metrics = _fit_score(pipe, candidate_params, X_train, y_train, X_val, y_val)
                if _metric_tuple(candidate_metrics) > _metric_tuple(round_best_metrics):
                    round_best_pipe = candidate_pipe
                    round_best_params = candidate_params
                    round_best_metrics = candidate_metrics
                    improved = True

        improvement = round_best_metrics["macro_f1"] - current_metrics["macro_f1"]
        if not improved or improvement < HILLCLIMB_MIN_DELTA:
            break

        current_params = round_best_params
        current_metrics = round_best_metrics
        best_pipe = round_best_pipe

    return best_pipe, current_metrics, current_params, []


def _fit_score(pipe, params, X_train, y_train, X_val, y_val):
    candidate = clone(pipe).set_params(**params)
    candidate.fit(X_train, y_train)
    y_pred = candidate.predict(X_val)
    return candidate, multiclass_metrics(y_val, y_pred)


def _metric_tuple(metrics):
    return (metrics["macro_f1"], metrics["balanced_accuracy"], metrics["weighted_f1"])


def _params_key(params):
    return tuple(sorted((k, repr(v)) for k, v in params.items()))


def _format_params(params):
    if not params:
        return "{}"
    clean = {k.replace("model__", ""): v for k, v in params.items()}
    return "; ".join(f"{k}={v!r}" for k, v in sorted(clean.items()))


def _params_from_row(row):
    text = str(row.get("best_params", "{}"))
    if text == "{}":
        return {}
    result = {}
    for part in text.split(";"):
        part = part.strip()
        if "=" not in part:
            continue
        k, _, v = part.partition("=")
        try:
            result[f"model__{k.strip()}"] = eval(v.strip())
        except Exception:
            result[f"model__{k.strip()}"] = v.strip()
    return result


def _multiclass_hillclimb_param_spaces():
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


def _normalise_disease_col(metadata):
    if "primary_disease" in metadata.columns:
        return metadata
    for col in metadata.columns:
        if col.lower().replace("_", "") in ("oncotreeprimarydisease", "primarydisease"):
            return metadata.rename(columns={col: "primary_disease"})
    metadata = metadata.copy()
    metadata["primary_disease"] = "Unknown"
    return metadata


if __name__ == "__main__":
    main()
