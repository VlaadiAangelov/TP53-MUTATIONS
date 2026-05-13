#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from tp53_ml.config import load_config
from tp53_ml.evaluation import multiclass_metrics, save_confusion_matrix
from tp53_ml.models import feature_selectors, make_pipeline, multiclass_model_specs


SCALE_MODELS = {"logistic_l2"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train mutation-type classifiers.")
    parser.add_argument("--config", default="config/project.yaml")
    parser.add_argument("--min-class-count", type=int, default=10)
    parser.add_argument("--data-dir", default=None,
                        help="Override processed data directory (e.g. data/processed/tcga)")
    parser.add_argument("--tag", default=None,
                        help="Prefix for output files, e.g. 'tcga' → tcga_multiclass_test_performance.csv")
    args = parser.parse_args()

    cfg = load_config(args.config)
    random_state = cfg["project"]["random_state"]

    if args.data_dir:
        data_dir = Path(args.data_dir)
        expr_path = data_dir / "expression_matched.csv.gz"
        labels_path = data_dir / "tp53_labels.csv"
    else:
        expr_path = Path(cfg["data"]["expression_processed"])
        labels_path = Path(cfg["data"]["labels_processed"])

    tag = (args.tag or (Path(args.data_dir).name if args.data_dir else "ccle")) + "_"

    Path("models").mkdir(exist_ok=True)
    Path("reports/tables").mkdir(parents=True, exist_ok=True)
    Path("reports/figures").mkdir(parents=True, exist_ok=True)

    X = pd.read_csv(expr_path, index_col=0)
    labels = pd.read_csv(labels_path)
    y_raw = labels["mutation_type_collapsed"].astype(str)

    counts = y_raw.value_counts()
    keep = y_raw.isin(counts[counts >= args.min_class_count].index)
    X = X.loc[keep.to_numpy()]
    y = y_raw.loc[keep].reset_index(drop=True)

    if y.nunique() < 3:
        raise RuntimeError("Not enough mutation-type classes after filtering. Try lowering --min-class-count.")

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X,
        y,
        test_size=cfg["split"]["test_size"],
        stratify=y,
        random_state=random_state,
    )
    val_fraction = cfg["split"]["validation_size"] / (1 - cfg["split"]["test_size"])
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=val_fraction,
        stratify=y_trainval,
        random_state=random_state,
    )

    selectors = feature_selectors(cfg["features"]["top_variable_genes"], cfg["features"]["min_mean_expression"])
    models = multiclass_model_specs(random_state)
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
    best = results.iloc[0]
    best_pipe = make_pipeline(selectors[best["feature_set"]], models[best["model"]], scale=best["model"] in SCALE_MODELS)
    best_pipe.fit(X_trainval, y_trainval)
    y_test_pred = best_pipe.predict(X_test)
    test_metrics = multiclass_metrics(y_test, y_test_pred)
    pd.DataFrame([{**test_metrics, "feature_set": best["feature_set"], "model": best["model"], "split": "test"}]).to_csv(
        f"reports/tables/{tag}multiclass_test_performance.csv", index=False
    )
    ordered_labels = sorted(y.unique())
    save_confusion_matrix(
        y_test,
        y_test_pred,
        labels=ordered_labels,
        path=f"reports/figures/{tag}multiclass_confusion_matrix.png",
        title="TP53 mutation type",
    )
    joblib.dump(best_pipe, f"models/{tag}best_multiclass_model.joblib")

    print("Best validation model:")
    print(best[["feature_set", "model", "macro_f1", "balanced_accuracy", "weighted_f1"]])
    print("Test metrics:")
    print(test_metrics)


if __name__ == "__main__":
    main()
