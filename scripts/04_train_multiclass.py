#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from tp53_ml.config import load_config
from tp53_ml.evaluation import multiclass_metrics, save_confusion_matrix
from tp53_ml.models import feature_selectors, make_pipeline, multiclass_model_specs


SCALE_MODELS = {"logistic_l2", "elastic_net_logistic", "linear_svm", "mlp"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train mutation-type classifiers.")
    parser.add_argument("--config", default="config/project.yaml")
    parser.add_argument("--min-class-count", type=int, default=10)
    args = parser.parse_args()

    cfg = load_config(args.config)
    random_state = cfg["project"]["random_state"]
    Path("models").mkdir(exist_ok=True)
    Path("reports/tables").mkdir(parents=True, exist_ok=True)
    Path("reports/figures").mkdir(parents=True, exist_ok=True)

    X = pd.read_csv(cfg["data"]["expression_processed"], index_col=0)
    labels = pd.read_csv(cfg["data"]["labels_processed"])
    label_col = "mutation_type_hotspot" if "mutation_type_hotspot" in labels.columns else "mutation_type_collapsed"
    y_raw = labels[label_col].astype(str)

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
    print(f"Using mutation-type label column: {label_col}")
    print("Kept class counts:")
    print(y.value_counts().to_string())
    for feature_name, selector in selectors.items():
        for model_name, model in models.items():
            if model_name in {"hist_gradient_boosting", "mlp"} and not feature_name.startswith("top_"):
                continue
            pipe = make_pipeline(selector, model, scale=model_name in SCALE_MODELS)
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_val)
            metric_row = multiclass_metrics(y_val, y_pred)
            metric_row.update({"feature_set": feature_name, "model": model_name, "split": "validation"})
            rows.append(metric_row)
            print(f"{feature_name} / {model_name}: val macro-F1={metric_row['macro_f1']:.3f}")

    results = pd.DataFrame(rows).sort_values(["macro_f1", "balanced_accuracy", "weighted_f1"], ascending=False)
    results.to_csv("reports/tables/multiclass_validation_performance.csv", index=False)
    best = results.iloc[0]
    best_pipe = make_pipeline(selectors[best["feature_set"]], models[best["model"]], scale=best["model"] in SCALE_MODELS)
    best_pipe.fit(X_trainval, y_trainval)
    y_test_pred = best_pipe.predict(X_test)
    test_metrics = multiclass_metrics(y_test, y_test_pred)
    pd.DataFrame([{**test_metrics, "feature_set": best["feature_set"], "model": best["model"], "split": "test"}]).to_csv(
        "reports/tables/multiclass_test_performance.csv", index=False
    )
    ordered_labels = sorted(y.unique())
    save_confusion_matrix(
        y_test,
        y_test_pred,
        labels=ordered_labels,
        path="reports/figures/multiclass_confusion_matrix.png",
        title="TP53 mutation type",
    )
    _save_confusion_matrices(y_test, y_test_pred, ordered_labels, "reports/figures/multiclass_confusion_matrices.png")
    per_class = _per_class_metrics(y_test, y_test_pred, ordered_labels)
    per_class.to_csv("reports/tables/multiclass_per_class_metrics.csv")
    joblib.dump(best_pipe, "models/best_multiclass_model.joblib")

    print("Best validation model:")
    print(best[["feature_set", "model", "macro_f1", "balanced_accuracy", "weighted_f1"]])
    print("Test metrics:")
    print(test_metrics)
    print("Per-class metrics:")
    print(per_class)


def _save_confusion_matrices(y_true, y_pred, labels: list[str], path: str) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm_raw = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.heatmap(cm_raw, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=axes[0])
    axes[0].set_title("Mutation-type confusion matrix - raw counts")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Observed")

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        vmin=0,
        vmax=1,
        xticklabels=labels,
        yticklabels=labels,
        ax=axes[1],
    )
    axes[1].set_title("Mutation-type confusion matrix - row-normalised recall")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Observed")

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _per_class_metrics(y_true, y_pred, labels: list[str]) -> pd.DataFrame:
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    return (
        pd.DataFrame(report)
        .T.loc[[label for label in labels if label in report], ["precision", "recall", "f1-score", "support"]]
        .round(3)
    )


if __name__ == "__main__":
    main()
