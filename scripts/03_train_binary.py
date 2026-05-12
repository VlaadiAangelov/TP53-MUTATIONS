#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from tp53_ml.config import load_config
from tp53_ml.evaluation import binary_metrics, save_binary_curves, save_confusion_matrix, save_per_cancer_metrics
from tp53_ml.models import binary_model_specs, feature_selectors, make_pipeline, positive_class_scores, selected_feature_names


SCALE_MODELS = {"logistic_l2", "elastic_net_logistic", "linear_svm", "mlp"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train binary TP53 mutant vs WT classifiers.")
    parser.add_argument("--config", default="config/project.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    random_state = cfg["project"]["random_state"]
    Path("models").mkdir(exist_ok=True)
    Path("reports/tables").mkdir(parents=True, exist_ok=True)
    Path("reports/figures").mkdir(parents=True, exist_ok=True)

    X = pd.read_csv(cfg["data"]["expression_processed"], index_col=0)
    labels = pd.read_csv(cfg["data"]["labels_processed"])
    metadata = pd.read_csv(cfg["data"]["metadata_processed"])
    y = labels["tp53_mutant"].astype(int)

    X_trainval, X_test, y_trainval, y_test, lab_trainval, lab_test = train_test_split(
        X,
        y,
        labels,
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
    results.to_csv("reports/tables/binary_validation_performance.csv", index=False)
    best = results.iloc[0]
    best_key = (best["feature_set"], best["model"])
    best_pipe = make_pipeline(selectors[best_key[0]], models[best_key[1]], scale=best_key[1] in SCALE_MODELS)
    best_pipe.fit(X_trainval, y_trainval)

    y_test_pred = best_pipe.predict(X_test)
    y_test_score = positive_class_scores(best_pipe, X_test)
    test_metrics = binary_metrics(y_test, y_test_pred, y_test_score)
    pd.DataFrame([{**test_metrics, "feature_set": best_key[0], "model": best_key[1], "split": "test"}]).to_csv(
        "reports/tables/binary_test_performance.csv", index=False
    )
    save_confusion_matrix(y_test, y_test_pred, labels=[0, 1], path="reports/figures/binary_confusion_matrix.png", title="TP53 mutant vs WT")
    save_binary_curves(y_test, y_test_score, "reports/figures/binary_best_model")
    save_per_cancer_metrics(lab_test, metadata, y_test_pred, "reports/tables/binary_per_lineage_performance.csv")
    joblib.dump(best_pipe, "models/best_binary_model.joblib")

    feature_names = selected_feature_names(best_pipe)
    importance = _feature_importance(best_pipe, feature_names)
    if importance is not None:
        importance.to_csv("reports/tables/binary_best_model_top_genes.csv", index=False)
        _plot_importance(importance.head(25), "reports/figures/binary_top_genes.png")

    print("Best validation model:")
    print(best[["feature_set", "model", "roc_auc", "balanced_accuracy", "f1"]])
    print("Test metrics:")
    print(test_metrics)


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


if __name__ == "__main__":
    main()
