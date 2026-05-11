from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def binary_metrics(y_true, y_pred, y_score) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score),
        "pr_auc": average_precision_score(y_true, y_score),
    }


def multiclass_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def save_confusion_matrix(y_true, y_pred, labels, path: str | Path, title: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(max(4, len(labels) * 0.9), max(3.5, len(labels) * 0.7)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Observed")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_binary_curves(y_true, y_score, prefix: str | Path) -> None:
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_true, y_score, ax=ax)
    ax.set_title("TP53 binary classification ROC")
    fig.tight_layout()
    fig.savefig(prefix.with_name(prefix.name + "_roc.png"), dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    PrecisionRecallDisplay.from_predictions(y_true, y_score, ax=ax)
    ax.set_title("TP53 binary classification precision-recall")
    fig.tight_layout()
    fig.savefig(prefix.with_name(prefix.name + "_pr.png"), dpi=180)
    plt.close(fig)


def save_per_cancer_metrics(labels: pd.DataFrame, metadata: pd.DataFrame, y_pred, out_path: str | Path) -> None:
    if metadata.empty:
        return
    meta_cols = {col.lower(): col for col in metadata.columns}
    lineage_col = None
    for candidate in ["onCotreeLineage", "OncotreeLineage", "lineage", "primary_disease", "CCLE_Name"]:
        if candidate in metadata.columns:
            lineage_col = candidate
            break
        if candidate.lower() in meta_cols:
            lineage_col = meta_cols[candidate.lower()]
            break
    if lineage_col is None:
        return

    tmp = labels[["sample_id", "tp53_mutant"]].copy()
    tmp["prediction"] = np.asarray(y_pred)
    tmp = tmp.merge(metadata[["sample_id", lineage_col]], on="sample_id", how="left")
    rows = []
    for lineage, group in tmp.dropna(subset=[lineage_col]).groupby(lineage_col):
        if len(group) < 10 or group["tp53_mutant"].nunique() < 2:
            continue
        rows.append(
            {
                "lineage": lineage,
                "n": len(group),
                "balanced_accuracy": balanced_accuracy_score(group["tp53_mutant"], group["prediction"]),
                "f1": f1_score(group["tp53_mutant"], group["prediction"], zero_division=0),
            }
        )
    if rows:
        pd.DataFrame(rows).sort_values("n", ascending=False).to_csv(out_path, index=False)

