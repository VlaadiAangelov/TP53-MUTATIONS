#!/usr/bin/env python3
"""
Advanced multiclass TP53 mutation-type classification.

Trains two strategies on top of the best feature set found in 04_train_multiclass.py:
  1. LightGBM  — native multiclass with class weighting
  2. One-vs-Rest (OvR) — one LightGBM binary classifier per mutation type

Run after 04_train_multiclass.py. Reuses the same train/val/test split
(same random_state) so results are directly comparable.

Usage:
    python scripts/06_train_advanced_multiclass.py
    python scripts/06_train_advanced_multiclass.py --data-dir data/processed/tcga --tag tcga
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer

try:
    import lightgbm as lgb
except ImportError:
    print("ERROR: lightgbm not installed. Run: pip install lightgbm", file=sys.stderr)
    sys.exit(1)

from tp53_ml.config import load_config
from tp53_ml.evaluation import multiclass_metrics, save_confusion_matrix
from tp53_ml.genes import P53_PATHWAY_GENES, TP53_TARGET_GENES
from tp53_ml.preprocessing import GeneSetSelector, TopVarianceSelector


def build_lgbm(random_state: int, class_weight: str = "balanced") -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=10,
        class_weight=class_weight,
        random_state=random_state,
        verbose=-1,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Advanced multiclass training: LightGBM + One-vs-Rest.")
    parser.add_argument("--config", default="config/project.yaml")
    parser.add_argument("--min-class-count", type=int, default=10)
    parser.add_argument("--data-dir", default=None,
                        help="Override processed data directory (e.g. data/processed/tcga)")
    parser.add_argument("--tag", default=None,
                        help="Prefix for output files, e.g. 'tcga'")
    parser.add_argument("--top-k", type=int, default=3000,
                        help="Number of top variable genes (only used when --feature-set=top_k_variable)")
    parser.add_argument("--feature-set", default="top_k_variable",
                        choices=["top_k_variable", "tp53_targets", "p53_pathway"],
                        help="Feature set to use (default: top_k_variable)")
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
    fs_tag = args.feature_set  # included in output filenames so results don't overwrite each other

    Path("models").mkdir(exist_ok=True)
    Path("reports/tables").mkdir(parents=True, exist_ok=True)
    Path("reports/figures").mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Load data — same split as 04_train_multiclass.py
    # ------------------------------------------------------------------ #
    print("Loading data …")
    X = pd.read_csv(expr_path, index_col=0)
    labels = pd.read_csv(labels_path)
    y_raw = labels["mutation_type_collapsed"].astype(str)

    counts = y_raw.value_counts()
    keep = y_raw.isin(counts[counts >= args.min_class_count].index)
    X = X.loc[keep.to_numpy()]
    y = y_raw.loc[keep].reset_index(drop=True)
    print(f"  Classes: {sorted(y.unique())}")
    print(f"  Class counts:\n{y.value_counts().to_string()}")

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y,
        test_size=cfg["split"]["test_size"],
        stratify=y,
        random_state=random_state,
    )
    val_fraction = cfg["split"]["validation_size"] / (1 - cfg["split"]["test_size"])
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_fraction,
        stratify=y_trainval,
        random_state=random_state,
    )

    # Feature selection — fitted on train only
    if args.feature_set == "p53_pathway":
        selector = GeneSetSelector(P53_PATHWAY_GENES)
    elif args.feature_set == "tp53_targets":
        selector = GeneSetSelector(TP53_TARGET_GENES)
    else:
        selector = TopVarianceSelector(
            k=args.top_k,
            min_mean_expression=cfg["features"]["min_mean_expression"],
        )

    X_train_sel    = selector.fit_transform(X_train)
    X_val_sel      = selector.transform(X_val)
    X_test_sel     = selector.transform(X_test)
    X_trainval_sel = selector.transform(X_trainval)
    print(f"  Feature set: {args.feature_set}  ({X_train_sel.shape[1]} features)")

    rows = []

    # ------------------------------------------------------------------ #
    # 1. LightGBM — native multiclass
    # ------------------------------------------------------------------ #
    print("\n[1/2] Training LightGBM multiclass …")
    lgbm = build_lgbm(random_state)
    lgbm.fit(X_train_sel, y_train)
    y_pred_val = lgbm.predict(X_val_sel)
    metrics_val = multiclass_metrics(y_val, y_pred_val)
    print(f"  val macro-F1={metrics_val['macro_f1']:.3f}, balanced_acc={metrics_val['balanced_accuracy']:.3f}")
    rows.append({**metrics_val, "model": "lightgbm_multiclass", "split": "validation"})

    # Retrain on trainval, evaluate on test
    lgbm_final = build_lgbm(random_state)
    lgbm_final.fit(X_trainval_sel, y_trainval)
    y_pred_test = lgbm_final.predict(X_test_sel)
    metrics_test = multiclass_metrics(y_test, y_pred_test)
    print(f"  test macro-F1={metrics_test['macro_f1']:.3f}, balanced_acc={metrics_test['balanced_accuracy']:.3f}")
    rows.append({**metrics_test, "model": "lightgbm_multiclass", "split": "test"})

    save_confusion_matrix(
        y_test, y_pred_test,
        labels=sorted(y.unique()),
        path=f"reports/figures/{tag}advanced_lgbm_{fs_tag}_confusion_matrix.png",
        title=f"LightGBM multiclass — {args.feature_set}",
    )
    joblib.dump((selector, lgbm_final), f"models/{tag}lgbm_multiclass_{fs_tag}_model.joblib")

    # ------------------------------------------------------------------ #
    # 2. One-vs-Rest with LightGBM
    # ------------------------------------------------------------------ #
    print("\n[2/2] Training One-vs-Rest LightGBM …")
    ovr = OneVsRestClassifier(build_lgbm(random_state, class_weight=None), n_jobs=1)
    ovr.fit(X_train_sel, y_train)
    y_pred_val_ovr = ovr.predict(X_val_sel)
    metrics_val_ovr = multiclass_metrics(y_val, y_pred_val_ovr)
    print(f"  val macro-F1={metrics_val_ovr['macro_f1']:.3f}, balanced_acc={metrics_val_ovr['balanced_accuracy']:.3f}")
    rows.append({**metrics_val_ovr, "model": "lightgbm_ovr", "split": "validation"})

    ovr_final = OneVsRestClassifier(build_lgbm(random_state, class_weight=None), n_jobs=1)
    ovr_final.fit(X_trainval_sel, y_trainval)
    y_pred_test_ovr = ovr_final.predict(X_test_sel)
    metrics_test_ovr = multiclass_metrics(y_test, y_pred_test_ovr)
    print(f"  test macro-F1={metrics_test_ovr['macro_f1']:.3f}, balanced_acc={metrics_test_ovr['balanced_accuracy']:.3f}")
    rows.append({**metrics_test_ovr, "model": "lightgbm_ovr", "split": "test"})

    save_confusion_matrix(
        y_test, y_pred_test_ovr,
        labels=sorted(y.unique()),
        path=f"reports/figures/{tag}advanced_ovr_{fs_tag}_confusion_matrix.png",
        title=f"One-vs-Rest LightGBM — {args.feature_set}",
    )
    joblib.dump((selector, ovr_final), f"models/{tag}lgbm_ovr_{fs_tag}_model.joblib")

    # ------------------------------------------------------------------ #
    # Save combined results
    # ------------------------------------------------------------------ #
    results = pd.DataFrame(rows)
    results["feature_set"] = args.feature_set
    results.to_csv(f"reports/tables/{tag}advanced_multiclass_{fs_tag}_results.csv", index=False)

    print("\n=== Summary ===")
    print(results[["model", "split", "macro_f1", "balanced_accuracy", "weighted_f1"]].to_string(index=False))


if __name__ == "__main__":
    main()
