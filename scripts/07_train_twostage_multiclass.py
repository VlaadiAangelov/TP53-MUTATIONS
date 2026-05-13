#!/usr/bin/env python3
"""
Two-stage TP53 mutation-type classification.

Stage 1 (binary): mutant vs WT — handled by an existing binary model.
Stage 2 (multiclass): trained on *mutant samples only*, predicts mutation type
    (Frameshift, Missense, Nonsense, Other, Splice).

Removing WT from the multiclass task prevents the dominant WT class from
overwhelming the minority mutation types and forces the model to actually
discriminate between mutation mechanisms.

Evaluation uses an oracle test set (true mutants only) — the upper bound
assuming perfect stage-1 binary classification.

Usage:
    python scripts/07_train_twostage_multiclass.py
    python scripts/07_train_twostage_multiclass.py --data-dir data/processed/tcga --tag tcga
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print("WARNING: lightgbm not installed — LightGBM model will be skipped.", file=sys.stderr)

from tp53_ml.config import load_config
from tp53_ml.evaluation import multiclass_metrics, save_confusion_matrix
from tp53_ml.models import feature_selectors, make_pipeline, multiclass_model_specs

SCALE_MODELS = {"logistic_l2", "elastic_net_logistic"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Two-stage multiclass: mutants only.")
    parser.add_argument("--config", default="config/project.yaml")
    parser.add_argument("--min-class-count", type=int, default=10)
    parser.add_argument("--data-dir", default=None,
                        help="Override processed data directory (e.g. data/processed/tcga)")
    parser.add_argument("--tag", default=None,
                        help="Prefix for output files, e.g. 'tcga'")
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

    # ------------------------------------------------------------------ #
    # Load data — filter to mutants only before splitting
    # ------------------------------------------------------------------ #
    print("Loading data …")
    X_all = pd.read_csv(expr_path, index_col=0)
    labels = pd.read_csv(labels_path)
    y_raw = labels["mutation_type_collapsed"].astype(str)

    mask_mutant = y_raw != "WT"
    X = X_all.loc[mask_mutant.to_numpy()]
    y_mut = y_raw.loc[mask_mutant].reset_index(drop=True)

    counts = y_mut.value_counts()
    keep = y_mut.isin(counts[counts >= args.min_class_count].index)
    X = X.loc[keep.to_numpy()]
    y = y_mut.loc[keep].reset_index(drop=True)

    print(f"  Mutant samples after filtering: {len(y)}")
    print(f"  Classes: {sorted(y.unique())}")
    print(f"  Class counts:\n{y.value_counts().to_string()}")

    # ------------------------------------------------------------------ #
    # Train / val / test split — stratified over mutation types only
    # ------------------------------------------------------------------ #
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
    print(f"  Train: {len(y_train)} | Val: {len(y_val)} | Test: {len(y_test)}")

    # ------------------------------------------------------------------ #
    # Model grid — same as 04_train_multiclass.py + optional LightGBM
    # ------------------------------------------------------------------ #
    selectors = feature_selectors(cfg["features"]["top_variable_genes"], cfg["features"]["min_mean_expression"])
    models = multiclass_model_specs(random_state)

    if HAS_LGBM:
        models["lightgbm"] = lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=63,
            min_child_samples=10,
            class_weight="balanced",
            random_state=random_state,
            verbose=-1,
        )

    val_rows = []
    print("\nTraining models on mutant-only data …")
    for feature_name, selector in selectors.items():
        for model_name, model in models.items():
            pipe = make_pipeline(selector, model, scale=model_name in SCALE_MODELS)
            pipe.fit(X_train, y_train)
            y_pred_val = pipe.predict(X_val)
            row = multiclass_metrics(y_val, y_pred_val)
            row.update({"feature_set": feature_name, "model": model_name})
            val_rows.append(row)
            print(f"  {feature_name} / {model_name}: "
                  f"val macro-F1={row['macro_f1']:.3f}  bal-acc={row['balanced_accuracy']:.3f}")

    val_results = pd.DataFrame(val_rows).sort_values(
        ["macro_f1", "balanced_accuracy", "weighted_f1"], ascending=False
    )
    val_results.to_csv(f"reports/tables/{tag}twostage_validation_performance.csv", index=False)

    # ------------------------------------------------------------------ #
    # Best model — retrain on trainval, evaluate on test (oracle)
    # ------------------------------------------------------------------ #
    best = val_results.iloc[0]
    print(f"\nBest validation model: {best['model']} / {best['feature_set']}")

    best_pipe = make_pipeline(
        selectors[best["feature_set"]],
        models[best["model"]],
        scale=best["model"] in SCALE_MODELS,
    )
    best_pipe.fit(X_trainval, y_trainval)
    y_pred_test = best_pipe.predict(X_test)

    test_metrics = multiclass_metrics(y_test, y_pred_test)
    test_metrics.update({
        "feature_set": best["feature_set"],
        "model": best["model"],
    })
    print(f"  Test macro-F1={test_metrics['macro_f1']:.3f}  "
          f"bal-acc={test_metrics['balanced_accuracy']:.3f}")

    ordered_labels = sorted(y.unique())
    save_confusion_matrix(
        y_test, y_pred_test,
        labels=ordered_labels,
        path=f"reports/figures/{tag}twostage_confusion_matrix.png",
        title=f"Two-stage multiclass (mutants only) — {best['model']} / {best['feature_set']}",
    )
    joblib.dump(best_pipe, f"models/{tag}twostage_multiclass_model.joblib")

    pd.DataFrame([test_metrics]).to_csv(
        f"reports/tables/{tag}twostage_test_performance.csv", index=False
    )

    print("\n=== Summary ===")
    cols = ["model", "feature_set", "macro_f1", "balanced_accuracy", "weighted_f1"]
    print(pd.DataFrame([test_metrics])[cols].to_string(index=False))


if __name__ == "__main__":
    main()
