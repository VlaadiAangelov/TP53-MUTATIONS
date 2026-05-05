from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

warnings.filterwarnings(
    "ignore",
    message=".*encountered in matmul",
    category=RuntimeWarning,
)

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import (
    ID_COLUMN,
    TARGET_COLUMN,
    build_labeled_dataset,
    load_expression_data,
    load_mutation_data,
)


def load_top_feature_names(path: Path) -> List[str]:
    top_features_df = pd.read_csv(path)
    if "feature" not in top_features_df.columns:
        raise ValueError("Top feature file must include a 'feature' column.")
    return top_features_df["feature"].dropna().astype(str).tolist()


def build_modeling_table(expression_csv: Path, mutation_csv: Path, top_features_csv: Path) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    expression_df = load_expression_data(expression_csv)
    mutation_df = load_mutation_data(mutation_csv)
    dataset = build_labeled_dataset(expression_df, mutation_df)
    feature_names = load_top_feature_names(top_features_csv)

    missing_features = sorted(set(feature_names).difference(dataset.columns))
    if missing_features:
        preview = ", ".join(missing_features[:5])
        raise ValueError(f"Top feature file references missing expression columns: {preview}")

    features = dataset[feature_names].select_dtypes(include="number")
    dropped_non_numeric = sorted(set(feature_names).difference(features.columns))
    if dropped_non_numeric:
        raise ValueError(f"Top feature file includes non-numeric columns: {dropped_non_numeric[:5]}")

    return dataset[ID_COLUMN], features, dataset[TARGET_COLUMN]


def evaluate_classifier(model: Any, x_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*encountered in matmul",
            category=RuntimeWarning,
        )
        y_pred = model.predict(x_test)
        y_score = model.predict_proba(x_test)[:, 1]
    labels = [0, 1]
    matrix = confusion_matrix(y_test, y_pred, labels=labels)

    return {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 6),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_test, y_pred)), 6),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 6),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 6),
        "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 6),
        "roc_auc": round(float(roc_auc_score(y_test, y_score)), 6),
        "confusion_matrix": {
            "labels": labels,
            "values": matrix.tolist(),
        },
    }


def train_top_features_model(
    expression_csv: Path,
    mutation_csv: Path,
    top_features_csv: Path,
    output_dir: Path,
    test_size: float = 0.25,
    random_state: int = 42,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    depmap_ids, features, target = build_modeling_table(expression_csv, mutation_csv, top_features_csv)

    x_train, x_test, y_train, y_test, ids_train, ids_test = train_test_split(
        features,
        target,
        depmap_ids,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )

    dummy_model = DummyClassifier(strategy="stratified", random_state=random_state)
    dummy_model.fit(x_train, y_train)

    logistic_model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=0.1,
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=random_state,
                    solver="liblinear",
                ),
            ),
        ]
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*encountered in matmul",
            category=RuntimeWarning,
        )
        logistic_model.fit(x_train, y_train)

    metrics = {
        "dataset": {
            "samples": int(len(features)),
            "features": int(features.shape[1]),
            "positive_samples": int(target.sum()),
            "negative_samples": int((target == 0).sum()),
            "test_size": test_size,
            "random_state": random_state,
            "model": "StandardScaler + LogisticRegression(C=0.1, class_weight='balanced', solver='liblinear')",
            "note": "Top features were selected before this split, so treat scores as exploratory.",
        },
        "dummy_stratified": evaluate_classifier(dummy_model, x_test, y_test),
        "logistic_regression_top500": evaluate_classifier(logistic_model, x_test, y_test),
    }

    y_pred = logistic_model.predict(x_test)
    y_score = logistic_model.predict_proba(x_test)[:, 1]
    predictions_df = pd.DataFrame(
        {
            ID_COLUMN: ids_test.to_numpy(),
            "true_status": y_test.to_numpy(),
            "predicted_status": y_pred,
            "predicted_probability": y_score,
        }
    )

    classifier = logistic_model.named_steps["classifier"]
    coefficients_df = pd.DataFrame(
        {
            "feature": features.columns,
            "coefficient": classifier.coef_[0],
        }
    )
    coefficients_df["abs_coefficient"] = coefficients_df["coefficient"].abs()
    coefficients_df = coefficients_df.sort_values("abs_coefficient", ascending=False)

    metrics_path = output_dir / "model_metrics.json"
    predictions_path = output_dir / "test_predictions.csv"
    coefficients_path = output_dir / "logistic_coefficients.csv"

    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    predictions_df.to_csv(predictions_path, index=False)
    coefficients_df.to_csv(coefficients_path, index=False)

    return {
        "metrics": metrics,
        "metrics_path": metrics_path,
        "predictions_path": predictions_path,
        "coefficients_path": coefficients_path,
    }
