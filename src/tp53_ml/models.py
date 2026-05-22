from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from .genes import P53_PATHWAY_GENES, TP53_TARGET_GENES
from .preprocessing import GeneSetSelector, TopVarianceSelector


def feature_selectors(k: int, min_mean_expression: float) -> dict:
    return {
        f"top_{k}_variable": TopVarianceSelector(k=k, min_mean_expression=min_mean_expression),
        "tp53_targets": GeneSetSelector(TP53_TARGET_GENES),
        "p53_pathway": GeneSetSelector(P53_PATHWAY_GENES),
    }


def binary_model_specs(random_state: int = 42) -> dict:
    return {
        "majority": DummyClassifier(strategy="most_frequent"),
        "logistic_l2": LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=5000,
            solver="liblinear",
            random_state=random_state,
        ),
        "elastic_net_logistic": LogisticRegression(
            C=0.2,
            l1_ratio=0.5,
            class_weight="balanced",
            max_iter=5000,
            penalty="elasticnet",
            solver="saga",
            random_state=random_state,
        ),
        "linear_svm": LinearSVC(class_weight="balanced", C=0.2, random_state=random_state, max_iter=10000),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=500,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=250,
            l2_regularization=0.1,
            random_state=random_state,
        ),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(256, 64),
            activation="relu",
            alpha=0.001,
            early_stopping=True,
            max_iter=300,
            random_state=random_state,
        ),
    }


def multiclass_model_specs(random_state: int = 42) -> dict:
    return {
        "majority": DummyClassifier(strategy="most_frequent"),
        "logistic_l2": LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=5000,
            solver="lbfgs",
            random_state=random_state,
        ),
        "elastic_net_logistic": LogisticRegression(
            C=0.2,
            l1_ratio=0.5,
            class_weight="balanced",
            max_iter=5000,
            solver="saga",
            random_state=random_state,
        ),
        "linear_svm": LinearSVC(
            class_weight="balanced",
            C=0.2,
            random_state=random_state,
            max_iter=10000,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=500,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=250,
            l2_regularization=0.1,
            random_state=random_state,
        ),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(256, 64),
            activation="relu",
            alpha=0.001,
            max_iter=300,
            random_state=random_state,
        ),
    }


def make_pipeline(selector, model, scale: bool) -> Pipeline:
    steps = [("selector", clone(selector))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", clone(model)))
    return Pipeline(steps)


def positive_class_scores(pipeline: Pipeline, X) -> np.ndarray:
    model = pipeline.named_steps["model"]
    if hasattr(pipeline, "predict_proba") and hasattr(model, "predict_proba"):
        return pipeline.predict_proba(X)[:, 1]
    if hasattr(pipeline, "decision_function"):
        scores = pipeline.decision_function(X)
        return np.asarray(scores, dtype=float)
    return pipeline.predict(X).astype(float)


def selected_feature_names(pipeline: Pipeline) -> list[str]:
    selector = pipeline.named_steps["selector"]
    if hasattr(selector, "get_feature_names_out"):
        return list(selector.get_feature_names_out())
    return []
