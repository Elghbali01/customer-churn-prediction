"""Model definitions and complete leakage-safe ML pipelines."""

from __future__ import annotations

from collections.abc import Callable

from sklearn.base import ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from customer_churn_prediction.feature_engineering import ChurnFeatureEngineer
from customer_churn_prediction.preprocessing import build_base_preprocessor, build_preprocessor

RANDOM_STATE = 42
MODEL_NAMES = [
    "Dummy",
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "Gradient Boosting",
]


def _model_factories() -> dict[str, Callable[[], ClassifierMixin]]:
    return {
        "Dummy": lambda: DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": lambda: LogisticRegression(
            max_iter=2000,
            solver="lbfgs",
            random_state=RANDOM_STATE,
        ),
        "Decision Tree": lambda: DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": lambda: RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient Boosting": lambda: GradientBoostingClassifier(random_state=RANDOM_STATE),
    }


def build_estimator(model_name: str) -> ClassifierMixin:
    """Return a fresh, untuned estimator for the requested model family."""
    factories = _model_factories()
    if model_name not in factories:
        raise ValueError(f"Unknown model: {model_name}. Expected one of {MODEL_NAMES}")
    return factories[model_name]()


def build_model_pipeline(model_name: str, *, feature_engineering: bool = True) -> Pipeline:
    """Build a fresh full pipeline with identical preprocessing policy."""
    estimator = build_estimator(model_name)
    if feature_engineering:
        return Pipeline(
            steps=[
                ("feature_engineering", ChurnFeatureEngineer()),
                ("preprocessor", build_preprocessor()),
                ("model", estimator),
            ]
        )
    return Pipeline(
        steps=[
            ("preprocessor", build_base_preprocessor()),
            ("model", estimator),
        ]
    )
