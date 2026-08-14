"""Deterministic and leakage-free feature engineering for churn prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

TENURE_GROUP_FEATURE = "tenure_group"
CONTRACT_TENURE_FEATURE = "contract_tenure"
INTERNET_CONTRACT_FEATURE = "internet_contract"
TOTAL_SERVICES_FEATURE = "total_services"

ENGINEERED_NUMERIC_FEATURES = [TOTAL_SERVICES_FEATURE]
ENGINEERED_CATEGORICAL_FEATURES = [
    TENURE_GROUP_FEATURE,
    CONTRACT_TENURE_FEATURE,
    INTERNET_CONTRACT_FEATURE,
]
ENGINEERED_FEATURES = ENGINEERED_CATEGORICAL_FEATURES + ENGINEERED_NUMERIC_FEATURES

SERVICE_COLUMNS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]
REQUIRED_SOURCE_COLUMNS = ["tenure", "Contract", "InternetService", *SERVICE_COLUMNS]
TENURE_BINS = [-np.inf, 12, 24, 48, np.inf]
TENURE_LABELS = ["0-12", "13-24", "25-48", "49+"]
VALID_SERVICE_VALUES = {"Yes", "No", "No internet service"}


class ChurnFeatureEngineer(TransformerMixin, BaseEstimator):
    """Add four interpretable features without learning from the target."""

    def fit(self, X: pd.DataFrame, y: object = None) -> "ChurnFeatureEngineer":
        """Validate the schema; no parameter is learned from X or y."""
        self._validate_input(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of X with deterministic engineered features."""
        check_is_fitted(self, attributes=["feature_names_in_", "n_features_in_"])
        self._validate_input(X)
        if X.columns.tolist() != self.feature_names_in_.tolist():
            raise ValueError("Input feature order differs from the fitted schema")

        transformed = X.copy(deep=True)
        transformed[TENURE_GROUP_FEATURE] = pd.cut(
            transformed["tenure"],
            bins=TENURE_BINS,
            labels=TENURE_LABELS,
            right=True,
        ).astype("object")
        transformed[CONTRACT_TENURE_FEATURE] = (
            transformed["Contract"].astype(str)
            + " | "
            + transformed[TENURE_GROUP_FEATURE].astype(str)
        )
        transformed[INTERNET_CONTRACT_FEATURE] = (
            transformed["InternetService"].astype(str)
            + " | "
            + transformed["Contract"].astype(str)
        )
        transformed[TOTAL_SERVICES_FEATURE] = (
            transformed[SERVICE_COLUMNS].eq("Yes").sum(axis=1).astype("int8")
        )

        if transformed[ENGINEERED_FEATURES].isna().any().any():
            raise ValueError("Feature engineering introduced missing values")
        numeric = transformed[ENGINEERED_NUMERIC_FEATURES].to_numpy(dtype=float)
        if not np.isfinite(numeric).all():
            raise ValueError("Feature engineering introduced non-finite values")
        return transformed

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        """Return original feature names followed by engineered names."""
        check_is_fitted(self, attributes=["feature_names_in_"])
        return np.concatenate([self.feature_names_in_, np.asarray(ENGINEERED_FEATURES, dtype=object)])

    @staticmethod
    def _validate_input(X: pd.DataFrame) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("ChurnFeatureEngineer requires a pandas DataFrame")
        missing = set(REQUIRED_SOURCE_COLUMNS) - set(X.columns)
        if missing:
            raise ValueError(f"Missing feature-engineering source columns: {sorted(missing)}")
        if X[REQUIRED_SOURCE_COLUMNS].isna().any().any():
            raise ValueError("Feature-engineering source columns contain missing values")
        tenure = X["tenure"].to_numpy(dtype=float)
        if not np.isfinite(tenure).all() or (tenure < 0).any():
            raise ValueError("tenure must contain finite non-negative values")
        invalid_services = {
            column: sorted(set(X[column].unique()) - VALID_SERVICE_VALUES)
            for column in SERVICE_COLUMNS
            if set(X[column].unique()) - VALID_SERVICE_VALUES
        }
        if invalid_services:
            raise ValueError(f"Unexpected service categories: {invalid_services}")
