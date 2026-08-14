"""Leakage-safe preprocessing infrastructure for future churn models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from customer_churn_prediction.feature_engineering import (
    ChurnFeatureEngineer,
    ENGINEERED_CATEGORICAL_FEATURES,
    ENGINEERED_NUMERIC_FEATURES,
)

DEFAULT_DATASET = Path("data/processed/telco_customer_churn_clean.csv")
TARGET_COLUMN = "Churn"
IDENTIFIER_COLUMN = "customerID"
TARGET_MAPPING = {"No": 0, "Yes": 1}
BASE_NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
BASE_CATEGORICAL_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
BASE_FEATURES = BASE_NUMERIC_FEATURES + BASE_CATEGORICAL_FEATURES
NUMERIC_FEATURES = BASE_NUMERIC_FEATURES + ENGINEERED_NUMERIC_FEATURES
CATEGORICAL_FEATURES = BASE_CATEGORICAL_FEATURES + ENGINEERED_CATEGORICAL_FEATURES
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TEST_SIZE = 0.20
RANDOM_STATE = 42


@dataclass(frozen=True)
class DataSplit:
    """Container for the reproducible train/test protocol."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def validate_feature_groups() -> None:
    """Ensure explicit feature groups are complete, unique, and leakage-free."""
    numeric = set(NUMERIC_FEATURES)
    categorical = set(CATEGORICAL_FEATURES)
    if numeric & categorical:
        raise ValueError("Numeric and categorical feature groups overlap")
    if len(FEATURES) != len(set(FEATURES)):
        raise ValueError("Feature lists contain duplicates")
    forbidden = {IDENTIFIER_COLUMN, TARGET_COLUMN}
    if forbidden & set(FEATURES):
        raise ValueError("Identifier or target present in model features")


def prepare_features(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Create X and encoded y while keeping the source dataframe unchanged."""
    validate_feature_groups()
    required = set(BASE_FEATURES) | {IDENTIFIER_COLUMN, TARGET_COLUMN}
    missing = required - set(data.columns)
    unexpected = set(data.columns) - required
    if missing or unexpected:
        raise ValueError(f"Schema mismatch; missing={sorted(missing)}, unexpected={sorted(unexpected)}")

    modalities = set(data[TARGET_COLUMN].dropna().unique())
    if modalities != set(TARGET_MAPPING):
        raise ValueError(f"Unexpected target modalities: {sorted(modalities)}")
    if data[TARGET_COLUMN].isna().any():
        raise ValueError("Target contains missing values")

    X = data.loc[:, BASE_FEATURES].copy()
    y = data[TARGET_COLUMN].map(TARGET_MAPPING).astype("int8").rename(TARGET_COLUMN)
    if X.columns.tolist() != BASE_FEATURES:
        raise RuntimeError("Feature order changed unexpectedly")
    return X, y


def split_data(X: pd.DataFrame, y: pd.Series) -> DataSplit:
    """Create the initial 80/20 stratified and reproducible holdout split."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    return DataSplit(X_train, X_test, y_train, y_test)


def _build_column_transformer(
    numeric_features: list[str], categorical_features: list[str]
) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[("scaler", StandardScaler())]
    )
    categorical_pipeline = Pipeline(
        steps=[
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
            )
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_preprocessor() -> ColumnTransformer:
    """Build an unfitted transformer for base and engineered features."""
    return _build_column_transformer(NUMERIC_FEATURES, CATEGORICAL_FEATURES)


def build_base_preprocessor() -> ColumnTransformer:
    """Build the original preprocessing used for controlled FE comparison."""
    return _build_column_transformer(BASE_NUMERIC_FEATURES, BASE_CATEGORICAL_FEATURES)


def build_feature_preprocessing_pipeline() -> Pipeline:
    """Build the future-ready deterministic-features then preprocessing pipeline."""
    return Pipeline(
        steps=[
            ("feature_engineering", ChurnFeatureEngineer()),
            ("preprocessor", build_preprocessor()),
        ]
    )


def prepare_train_test(
    dataset: Path = DEFAULT_DATASET,
) -> tuple[DataSplit, Pipeline]:
    """Load, split, then fit preprocessing on train only to prevent leakage."""
    data = pd.read_csv(dataset)
    X, y = prepare_features(data)
    split = split_data(X, y)
    pipeline = build_feature_preprocessing_pipeline()
    pipeline.fit(split.X_train, split.y_train)
    return split, pipeline
