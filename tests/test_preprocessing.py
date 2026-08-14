"""Focused tests for the leakage-safe preprocessing infrastructure."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from customer_churn_prediction.preprocessing import (  # noqa: E402
    BASE_FEATURES,
    BASE_NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURES,
    IDENTIFIER_COLUMN,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TARGET_COLUMN,
    TARGET_MAPPING,
    TEST_SIZE,
    build_feature_preprocessing_pipeline,
    prepare_features,
    split_data,
)


class PreprocessingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = pd.read_csv(PROJECT_ROOT / "data/processed/telco_customer_churn_clean.csv")
        cls.X, cls.y = prepare_features(cls.data)
        cls.split = split_data(cls.X, cls.y)
        cls.preprocessor = build_feature_preprocessing_pipeline()
        cls.preprocessor.fit(cls.split.X_train, cls.split.y_train)

    def test_target_mapping_is_exact(self) -> None:
        self.assertEqual(TARGET_MAPPING, {"No": 0, "Yes": 1})
        expected = self.data[TARGET_COLUMN].map(TARGET_MAPPING).astype("int8")
        pd.testing.assert_series_equal(self.y, expected.rename(TARGET_COLUMN))

    def test_explicit_feature_groups_are_complete_and_disjoint(self) -> None:
        self.assertEqual(BASE_NUMERIC_FEATURES, ["tenure", "MonthlyCharges", "TotalCharges"])
        self.assertIn("total_services", NUMERIC_FEATURES)
        self.assertIn("SeniorCitizen", CATEGORICAL_FEATURES)
        self.assertNotIn("SeniorCitizen", NUMERIC_FEATURES)
        self.assertFalse(set(NUMERIC_FEATURES) & set(CATEGORICAL_FEATURES))
        self.assertEqual(set(BASE_FEATURES), set(self.data.columns) - {IDENTIFIER_COLUMN, TARGET_COLUMN})
        self.assertNotIn(IDENTIFIER_COLUMN, self.X.columns)
        self.assertNotIn(TARGET_COLUMN, self.X.columns)

    def test_split_is_reproducible_and_stratified(self) -> None:
        repeated = split_data(self.X, self.y)
        self.assertEqual(TEST_SIZE, 0.20)
        self.assertEqual(RANDOM_STATE, 42)
        pd.testing.assert_frame_equal(self.split.X_train, repeated.X_train)
        pd.testing.assert_frame_equal(self.split.X_test, repeated.X_test)
        self.assertEqual((len(self.split.X_train), len(self.split.X_test)), (5634, 1409))
        global_rate = self.y.mean()
        self.assertLess(abs(self.split.y_train.mean() - global_rate), 0.001)
        self.assertLess(abs(self.split.y_test.mean() - global_rate), 0.001)

    def test_preprocessor_is_fitted_on_train_only(self) -> None:
        feature_engineer = self.preprocessor.named_steps["feature_engineering"]
        column_transformer = self.preprocessor.named_steps["preprocessor"]
        scaler = column_transformer.named_transformers_["numeric"].named_steps["scaler"]
        engineered_train = feature_engineer.transform(self.split.X_train)
        engineered_full = feature_engineer.transform(self.X)
        np.testing.assert_allclose(scaler.mean_, engineered_train[NUMERIC_FEATURES].mean())
        self.assertFalse(np.allclose(scaler.mean_, engineered_full[NUMERIC_FEATURES].mean()))

    def test_train_and_test_transform_without_missing_rows(self) -> None:
        train_transformed = self.preprocessor.transform(self.split.X_train)
        test_transformed = self.preprocessor.transform(self.split.X_test)
        self.assertEqual(train_transformed.shape[0], self.split.X_train.shape[0])
        self.assertEqual(test_transformed.shape[0], self.split.X_test.shape[0])
        self.assertEqual(train_transformed.shape[1], test_transformed.shape[1])
        self.assertFalse(np.isnan(train_transformed).any())
        self.assertFalse(np.isnan(test_transformed).any())

    def test_unknown_category_is_ignored(self) -> None:
        future = self.split.X_test.iloc[[0]].copy()
        future.loc[:, "gender"] = "Unknown future category"
        transformed = self.preprocessor.transform(future)
        self.assertEqual(transformed.shape, (1, self.preprocessor.get_feature_names_out().size))


if __name__ == "__main__":
    unittest.main()
