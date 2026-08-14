"""Tests for deterministic, target-independent feature engineering."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from customer_churn_prediction.feature_engineering import (  # noqa: E402
    ChurnFeatureEngineer,
    ENGINEERED_FEATURES,
    SERVICE_COLUMNS,
)
from customer_churn_prediction.preprocessing import (  # noqa: E402
    BASE_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURES,
    IDENTIFIER_COLUMN,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    build_feature_preprocessing_pipeline,
    prepare_features,
    split_data,
)


class FeatureEngineeringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = pd.read_csv(PROJECT_ROOT / "data/processed/telco_customer_churn_clean.csv")
        cls.X, cls.y = prepare_features(cls.data)
        cls.transformer = ChurnFeatureEngineer().fit(cls.X)
        cls.engineered = cls.transformer.transform(cls.X)

    def test_features_are_created_deterministically_without_mutation(self) -> None:
        original = self.X.copy(deep=True)
        repeated = self.transformer.transform(self.X)
        pd.testing.assert_frame_equal(self.X, original)
        pd.testing.assert_frame_equal(self.engineered, repeated)
        self.assertEqual(self.engineered.shape, (7043, 23))
        self.assertEqual(self.engineered.columns[-4:].tolist(), ENGINEERED_FEATURES)

    def test_tenure_group_boundaries(self) -> None:
        sample = self.X.iloc[[0] * 8].copy()
        sample.loc[:, "tenure"] = [0, 12, 13, 24, 25, 48, 49, 72]
        result = ChurnFeatureEngineer().fit_transform(sample)
        self.assertEqual(
            result["tenure_group"].tolist(),
            ["0-12", "0-12", "13-24", "13-24", "25-48", "25-48", "49+", "49+"],
        )

    def test_interactions_are_exact(self) -> None:
        row = self.X.iloc[[0]].copy()
        row.loc[:, ["tenure", "Contract", "InternetService"]] = [6, "Month-to-month", "Fiber optic"]
        result = ChurnFeatureEngineer().fit_transform(row).iloc[0]
        self.assertEqual(result["contract_tenure"], "Month-to-month | 0-12")
        self.assertEqual(result["internet_contract"], "Fiber optic | Month-to-month")

    def test_total_services_counts_only_yes(self) -> None:
        row = self.X.iloc[[0]].copy()
        row.loc[:, SERVICE_COLUMNS] = ["Yes", "No", "No internet service", "Yes", "No", "Yes"]
        result = ChurnFeatureEngineer().fit_transform(row)
        self.assertEqual(int(result["total_services"].iloc[0]), 3)

        no_internet = self.X.loc[self.X["InternetService"].eq("No")].iloc[[0]]
        result_no_internet = ChurnFeatureEngineer().fit_transform(no_internet)
        self.assertEqual(int(result_no_internet["total_services"].iloc[0]), 0)

    def test_output_has_no_missing_or_infinite_values(self) -> None:
        self.assertFalse(self.engineered.isna().any().any())
        numeric = self.engineered[NUMERIC_FEATURES].to_numpy(dtype=float)
        self.assertTrue(np.isfinite(numeric).all())
        self.assertEqual(len(self.engineered), len(self.X))

    def test_feature_selection_and_pipeline_integration(self) -> None:
        self.assertNotIn(IDENTIFIER_COLUMN, BASE_FEATURES)
        self.assertNotIn(TARGET_COLUMN, BASE_FEATURES)
        self.assertEqual(len(BASE_FEATURES), 19)
        self.assertEqual(len(FEATURES), 23)
        self.assertIn("total_services", NUMERIC_FEATURES)
        for feature in ["tenure_group", "contract_tenure", "internet_contract"]:
            self.assertIn(feature, CATEGORICAL_FEATURES)

        split = split_data(self.X, self.y)
        pipeline = build_feature_preprocessing_pipeline()
        pipeline.fit(split.X_train, split.y_train)
        train = pipeline.transform(split.X_train)
        test = pipeline.transform(split.X_test)
        self.assertEqual(train.shape[0], 5634)
        self.assertEqual(test.shape[0], 1409)
        self.assertEqual(train.shape[1], test.shape[1])
        self.assertFalse(np.isnan(train).any())
        self.assertFalse(np.isnan(test).any())

    def test_feature_engineering_does_not_depend_on_target(self) -> None:
        reversed_target = 1 - self.y
        first = ChurnFeatureEngineer().fit_transform(self.X, self.y)
        second = ChurnFeatureEngineer().fit_transform(self.X, reversed_target)
        pd.testing.assert_frame_equal(first, second)


if __name__ == "__main__":
    unittest.main()
