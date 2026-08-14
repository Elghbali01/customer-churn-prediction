"""Tests for the frozen final pipeline and serialized prediction contract."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from customer_churn_prediction.feature_engineering import ChurnFeatureEngineer  # noqa: E402
from customer_churn_prediction.final_pipeline import (  # noqa: E402
    DEFAULT_METADATA_PATH, DEFAULT_MODEL_PATH, DEFAULT_REPORT_PATH,
    OPERATIONAL_THRESHOLD, build_final_pipeline, coefficient_table,
    predict_churn, transformed_features,
)
from customer_churn_prediction.preprocessing import DEFAULT_DATASET, prepare_features, split_data  # noqa: E402


class FinalPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        data = pd.read_csv(PROJECT_ROOT / DEFAULT_DATASET)
        X, y = prepare_features(data); cls.split = split_data(X, y)
        cls.pipeline = build_final_pipeline().fit(cls.split.X_train, cls.split.y_train)

    def test_exact_frozen_architecture_and_parameters(self) -> None:
        self.assertEqual(list(self.pipeline.named_steps), ["feature_engineering", "preprocessor", "model"])
        self.assertIsInstance(self.pipeline.named_steps["feature_engineering"], ChurnFeatureEngineer)
        model = self.pipeline.named_steps["model"]
        self.assertIsInstance(model, LogisticRegression); self.assertEqual(model.C, 2.0)
        self.assertEqual(model.solver, "lbfgs"); self.assertIsNone(model.class_weight)
        self.assertEqual(OPERATIONAL_THRESHOLD, 0.30)

    def test_probabilities_threshold_and_reusable_prediction(self) -> None:
        raw = self.split.X_test.iloc[:12]
        result = predict_churn(raw, pipeline=self.pipeline)
        expected = self.pipeline.predict_proba(raw)[:, 1]
        np.testing.assert_allclose(result["churn_probability"], expected)
        np.testing.assert_array_equal(result["churn_prediction"], expected >= 0.30)
        self.assertTrue(((expected >= 0) & (expected <= 1)).all())

    def test_feature_coefficient_mapping(self) -> None:
        transformed, names = transformed_features(self.pipeline, self.split.X_test.iloc[:20])
        table = coefficient_table(self.pipeline, names)
        self.assertEqual(transformed.shape[1], len(names)); self.assertEqual(len(names), len(table))
        self.assertEqual(len(names), self.pipeline.named_steps["model"].coef_.shape[1])
        self.assertTrue(np.isfinite(table.select_dtypes("number").to_numpy()).all())

    def test_final_report_and_shap_mapping(self) -> None:
        payload = json.loads((PROJECT_ROOT / DEFAULT_REPORT_PATH).read_text(encoding="utf-8"))
        self.assertEqual(payload["protocol"]["fit_scope"], "train_only")
        self.assertTrue(payload["protocol"]["test_set_consumed_for_final_evaluation"])
        self.assertFalse(payload["protocol"]["selection_after_test"])
        self.assertEqual(payload["shap"]["shape"], [1409, payload["feature_count"]])
        shap_table = pd.read_csv(PROJECT_ROOT / "reports/final_shap_importance.csv")
        self.assertEqual(len(shap_table), payload["feature_count"])
        self.assertTrue(np.isfinite(shap_table["mean_abs_shap"]).all())

    def test_reported_test_metrics_match_frozen_predictions(self) -> None:
        probabilities = self.pipeline.predict_proba(self.split.X_test)[:, 1]
        predictions = probabilities >= OPERATIONAL_THRESHOLD
        tn, fp, fn, tp = confusion_matrix(self.split.y_test, predictions, labels=[0, 1]).ravel()
        report = json.loads((PROJECT_ROOT / DEFAULT_REPORT_PATH).read_text(encoding="utf-8"))["test"]["thresholds"]["0.30"]
        self.assertEqual((report["tn"], report["fp"], report["fn"], report["tp"]), (tn, fp, fn, tp))
        self.assertAlmostEqual(report["precision"], precision_score(self.split.y_test, predictions))
        self.assertAlmostEqual(report["recall"], recall_score(self.split.y_test, predictions))

    def test_serialization_round_trip_and_metadata(self) -> None:
        loaded = joblib.load(PROJECT_ROOT / DEFAULT_MODEL_PATH)
        raw = self.split.X_test.iloc[:10]
        before = self.pipeline.predict_proba(raw)[:, 1]; after = loaded.predict_proba(raw)[:, 1]
        np.testing.assert_array_equal(before, after)
        metadata = json.loads((PROJECT_ROOT / DEFAULT_METADATA_PATH).read_text(encoding="utf-8"))
        self.assertEqual(metadata["threshold"], 0.30); self.assertEqual(metadata["training_scope"], "train_only")


if __name__ == "__main__":
    unittest.main()
