"""Tests for train-only tuning, OOF probabilities, and threshold selection."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from customer_churn_prediction.improvement import (  # noqa: E402
    GRADIENT_N_ITER,
    GRADIENT_PARAM_DISTRIBUTIONS,
    LOGISTIC_PARAM_GRID,
    SELECTED_THRESHOLD,
    THRESHOLDS,
    build_gradient_search,
    build_logistic_search,
    build_selected_candidate,
    oof_probabilities,
    select_threshold,
    threshold_metrics,
)
from customer_churn_prediction.preprocessing import (  # noqa: E402
    DEFAULT_DATASET,
    prepare_features,
    split_data,
)


class ImprovementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        data = pd.read_csv(PROJECT_ROOT / DEFAULT_DATASET)
        cls.X, cls.y = prepare_features(data)
        cls.split = split_data(cls.X, cls.y)
        sample_indices = cls.split.X_train.groupby(cls.split.y_train).head(150).index
        cls.X_sample = cls.split.X_train.loc[sample_indices]
        cls.y_sample = cls.split.y_train.loc[sample_indices]

    def test_logistic_grid_has_only_compatible_combinations(self) -> None:
        self.assertEqual(len(LOGISTIC_PARAM_GRID), 2)
        for grid in LOGISTIC_PARAM_GRID:
            solvers = set(grid["model__solver"])
            penalties = set(grid["model__penalty"])
            if "lbfgs" in solvers:
                self.assertEqual(penalties, {"l2"})
            if "liblinear" in solvers:
                self.assertTrue(penalties.issubset({"l1", "l2"}))
        search = build_logistic_search()
        self.assertEqual(list(search.estimator.named_steps), ["feature_engineering", "preprocessor", "model"])
        self.assertEqual(search.refit, "average_precision")

    def test_gradient_search_is_bounded_and_reproducible(self) -> None:
        self.assertEqual(GRADIENT_N_ITER, 16)
        self.assertEqual(set(GRADIENT_PARAM_DISTRIBUTIONS), {
            "model__n_estimators", "model__learning_rate", "model__max_depth",
            "model__min_samples_split", "model__min_samples_leaf", "model__subsample",
        })
        search = build_gradient_search()
        self.assertEqual(search.random_state, 42)
        self.assertEqual(search.n_iter, 16)
        self.assertEqual(list(search.estimator.named_steps), ["feature_engineering", "preprocessor", "model"])

    def test_oof_probabilities_are_complete_bounded_and_reproducible(self) -> None:
        pipeline = build_selected_candidate()
        first = oof_probabilities(pipeline, self.X_sample, self.y_sample)
        second = oof_probabilities(build_selected_candidate(), self.X_sample, self.y_sample)
        self.assertEqual(first.shape, (len(self.X_sample),))
        self.assertTrue(np.isfinite(first).all())
        self.assertTrue(((first >= 0) & (first <= 1)).all())
        np.testing.assert_allclose(first, second)

    def test_threshold_metrics_and_selection_are_correct(self) -> None:
        y = pd.Series([0, 0, 1, 1])
        probabilities = np.array([0.1, 0.4, 0.3, 0.9])
        table = threshold_metrics(y, probabilities)
        self.assertTrue(np.array_equal(table["threshold"].to_numpy(), THRESHOLDS))
        row = table.loc[table["threshold"].eq(0.5)].iloc[0]
        self.assertEqual((row.tn, row.fp, row.fn, row.tp), (2, 0, 1, 1))
        self.assertEqual(select_threshold(table), 0.3)
        self.assertFalse(table.isna().any().any())
        self.assertTrue(np.isfinite(table.select_dtypes("number").to_numpy()).all())

    def test_selected_candidate_is_constructible(self) -> None:
        pipeline = build_selected_candidate()
        self.assertEqual(list(pipeline.named_steps), ["feature_engineering", "preprocessor", "model"])
        self.assertEqual(pipeline.get_params()["model__C"], 2.0)
        self.assertIsNone(pipeline.get_params()["model__class_weight"])
        self.assertEqual(SELECTED_THRESHOLD, 0.30)
        pipeline.fit(self.X_sample, self.y_sample)
        predictions = pipeline.predict(self.X_sample.iloc[:10])
        self.assertTrue(set(predictions).issubset({0, 1}))

    def test_report_confirms_test_set_not_consumed(self) -> None:
        payload = json.loads((PROJECT_ROOT / "reports/model_improvement.json").read_text(encoding="utf-8"))
        self.assertFalse(payload["protocol"]["test_set_consumed"])
        self.assertEqual(payload["protocol"]["train_rows"], 5634)
        self.assertEqual(payload["protocol"]["reserved_test_rows"], 1409)


if __name__ == "__main__":
    unittest.main()
