"""Focused tests for model pipelines and train-only evaluation utilities."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from customer_churn_prediction.evaluation import (  # noqa: E402
    SCORING,
    evaluate_pipeline,
    make_cv,
    out_of_fold_predictions,
)
from customer_churn_prediction.modeling import (  # noqa: E402
    MODEL_NAMES,
    build_estimator,
    build_model_pipeline,
)
from customer_churn_prediction.preprocessing import (  # noqa: E402
    DEFAULT_DATASET,
    IDENTIFIER_COLUMN,
    TARGET_COLUMN,
    prepare_features,
    split_data,
)


class ModelingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = pd.read_csv(PROJECT_ROOT / DEFAULT_DATASET)
        cls.X, cls.y = prepare_features(cls.data)
        cls.split = split_data(cls.X, cls.y)
        sample_indices = cls.split.X_train.groupby(cls.split.y_train).head(300).index
        cls.X_sample = cls.split.X_train.loc[sample_indices]
        cls.y_sample = cls.split.y_train.loc[sample_indices]

    def test_all_full_pipelines_have_required_steps_and_predict(self) -> None:
        original = self.X_sample.copy(deep=True)
        for model_name in MODEL_NAMES:
            with self.subTest(model=model_name):
                pipeline = build_model_pipeline(model_name)
                self.assertEqual(list(pipeline.named_steps), ["feature_engineering", "preprocessor", "model"])
                pipeline.fit(self.X_sample, self.y_sample)
                predictions = pipeline.predict(self.X_sample.iloc[:20])
                probabilities = pipeline.predict_proba(self.X_sample.iloc[:20])
                self.assertEqual(predictions.shape, (20,))
                self.assertTrue(set(np.unique(predictions)).issubset({0, 1}))
                self.assertEqual(probabilities.shape, (20, 2))
                self.assertTrue(np.isfinite(probabilities).all())
        pd.testing.assert_frame_equal(self.X_sample, original)

    def test_estimator_configurations_are_untuned_and_reproducible(self) -> None:
        self.assertEqual(build_estimator("Dummy").get_params()["strategy"], "most_frequent")
        logistic = build_estimator("Logistic Regression")
        self.assertEqual(logistic.get_params()["max_iter"], 2000)
        self.assertIsNone(logistic.get_params()["class_weight"])
        tree = build_estimator("Decision Tree")
        self.assertEqual(tree.get_params()["random_state"], 42)
        forest = build_estimator("Random Forest")
        self.assertEqual(forest.get_params()["n_estimators"], 300)
        self.assertIsNone(forest.get_params()["class_weight"])
        boosting = build_estimator("Gradient Boosting")
        self.assertEqual(boosting.get_params()["random_state"], 42)

    def test_feature_matrix_excludes_identifier_and_target(self) -> None:
        self.assertNotIn(IDENTIFIER_COLUMN, self.X.columns)
        self.assertNotIn(TARGET_COLUMN, self.X.columns)

    def test_cv_protocol_is_stratified_reproducible_and_train_only(self) -> None:
        cv = make_cv()
        self.assertIsInstance(cv, StratifiedKFold)
        self.assertEqual(cv.n_splits, 5)
        self.assertTrue(cv.shuffle)
        self.assertEqual(cv.random_state, 42)
        self.assertTrue(self.split.X_train.index.isin(self.split.X_test.index).sum() == 0)

        small_cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
        scores = evaluate_pipeline(
            build_model_pipeline("Logistic Regression"),
            self.X_sample,
            self.y_sample,
            small_cv,
        )
        self.assertEqual(set(scores), {f"{metric}_{suffix}" for metric in SCORING for suffix in ["mean", "std"]})
        self.assertTrue(np.isfinite(list(scores.values())).all())
        for metric in SCORING:
            self.assertGreaterEqual(scores[f"{metric}_mean"], 0.0)
            self.assertLessEqual(scores[f"{metric}_mean"], 1.0)

    def test_predictions_are_reproducible(self) -> None:
        first = build_model_pipeline("Random Forest").fit(self.X_sample, self.y_sample)
        second = build_model_pipeline("Random Forest").fit(self.X_sample, self.y_sample)
        np.testing.assert_array_equal(
            first.predict(self.X_sample.iloc[:50]),
            second.predict(self.X_sample.iloc[:50]),
        )

    def test_out_of_fold_predictions_are_complete_and_finite(self) -> None:
        predictions, probabilities = out_of_fold_predictions(
            "Dummy", self.X_sample, self.y_sample
        )
        self.assertEqual(predictions.shape, (len(self.X_sample),))
        self.assertEqual(probabilities.shape, (len(self.X_sample),))
        self.assertTrue(set(np.unique(predictions)).issubset({0, 1}))
        self.assertTrue(np.isfinite(probabilities).all())


if __name__ == "__main__":
    unittest.main()
