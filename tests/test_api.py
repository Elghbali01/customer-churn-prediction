"""HTTP contract and equivalence tests for the frozen churn API."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from api.dependencies import MODEL_PATH, load_model_resources  # noqa: E402
from api.main import app  # noqa: E402
from api.schemas import EXAMPLE_CLIENT  # noqa: E402
from customer_churn_prediction.final_pipeline import predict_churn  # noqa: E402


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.context = TestClient(app)
        cls.client = cls.context.__enter__()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.context.__exit__(None, None, None)

    def test_root_health_and_model_info(self) -> None:
        root = self.client.get("/")
        self.assertEqual(root.status_code, 200)
        self.assertIn("text/html", root.headers["content-type"])
        self.assertIn("Customer Churn Prediction", root.text)
        health = self.client.get("/health")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json(), {"status": "ok", "model_loaded": True, "model_version": "1.0.0"})
        info = self.client.get("/model-info")
        self.assertEqual(info.status_code, 200)
        self.assertEqual(info.json()["threshold"], 0.30)
        self.assertEqual(info.json()["feature_count"], 72)

    def test_valid_prediction_contract_and_python_equivalence(self) -> None:
        response = self.client.post("/predict", json=EXAMPLE_CLIENT)
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(0 <= body["churn_probability"] <= 1)
        self.assertEqual(body["churn_prediction"], int(body["churn_probability"] >= 0.30))
        self.assertEqual(body["churn_label"], "Churn" if body["churn_prediction"] else "No Churn")
        self.assertEqual(body["threshold"], 0.30)
        python_result = predict_churn(pd.DataFrame([EXAMPLE_CLIENT])).iloc[0]
        self.assertAlmostEqual(body["churn_probability"], python_result.churn_probability, places=15)
        self.assertEqual(body["churn_prediction"], python_result.churn_prediction)

    def test_missing_invalid_type_category_numeric_and_extra_fields(self) -> None:
        cases = []
        missing = dict(EXAMPLE_CLIENT); missing.pop("tenure"); cases.append(missing)
        invalid_type = dict(EXAMPLE_CLIENT, tenure="abc"); cases.append(invalid_type)
        invalid_category = dict(EXAMPLE_CLIENT, gender="Unknown"); cases.append(invalid_category)
        invalid_numeric = dict(EXAMPLE_CLIENT, tenure=-1); cases.append(invalid_numeric)
        forbidden = dict(EXAMPLE_CLIENT, customerID="secret", Churn="Yes"); cases.append(forbidden)
        for payload in cases:
            with self.subTest(payload=payload):
                self.assertEqual(self.client.post("/predict", json=payload).status_code, 422)

    def test_phone_and_internet_consistency_rules(self) -> None:
        invalid_phone = dict(EXAMPLE_CLIENT, PhoneService="No", MultipleLines="Yes")
        invalid_internet = dict(EXAMPLE_CLIENT, InternetService="No", OnlineSecurity="Yes")
        active_with_absent_service = dict(EXAMPLE_CLIENT, OnlineSecurity="No internet service")
        for payload in [invalid_phone, invalid_internet, active_with_absent_service]:
            with self.subTest(payload=payload):
                self.assertEqual(self.client.post("/predict", json=payload).status_code, 422)

    def test_batch_order_limits_and_unit_equivalence(self) -> None:
        second = dict(EXAMPLE_CLIENT, tenure=60, TotalCharges=5000.0, Contract="Two year")
        batch = self.client.post("/predict/batch", json={"clients": [EXAMPLE_CLIENT, second]})
        self.assertEqual(batch.status_code, 200)
        body = batch.json(); self.assertEqual(body["count"], 2)
        singles = [self.client.post("/predict", json=item).json() for item in [EXAMPLE_CLIENT, second]]
        for batch_prediction, single_prediction in zip(body["predictions"], singles):
            self.assertAlmostEqual(batch_prediction["churn_probability"], single_prediction["churn_probability"], places=15)
            self.assertEqual(batch_prediction["churn_prediction"], single_prediction["churn_prediction"])
            self.assertEqual(batch_prediction["churn_label"], single_prediction["churn_label"])
            self.assertEqual(batch_prediction["threshold"], single_prediction["threshold"])
        self.assertEqual(self.client.post("/predict/batch", json={"clients": []}).status_code, 422)
        self.assertEqual(self.client.post("/predict/batch", json={"clients": [EXAMPLE_CLIENT] * 101}).status_code, 422)
        invalid = dict(EXAMPLE_CLIENT, tenure=-1)
        self.assertEqual(self.client.post("/predict/batch", json={"clients": [EXAMPLE_CLIENT, invalid]}).status_code, 422)

    def test_missing_artifact_fails_explicitly(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "Model artifact not found"):
            load_model_resources(MODEL_PATH.with_name("missing.joblib"))

    def test_openapi_contains_prediction_routes(self) -> None:
        response = self.client.get("/openapi.json")
        self.assertEqual(response.status_code, 200)
        paths = response.json()["paths"]
        self.assertTrue({"/health", "/model-info", "/predict", "/predict/batch"}.issubset(paths))

    def test_user_interface_assets_are_served(self) -> None:
        css = self.client.get("/static/css/app.css")
        javascript = self.client.get("/static/js/app.js")
        self.assertEqual(css.status_code, 200)
        self.assertEqual(javascript.status_code, 200)
        self.assertIn("@media (max-width: 680px)", css.text)
        self.assertIn('fetch("/predict"', javascript.text)
        self.assertIn("const EXAMPLE", javascript.text)


if __name__ == "__main__":
    unittest.main()
