"""Centralized, one-time loading of the frozen model and public metadata."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "churn_pipeline.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"


@dataclass(frozen=True)
class ModelResources:
    pipeline: Pipeline
    metadata: dict[str, Any]


def load_model_resources(
    model_path: Path = MODEL_PATH,
    metadata_path: Path = METADATA_PATH,
) -> ModelResources:
    """Load and validate artifacts once; raise an explicit startup error."""
    if not model_path.is_file():
        raise RuntimeError(f"Model artifact not found: {model_path}")
    if not metadata_path.is_file():
        raise RuntimeError(f"Model metadata not found: {metadata_path}")
    try:
        pipeline = joblib.load(model_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Unable to load model resources: {exc}") from exc
    required = {"model_name", "model_version", "threshold", "positive_class", "feature_count", "training_rows", "metrics_test"}
    missing = required - set(metadata)
    if missing:
        raise RuntimeError(f"Model metadata is incomplete: {sorted(missing)}")
    if float(metadata["threshold"]) != 0.30:
        raise RuntimeError("Unexpected operational threshold in metadata")
    if not hasattr(pipeline, "predict_proba"):
        raise RuntimeError("Loaded artifact does not support predict_proba")
    return ModelResources(pipeline=pipeline, metadata=metadata)
