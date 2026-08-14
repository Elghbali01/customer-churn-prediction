"""FastAPI application exposing the frozen customer churn model."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.dependencies import ModelResources, load_model_resources
from api.schemas import (
    BatchRequest, BatchResponse, ChurnRequest, ChurnResponse,
    HealthResponse, ModelInfoResponse,
)
from customer_churn_prediction.final_pipeline import predict_churn


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model_resources = load_model_resources()
    yield
    app.state.model_resources = None


app = FastAPI(
    title="Customer Churn Prediction API",
    description="Inference API for the frozen telecom churn prediction pipeline.",
    version="1.0.0",
    lifespan=lifespan,
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _resources(request: Request) -> ModelResources:
    resources = getattr(request.app.state, "model_resources", None)
    if resources is None:
        raise HTTPException(status_code=500, detail="Model resources are not available")
    return resources


def _predict(clients: list[ChurnRequest], resources: ModelResources) -> list[ChurnResponse]:
    frame = pd.DataFrame([client.model_dump() for client in clients])
    try:
        predictions = predict_churn(
            frame,
            pipeline=resources.pipeline,
            threshold=float(resources.metadata["threshold"]),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Model inference failed") from exc
    return [
        ChurnResponse(
            churn_probability=float(row.churn_probability),
            churn_prediction=int(row.churn_prediction),
            churn_label="Churn" if int(row.churn_prediction) == 1 else "No Churn",
            threshold=float(row.threshold),
        )
        for row in predictions.itertuples(index=False)
    ]


@app.get("/", response_class=FileResponse, include_in_schema=False)
def root() -> FileResponse:
    """Serve the user-facing prediction interface."""
    return FileResponse(STATIC_DIR / "index.html", media_type="text/html")


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health(request: Request) -> HealthResponse:
    resources = _resources(request)
    return HealthResponse(status="ok", model_loaded=True, model_version=str(resources.metadata["model_version"]))


@app.get("/model-info", response_model=ModelInfoResponse, tags=["model"])
def model_info(request: Request) -> ModelInfoResponse:
    metadata = _resources(request).metadata
    return ModelInfoResponse(
        model_name=str(metadata["model_name"]), model_version=str(metadata["model_version"]),
        threshold=float(metadata["threshold"]), positive_class=int(metadata["positive_class"]),
        feature_count=int(metadata["feature_count"]), training_rows=int(metadata["training_rows"]),
        test_metrics=metadata["metrics_test"],
    )


@app.post("/predict", response_model=ChurnResponse, tags=["prediction"])
def predict(client: ChurnRequest, request: Request) -> ChurnResponse:
    return _predict([client], _resources(request))[0]


@app.post("/predict/batch", response_model=BatchResponse, tags=["prediction"])
def predict_batch(batch: BatchRequest, request: Request) -> BatchResponse:
    predictions = _predict(batch.clients, _resources(request))
    return BatchResponse(count=len(predictions), predictions=predictions)
