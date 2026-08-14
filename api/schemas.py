"""Strict HTTP request and response contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

YesNo = Literal["Yes", "No"]
InternetOption = Literal["Yes", "No", "No internet service"]

EXAMPLE_CLIENT = {
    "tenure": 5,
    "MonthlyCharges": 89.9,
    "TotalCharges": 450.5,
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
}


class ChurnRequest(BaseModel):
    """Exactly the 19 raw features accepted by the serialized pipeline."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [EXAMPLE_CLIENT]})

    tenure: int = Field(ge=0)
    MonthlyCharges: float = Field(ge=0, allow_inf_nan=False)
    TotalCharges: float = Field(ge=0, allow_inf_nan=False)
    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1]
    Partner: YesNo
    Dependents: YesNo
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: InternetOption
    OnlineBackup: InternetOption
    DeviceProtection: InternetOption
    TechSupport: InternetOption
    StreamingTV: InternetOption
    StreamingMovies: InternetOption
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ]

    @model_validator(mode="after")
    def validate_service_consistency(self) -> "ChurnRequest":
        if self.PhoneService == "No" and self.MultipleLines != "No phone service":
            raise ValueError("MultipleLines must be 'No phone service' when PhoneService is 'No'")
        if self.PhoneService == "Yes" and self.MultipleLines == "No phone service":
            raise ValueError("MultipleLines cannot be 'No phone service' when PhoneService is 'Yes'")

        internet_fields = (
            "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies",
        )
        values = [getattr(self, field) for field in internet_fields]
        if self.InternetService == "No" and any(value != "No internet service" for value in values):
            raise ValueError("Internet add-ons must be 'No internet service' when InternetService is 'No'")
        if self.InternetService != "No" and any(value == "No internet service" for value in values):
            raise ValueError("Internet add-ons cannot be 'No internet service' when InternetService is active")
        return self


class ChurnResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    churn_probability: float = Field(ge=0, le=1)
    churn_prediction: Literal[0, 1]
    churn_label: Literal["No Churn", "Churn"]
    threshold: float = Field(ge=0, le=1)


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    clients: list[ChurnRequest] = Field(min_length=1, max_length=100)


class BatchResponse(BaseModel):
    count: int = Field(ge=1, le=100)
    predictions: list[ChurnResponse]


class HealthResponse(BaseModel):
    status: Literal["ok", "error"]
    model_loaded: bool
    model_version: str | None


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    threshold: float
    positive_class: int
    feature_count: int
    training_rows: int
    test_metrics: dict[str, object]
