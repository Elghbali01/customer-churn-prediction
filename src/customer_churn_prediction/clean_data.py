"""Build the cleaned Telco Customer Churn dataset from the immutable raw CSV."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd

DEFAULT_INPUT = Path("data/raw/Telco-Customer-Churn.csv")
DEFAULT_OUTPUT = Path("data/processed/telco_customer_churn_clean.csv")
EXPECTED_RAW_SHA256 = "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"
EXPECTED_COLUMNS = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges", "Churn",
]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_raw(data: pd.DataFrame) -> None:
    """Validate assumptions used by the cleaning rules."""
    if data.columns.tolist() != EXPECTED_COLUMNS:
        raise ValueError("Unexpected raw schema")
    if data.shape != (7043, 21):
        raise ValueError(f"Unexpected raw dimensions: {data.shape}")
    if data.duplicated().any() or data["customerID"].duplicated().any():
        raise ValueError("Raw data contains unexpected duplicates")
    if data["customerID"].isna().any() or data["customerID"].str.strip().eq("").any():
        raise ValueError("Missing or empty customer identifier")
    if not data["customerID"].str.fullmatch(r"[0-9]{4}-[A-Z]{5}").all():
        raise ValueError("Unexpected customer identifier format")
    if set(data["SeniorCitizen"].unique()) != {0, 1}:
        raise ValueError("SeniorCitizen is not encoded as expected")
    if set(data["Churn"].unique()) != {"Yes", "No"}:
        raise ValueError("Unexpected Churn categories")


def clean(data: pd.DataFrame) -> pd.DataFrame:
    """Apply validated cleaning rules without mutating the input dataframe."""
    cleaned = data.copy(deep=True)

    blank_total_charges = cleaned["TotalCharges"].str.strip().eq("")
    if int(blank_total_charges.sum()) != 11:
        raise ValueError("Unexpected number of blank TotalCharges values")
    if not cleaned.loc[blank_total_charges, "tenure"].eq(0).all():
        raise ValueError("Blank TotalCharges found for a customer with non-zero tenure")

    normalized_total_charges = cleaned["TotalCharges"].replace(r"^\s*$", pd.NA, regex=True)
    cleaned["TotalCharges"] = pd.to_numeric(normalized_total_charges, errors="raise")
    cleaned.loc[blank_total_charges, "TotalCharges"] = 0.0

    validate_cleaned(cleaned)
    return cleaned


def validate_cleaned(data: pd.DataFrame) -> None:
    """Validate the cleaned dataset while preserving non-cleaning semantics."""
    if data.shape != (7043, 21):
        raise ValueError(f"Unexpected cleaned dimensions: {data.shape}")
    if data.duplicated().any() or data["customerID"].duplicated().any():
        raise ValueError("Cleaned data contains duplicates")
    if data.isna().any().any():
        raise ValueError("Cleaned data contains missing values")
    if not pd.api.types.is_float_dtype(data["TotalCharges"]):
        raise TypeError("TotalCharges must be a floating-point column")
    if (data[["tenure", "MonthlyCharges", "TotalCharges"]] < 0).any().any():
        raise ValueError("Negative tenure or charge detected")
    if set(data["SeniorCitizen"].unique()) != {0, 1}:
        raise ValueError("SeniorCitizen categories changed")
    if set(data["Churn"].unique()) != {"Yes", "No"}:
        raise ValueError("Churn categories changed")
    if not (data.loc[data["PhoneService"].eq("No"), "MultipleLines"] == "No phone service").all():
        raise ValueError("Inconsistent phone-service categories")
    no_internet = data["InternetService"].eq("No")
    internet_features = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies",
    ]
    if not data.loc[no_internet, internet_features].eq("No internet service").all().all():
        raise ValueError("Inconsistent internet-service categories")


def build_dataset(source: Path = DEFAULT_INPUT, output: Path = DEFAULT_OUTPUT) -> Path:
    """Read raw data, clean it deterministically, and write a processed CSV."""
    if not source.is_file():
        raise FileNotFoundError(f"Raw dataset not found: {source}")
    raw_hash = file_sha256(source)
    if raw_hash != EXPECTED_RAW_SHA256:
        raise ValueError(f"Unexpected raw SHA-256: {raw_hash}")

    raw = pd.read_csv(source)
    validate_raw(raw)
    original_target_distribution = raw["Churn"].value_counts().to_dict()
    cleaned = clean(raw)
    if cleaned["Churn"].value_counts().to_dict() != original_target_distribution:
        raise ValueError("Target distribution changed during cleaning")

    output.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output, index=False, lineterminator="\n")

    reloaded = pd.read_csv(output)
    validate_cleaned(reloaded)
    if file_sha256(source) != raw_hash:
        raise RuntimeError("Raw dataset changed during cleaning")
    print(f"Cleaned dataset written to {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    build_dataset(args.input, args.output)


if __name__ == "__main__":
    main()
