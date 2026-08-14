"""Reproducible calculations and figures for the exploratory data analysis."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

DEFAULT_INPUT = Path("data/processed/telco_customer_churn_clean.csv")
DEFAULT_FIGURES = Path("reports/figures")
EXPECTED_PROCESSED_SHA256 = "7dd93bbff704f59a3044237c6695f4cfb2ee1b6faff6c21265b2e4044195cbd8"
NUMERIC_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_data(path: Path = DEFAULT_INPUT) -> pd.DataFrame:
    """Load and validate the cleaned dataset without modifying it."""
    if file_sha256(path) != EXPECTED_PROCESSED_SHA256:
        raise ValueError("Unexpected processed dataset SHA-256")
    data = pd.read_csv(path)
    if data.shape != (7043, 21) or data.isna().any().any():
        raise ValueError("Unexpected processed dataset shape or missing values")
    if set(data["Churn"]) != {"Yes", "No"}:
        raise ValueError("Unexpected target categories")
    return data


def categorical_summary(data: pd.DataFrame, column: str) -> pd.DataFrame:
    """Return client counts, churner counts, and churn rate by category."""
    if column not in CATEGORICAL_COLUMNS:
        raise ValueError(f"Unsupported categorical column: {column}")
    working = data.assign(_churn=data["Churn"].eq("Yes"))
    return (
        working.groupby(column, dropna=False)
        .agg(clients=("Churn", "size"), churners=("_churn", "sum"), churn_rate=("_churn", "mean"))
        .sort_values("churn_rate", ascending=False)
    )


def numeric_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Return descriptive statistics for numeric variables by target class."""
    return data.groupby("Churn")[NUMERIC_COLUMNS].agg(["count", "mean", "median", "std"])


def multivariate_tables(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the two targeted segment tables used in the EDA."""
    working = data.assign(
        _churn=data["Churn"].eq("Yes"),
        tenure_band=pd.cut(
            data["tenure"], [-1, 12, 24, 48, 72],
            labels=["0-12", "13-24", "25-48", "49-72"],
        ),
    )
    contract_tenure = working.groupby(
        ["Contract", "tenure_band"], observed=True
    ).agg(clients=("Churn", "size"), churners=("_churn", "sum"), churn_rate=("_churn", "mean"))
    internet_contract = working.groupby(
        ["InternetService", "Contract"]
    ).agg(clients=("Churn", "size"), churners=("_churn", "sum"), churn_rate=("_churn", "mean"))
    return contract_tenure, internet_contract


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def generate_figures(data: pd.DataFrame, output_dir: Path = DEFAULT_FIGURES) -> list[Path]:
    """Generate the small set of figures used by the notebook and report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    palette = {"No": "#4C78A8", "Yes": "#E45756"}
    generated: list[Path] = []

    target = data["Churn"].value_counts().rename_axis("Churn").reset_index(name="Clients")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=target, x="Churn", y="Clients", hue="Churn", palette=palette, legend=False, ax=ax)
    ax.set(title="Distribution de la target Churn", xlabel="Churn", ylabel="Nombre de clients")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f")
    path = output_dir / "churn_target_distribution.png"; _save(fig, path); generated.append(path)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, column in zip(axes, NUMERIC_COLUMNS):
        sns.histplot(data=data, x=column, hue="Churn", palette=palette, element="step", stat="density", common_norm=False, ax=ax)
        ax.set_title(f"Distribution de {column} par churn")
    path = output_dir / "numeric_distributions_by_churn.png"; _save(fig, path); generated.append(path)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, column in zip(axes, NUMERIC_COLUMNS):
        sns.boxplot(data=data, x="Churn", y=column, hue="Churn", palette=palette, legend=False, ax=ax)
        ax.set_title(f"{column} selon Churn")
    path = output_dir / "numeric_boxplots_by_churn.png"; _save(fig, path); generated.append(path)

    key_columns = ["Contract", "InternetService", "TechSupport", "OnlineSecurity", "PaymentMethod", "PaperlessBilling"]
    fig, axes = plt.subplots(3, 2, figsize=(15, 15))
    for ax, column in zip(axes.flat, key_columns):
        summary = categorical_summary(data, column).reset_index()
        sns.barplot(data=summary, y=column, x="churn_rate", color="#E45756", ax=ax)
        ax.set(title=f"Churn rate par {column}", xlabel="Churn rate", ylabel="")
        ax.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    path = output_dir / "key_categorical_churn_rates.png"; _save(fig, path); generated.append(path)

    contract_tenure, internet_contract = multivariate_tables(data)
    pivot = contract_tenure["churn_rate"].unstack("tenure_band")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.heatmap(pivot, annot=True, fmt=".1%", cmap="YlOrRd", vmin=0, vmax=0.6, ax=ax)
    ax.set(title="Churn rate : contrat × ancienneté", xlabel="Ancienneté (mois)", ylabel="Contrat")
    path = output_dir / "contract_tenure_churn_heatmap.png"; _save(fig, path); generated.append(path)

    pivot = internet_contract["churn_rate"].unstack("Contract")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.heatmap(pivot, annot=True, fmt=".1%", cmap="YlOrRd", vmin=0, vmax=0.6, ax=ax)
    ax.set(title="Churn rate : service Internet × contrat", xlabel="Contrat", ylabel="Service Internet")
    path = output_dir / "internet_contract_churn_heatmap.png"; _save(fig, path); generated.append(path)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=data, x="Contract", y="MonthlyCharges", hue="Churn", palette=palette, ax=ax)
    ax.set(title="Charges mensuelles selon contrat et churn", xlabel="Contrat", ylabel="Charges mensuelles")
    path = output_dir / "monthly_charges_contract_churn.png"; _save(fig, path); generated.append(path)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(data[NUMERIC_COLUMNS].corr(), annot=True, fmt=".3f", cmap="vlag", vmin=-1, vmax=1, ax=ax)
    ax.set_title("Corrélations de Pearson — variables numériques")
    path = output_dir / "numeric_correlations.png"; _save(fig, path); generated.append(path)
    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--figures", type=Path, default=DEFAULT_FIGURES)
    args = parser.parse_args()
    figures = generate_figures(load_data(args.input), args.figures)
    print(f"Generated {len(figures)} figures in {args.figures}")


if __name__ == "__main__":
    main()
