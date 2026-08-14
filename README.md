# Customer Churn Prediction

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.116-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-validated-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![Tests](https://img.shields.io/badge/tests-39%20passed-brightgreen)](#testing-and-validation)
[![Local status](https://img.shields.io/badge/local-COMPLETE%20%2F%20VALIDATED-brightgreen)](#project-status)
[![Deployment](https://img.shields.io/badge/public%20deployment-PENDING-orange)](#deployment-status)

An end-to-end machine learning project that predicts telecom customer churn and turns model probabilities into actionable retention signals. The repository covers the complete lifecycle from business framing and data validation to explainability, a production-style FastAPI service, a responsive web interface, automated tests, and a validated Docker image.

> **Project status:** local development is **COMPLETE / VALIDATED**. Public deployment is **PENDING**; no public URL is currently available.

## Business Problem

Customer churn occurs when a subscriber ends their relationship with a telecom provider. Missed churners can represent lost revenue and missed retention opportunities, while incorrectly flagging loyal customers may lead to unnecessary incentives and contact costs.

This project frames churn prediction as supervised binary classification:

- **Positive class:** `Churn`
- **Negative class:** `No Churn`
- **System output:** churn probability and threshold-based classification
- **Business use:** rank customers by risk and prioritize retention outreach

False negatives are especially relevant because they correspond to actual churners who receive no intervention. Their financial cost is not available in the dataset, so the operating decision is based on a transparent precision–recall trade-off rather than an invented monetary objective.

## Project Objectives

- Build a reproducible, leakage-resistant ML pipeline.
- Produce calibrated-to-purpose churn risk scores for individual customers.
- Improve churner detection through an explicit operating threshold.
- Explain global model behavior and individual predictions.
- Expose the frozen model through validated API and browser interfaces.
- Package and verify the complete inference application with Docker.

## Dataset

The project uses the official IBM Telco Customer Churn sample for a fictional telecommunications company.

| Property | Value |
|---|---:|
| Observations | 7,043 customers |
| Raw columns | 21 |
| Input features | 19 |
| Target | `Churn` (`Yes` / `No`) |
| Churners | 1,869 (26.54%) |
| Non-churners | 5,174 (73.46%) |

The raw dataset contains customer demographics, tenure, subscribed services, contract and billing information, and charges. `customerID` is retained for traceability but excluded from modeling. Eleven blank `TotalCharges` values belong to customers with zero tenure and are converted to `0.0` under a documented cleaning rule.

Dataset provenance, acquisition URL, integrity hash, schema, and cleaning decisions are documented in [dataset provenance](docs/dataset-provenance.md), [data understanding](reports/data-understanding.md), and [data cleaning](docs/data-cleaning.md).

## End-to-End Pipeline

```text
Business understanding
        ↓
Verified IBM data acquisition and integrity checks
        ↓
Data understanding, cleaning, and validation
        ↓
Exploratory data analysis
        ↓
Stratified train/test split (80/20, random_state=42)
        ↓
Deterministic feature engineering
        ↓
Scaling + one-hot encoding inside a fitted pipeline
        ↓
Cross-validation, tuning, and threshold selection on training data
        ↓
One-time final test evaluation and SHAP explainability
        ↓
Serialized pipeline → FastAPI → Web UI → Tests → Docker
```

All learned preprocessing is fitted on the training partition only. The untouched test set contains 1,409 rows and is consumed once for final evaluation after model and threshold selection.

## Exploratory Data Analysis

The EDA reports associations, not causal effects. Key observations include:

- Month-to-month customers have a **42.71%** churn rate, compared with **2.83%** for two-year contracts.
- Fiber optic customers show a **41.89%** churn rate; fiber optic combined with a month-to-month contract reaches **54.61%**.
- Customers without technical support show **41.64%** churn, compared with **15.17%** among customers with support.
- Electronic check users show **45.29%** churn, versus **15.24%** for automatic credit card payments.
- Churners have shorter average tenure (**17.98 months**) than non-churners (**37.57 months**) and higher average monthly charges (**74.44** versus **61.27**).

The complete methodology, segment tables, caveats, and visualizations are available in the [EDA report](reports/eda-report.md) and [EDA notebook](notebooks/01_exploratory_data_analysis.ipynb).

![Churn rates for key categorical variables](reports/figures/key_categorical_churn_rates.png)

## Preprocessing and Feature Engineering

The pipeline preserves identical transformations during training and inference:

- Explicit numerical and categorical feature groups.
- `StandardScaler` for numerical variables.
- `OneHotEncoder(handle_unknown="ignore")` for categorical variables.
- Four deterministic, target-independent engineered features:
  - `tenure_group`
  - `contract_tenure`
  - `internet_contract`
  - `total_services`
- 23 features before preprocessing and **72 transformed features** after scaling and encoding.

The engineered variables are created at runtime inside the scikit-learn pipeline; no second enriched dataset is persisted. See [preprocessing](docs/preprocessing.md) and [feature engineering](docs/feature-engineering.md).

## Models Evaluated

Five model families were compared through reproducible five-fold stratified cross-validation:

- Dummy classifier
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting

Logistic Regression and Gradient Boosting were subsequently evaluated with bounded hyperparameter searches. The tuned Gradient Boosting candidate achieved a modestly higher cross-validated Average Precision, but the pre-defined selection rule favored Logistic Regression when the gap was at most `0.01`. The observed gap was `0.0089`, so the simpler and more interpretable candidate was retained.

Full comparisons and selection rules are documented in [ML evaluation](docs/ml-evaluation.md) and [model improvement](docs/model-improvement.md).

## Final Model

The frozen production candidate is a scikit-learn pipeline containing deterministic feature engineering, preprocessing, and Logistic Regression:

| Parameter | Selected value |
|---|---|
| Estimator | Logistic Regression |
| `C` | `2.0` |
| Solver | `lbfgs` |
| Penalty | L2 |
| Class weights | None |
| Maximum iterations | 2,000 |
| Model version | `1.0.0` |

The serialized artifact is stored at `models/churn_pipeline.joblib`; its metadata and exact evaluation protocol are stored alongside it.

## Final Test Metrics

Results below come from the held-out test set of 1,409 customers.

| Metric | Result |
|---|---:|
| ROC-AUC | **0.8429** |
| Average Precision | **0.6379** |
| Accuracy at 0.30 | **0.7608** |
| Precision at 0.30 | **0.5347** |
| Recall at 0.30 | **0.7620** |
| F1-score at 0.30 | **0.6284** |
| Churners detected | **285 / 374** |
| Confusion matrix | TN 787 · FP 248 · FN 89 · TP 285 |

![Final confusion matrix at threshold 0.30](reports/figures/final_test_confusion_threshold_030.png)

## Operational Threshold: 30%

The final classification rule is:

```text
probability < 0.30  → No Churn
probability ≥ 0.30  → Churn
```

The `0.30` threshold was selected from out-of-fold training predictions before the final test evaluation. Compared with the default `0.50` threshold, it increases test recall from `0.5321` to `0.7620`, detecting **285 instead of 199** of the 374 test churners. This reduces missed retention opportunities from 175 to 89, while accepting more false alerts. A future production threshold should be recalibrated when real retention costs, campaign capacity, and customer lifetime value become available.

## Explainability with SHAP

`shap.LinearExplainer` explains the final Logistic Regression in its transformed 72-feature space. The validated SHAP matrix covers all 1,409 test observations.

The leading global contributors include tenure, monthly charges, contract type, total charges, internet service, and engineered contract/tenure interactions. These values explain model behavior—not causal drivers of churn—and correlated or one-hot encoded variables may share importance.

![Global SHAP importance](reports/figures/final_shap_global_bar.png)

Detailed coefficient and SHAP results are available in [final ML pipeline documentation](docs/final-ml-pipeline.md), [model coefficients](reports/final_model_coefficients.csv), and [SHAP importance](reports/final_shap_importance.csv).

## Repository Architecture

```text
customer-churn-prediction/
├── api/                         # FastAPI application, schemas, and static UI
├── data/
│   ├── raw/                     # Immutable source data
│   └── processed/               # Validated cleaned data
├── docs/                        # Technical and methodological documentation
├── models/                      # Frozen pipeline and model metadata
├── notebooks/                   # Exploratory analysis notebook
├── reports/                     # Metrics, validation reports, tables, and figures
├── src/customer_churn_prediction/
│   ├── acquire_data.py          # Verified data acquisition
│   ├── clean_data.py            # Cleaning and validation
│   ├── eda.py                   # Reproducible EDA outputs
│   ├── preprocessing.py         # Leakage-safe preprocessing
│   ├── feature_engineering.py   # Deterministic features
│   ├── modeling.py              # Baseline model comparison
│   ├── improvement.py           # Tuning and threshold analysis
│   └── final_pipeline.py        # Final evaluation, SHAP, and serialization
├── tests/                       # Automated test suite
├── Dockerfile                   # Production-style container image
├── render.yaml                  # Pending Render deployment configuration
└── pyproject.toml               # Package metadata and dependencies
```

## FastAPI Service

The API loads the frozen model during application startup and validates strict request schemas with Pydantic.

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Service and model readiness |
| `GET` | `/model-info` | Model version, threshold, and metadata |
| `POST` | `/predict` | Single-customer prediction |
| `POST` | `/predict/batch` | Batch prediction for up to 100 customers |
| `GET` | `/docs` | Interactive OpenAPI/Swagger documentation |

Invalid or inconsistent inputs return structured HTTP `422` responses. The complete 19-field contract is documented in [API documentation](docs/api.md).

## Web Interface

FastAPI serves a responsive, dependency-free HTML/CSS/JavaScript interface at `/`. It provides:

- A guided form for all 19 model inputs.
- Automatic consistency handling for phone and internet service fields.
- A real request to the production `/predict` endpoint—no mock inference.
- Churn probability, predicted class, risk presentation, and operating threshold.
- Example loading, reset behavior, service health status, and validation feedback.

The high-risk validation profile returns approximately **78.78% → Churn**; the low-risk profile returns approximately **0.97% → No Churn**. See [interface documentation](docs/user-interface.md).

## Testing and Validation

Final local validation completed successfully:

- **39/39 automated tests passed** with no reported warnings.
- **31 additional functional and technical scenarios passed**.
- **70 total validations**.
- High-risk, low-risk, threshold-boundary, invalid-input, and numerical-boundary cases verified.
- Python pipeline, local FastAPI, and Docker predictions compared across seven valid profiles.
- Maximum probability difference: `5.55 × 10⁻17` (floating-point rounding only).
- Local and Docker model SHA-256 hashes are identical.
- Final verdict: **FINAL VALIDATION: PASS**.

Run the automated suite with:

```bash
python -m pytest -ra
```

The detailed evidence is recorded in [final application validation](reports/final-application-validation.txt).

## Docker

The image `customer-churn-api:1.0.0` was built and executed with Docker Desktop. Its health check passed, the application and model loaded successfully, HTTP routes were accessible, and the test container stopped cleanly with exit code `0`. The image runs as a non-root `app` user.

```bash
docker build -t customer-churn-api:1.0.0 .
docker run --rm --name customer-churn-api -p 8000:8000 customer-churn-api:1.0.0
```

Then open:

- Web interface: `http://localhost:8000/`
- Swagger UI: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

## Installation and Local Execution

### 1. Create an environment

```bash
git clone <repository-url>
cd customer-churn-prediction
python -m venv .venv
```

Activate it:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

### 2. Install the project

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

Python 3.11 or later is required.

### 3. Start the application

```bash
uvicorn api.main:app --reload
```

Open `http://127.0.0.1:8000/` for the web interface or `http://127.0.0.1:8000/docs` for Swagger.

### 4. Request a prediction

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
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
    "PaymentMethod": "Electronic check"
  }'
```

Expected model output (full precision may vary only in presentation):

```json
{
  "churn_probability": 0.7877771751334054,
  "churn_prediction": 1,
  "churn_label": "Churn",
  "threshold": 0.3
}
```

## Documentation

| Topic | Document |
|---|---|
| Business framing | [Business understanding](docs/business-understanding.md) |
| Dataset and schema | [Dataset provenance](docs/dataset-provenance.md) · [Data dictionary](docs/data-dictionary.md) |
| Data quality | [Data understanding](reports/data-understanding.md) · [Data cleaning](docs/data-cleaning.md) |
| Analysis | [EDA report](reports/eda-report.md) · [EDA notebook](notebooks/01_exploratory_data_analysis.ipynb) |
| ML pipeline | [Preprocessing](docs/preprocessing.md) · [Feature engineering](docs/feature-engineering.md) |
| Model development | [ML evaluation](docs/ml-evaluation.md) · [Model improvement](docs/model-improvement.md) |
| Final model and SHAP | [Final ML pipeline](docs/final-ml-pipeline.md) |
| Application | [API](docs/api.md) · [Web interface](docs/user-interface.md) |
| Containerization | [Docker and deployment](docs/docker-deployment.md) |
| Validation evidence | [Final application validation](reports/final-application-validation.txt) |
| Technical decisions | [Decision log](docs/decisions.md) |

## Technologies

Python 3.11+, pandas, NumPy, scikit-learn, Matplotlib, Seaborn, SHAP, FastAPI, Pydantic, Uvicorn, HTTPX, pytest, HTML, CSS, JavaScript, Docker, and Render configuration.

## Deployment Status

| Environment | Status |
|---|---|
| Local development | **COMPLETE / VALIDATED** |
| Automated and functional validation | **PASS** |
| Docker image | **BUILT / HEALTHY / VALIDATED** |
| Public deployment | **PENDING** |

`render.yaml` prepares a Docker web service with `/health` monitoring, but deploying it still requires connecting the repository to a Render account and validating the hosted service. **No public endpoint is currently claimed by this project.**

## Limitations and Future Improvements

- The IBM dataset represents one fictional telecom sample; external validity is unverified.
- No temporal or external validation dataset is currently available.
- The threshold is not yet optimized against real retention costs, campaign capacity, or customer lifetime value.
- SHAP and model coefficients explain predictions and associations, not causal effects.
- Extreme but syntactically valid numerical inputs are accepted; production-grade domain limits could be defined with business owners.
- Production monitoring for drift, data quality, latency, and model performance remains to be designed.
- Probability calibration and threshold review should be considered on representative live data.
- Public deployment, hosted smoke tests, security hardening, observability, and CI/CD remain future work.

## Project Status

The end-to-end system is complete and validated locally, including the model, FastAPI service, browser interface, automated tests, Docker execution, artifact integrity, and cross-environment prediction parity.

**Local development: COMPLETE / VALIDATED**

**Public deployment: PENDING**
