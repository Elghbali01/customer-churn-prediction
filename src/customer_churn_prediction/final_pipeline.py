"""Frozen final churn pipeline, prediction contract, evaluation, and explainability."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

from customer_churn_prediction.feature_engineering import ChurnFeatureEngineer
from customer_churn_prediction.preprocessing import (
    DEFAULT_DATASET,
    build_preprocessor,
    prepare_features,
    split_data,
)

MODEL_VERSION = "1.0.0"
OPERATIONAL_THRESHOLD = 0.30
DEFAULT_MODEL_PATH = Path("models/churn_pipeline.joblib")
DEFAULT_METADATA_PATH = Path("models/model_metadata.json")
DEFAULT_REPORT_PATH = Path("reports/final_ml_evaluation.json")
DEFAULT_COEFFICIENTS_PATH = Path("reports/final_model_coefficients.csv")
DEFAULT_SHAP_PATH = Path("reports/final_shap_importance.csv")
DEFAULT_FIGURES_DIR = Path("reports/figures")


def build_final_pipeline() -> Pipeline:
    """Return the exact unfitted candidate selected at Stage 10."""
    return Pipeline(
        steps=[
            ("feature_engineering", ChurnFeatureEngineer()),
            ("preprocessor", build_preprocessor()),
            (
                "model",
                LogisticRegression(
                    C=2.0,
                    solver="lbfgs",
                    class_weight=None,
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )


def predict_churn(
    raw_features: pd.DataFrame,
    *,
    pipeline: Pipeline | None = None,
    model_path: Path = DEFAULT_MODEL_PATH,
    threshold: float = OPERATIONAL_THRESHOLD,
) -> pd.DataFrame:
    """Return probabilities and explicit threshold decisions for raw features."""
    fitted_pipeline = pipeline if pipeline is not None else joblib.load(model_path)
    probabilities = fitted_pipeline.predict_proba(raw_features)[:, 1]
    return pd.DataFrame(
        {
            "churn_probability": probabilities,
            "churn_prediction": (probabilities >= threshold).astype("int8"),
            "threshold": np.full(len(probabilities), threshold),
        },
        index=raw_features.index,
    )


def _threshold_evaluation(y_true: pd.Series, probabilities: np.ndarray, threshold: float) -> dict[str, Any]:
    predictions = (probabilities >= threshold).astype("int8")
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def transformed_features(pipeline: Pipeline, raw_features: pd.DataFrame) -> tuple[Any, np.ndarray]:
    engineered = pipeline.named_steps["feature_engineering"].transform(raw_features)
    preprocessor = pipeline.named_steps["preprocessor"]
    transformed = preprocessor.transform(engineered)
    names = preprocessor.get_feature_names_out()
    if transformed.shape[1] != len(names):
        raise RuntimeError("Transformed feature-name mapping is inconsistent")
    return transformed, names


def coefficient_table(pipeline: Pipeline, feature_names: np.ndarray) -> pd.DataFrame:
    coefficients = pipeline.named_steps["model"].coef_.ravel()
    if len(coefficients) != len(feature_names):
        raise RuntimeError("Coefficient count differs from transformed feature count")
    return pd.DataFrame(
        {"feature": feature_names, "coefficient": coefficients,
         "abs_coefficient": np.abs(coefficients), "odds_ratio": np.exp(coefficients)}
    ).sort_values("abs_coefficient", ascending=False, ignore_index=True)


def _plot_test_evaluation(y: pd.Series, probabilities: np.ndarray, figures: Path) -> list[Path]:
    figures.mkdir(parents=True, exist_ok=True)
    predictions = probabilities >= OPERATIONAL_THRESHOLD
    matrix = confusion_matrix(y, predictions, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
    ax.set(title="Test — matrice de confusion (seuil 0,30)", xlabel="Prédit", ylabel="Réel")
    fig.tight_layout(); confusion_path = figures / "final_test_confusion_threshold_030.png"
    fig.savefig(confusion_path, dpi=160, bbox_inches="tight"); plt.close(fig)

    fpr, tpr, _ = roc_curve(y, probabilities)
    fig, ax = plt.subplots(figsize=(6.5, 5.5)); ax.plot(fpr, tpr, label=f"AUC={roc_auc_score(y, probabilities):.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey"); ax.set(title="Courbe ROC — test", xlabel="False Positive Rate", ylabel="True Positive Rate")
    ax.legend(); fig.tight_layout(); roc_path = figures / "final_test_roc_curve.png"
    fig.savefig(roc_path, dpi=160, bbox_inches="tight"); plt.close(fig)

    precision, recall, _ = precision_recall_curve(y, probabilities)
    fig, ax = plt.subplots(figsize=(6.5, 5.5)); ax.plot(recall, precision, label=f"AP={average_precision_score(y, probabilities):.3f}")
    ax.axhline(y.mean(), linestyle="--", color="grey", label=f"Prévalence={y.mean():.3f}")
    ax.set(title="Courbe Precision–Recall — test", xlabel="Recall", ylabel="Precision")
    ax.legend(); fig.tight_layout(); pr_path = figures / "final_test_pr_curve.png"
    fig.savefig(pr_path, dpi=160, bbox_inches="tight"); plt.close(fig)
    return [confusion_path, roc_path, pr_path]


def _plot_coefficients(table: pd.DataFrame, figures: Path, top_n: int = 20) -> Path:
    selected = table.nlargest(top_n, "abs_coefficient").sort_values("coefficient")
    fig, ax = plt.subplots(figsize=(10, 7)); colors = np.where(selected["coefficient"] >= 0, "#C44E52", "#4C72B0")
    ax.barh(selected["feature"], selected["coefficient"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8); ax.set(title=f"Logistic Regression — top {top_n} coefficients", xlabel="Coefficient")
    fig.tight_layout(); path = figures / "final_logistic_coefficients.png"
    fig.savefig(path, dpi=160, bbox_inches="tight"); plt.close(fig); return path


def _shap_analysis(
    pipeline: Pipeline, X_train: pd.DataFrame, X_test: pd.DataFrame,
    probabilities: np.ndarray, feature_names: np.ndarray, figures: Path,
) -> tuple[pd.DataFrame, dict[str, Any], list[Path], np.ndarray]:
    train_transformed, _ = transformed_features(pipeline, X_train)
    test_transformed, _ = transformed_features(pipeline, X_test)
    background = train_transformed
    explainer = shap.LinearExplainer(pipeline.named_steps["model"], background)
    explanation = explainer(test_transformed)
    values = np.asarray(explanation.values)
    if values.shape != test_transformed.shape or values.shape[1] != len(feature_names):
        raise RuntimeError("SHAP mapping does not match transformed features")
    if not np.isfinite(values).all():
        raise RuntimeError("SHAP contains non-finite values")

    importance = pd.DataFrame({"feature": feature_names, "mean_abs_shap": np.abs(values).mean(axis=0)})
    importance = importance.sort_values("mean_abs_shap", ascending=False, ignore_index=True)

    top = importance.head(20).sort_values("mean_abs_shap")
    fig, ax = plt.subplots(figsize=(10, 7)); ax.barh(top["feature"], top["mean_abs_shap"], color="#4C72B0")
    ax.set(title="SHAP global — importance moyenne absolue", xlabel="mean(|SHAP value|)")
    fig.tight_layout(); bar_path = figures / "final_shap_global_bar.png"
    fig.savefig(bar_path, dpi=160, bbox_inches="tight"); plt.close(fig)

    sample_size = min(600, len(X_test)); sample_indices = np.linspace(0, len(X_test) - 1, sample_size, dtype=int)
    dense_test = test_transformed.toarray() if sparse.issparse(test_transformed) else np.asarray(test_transformed)
    shap.summary_plot(values[sample_indices], dense_test[sample_indices], feature_names=feature_names, show=False, max_display=20)
    plt.title("SHAP summary — test"); plt.tight_layout(); summary_path = figures / "final_shap_summary.png"
    plt.savefig(summary_path, dpi=160, bbox_inches="tight"); plt.close()

    client_position = int(np.argmax(probabilities))
    client_values = values[client_position]
    local = pd.DataFrame({"feature": feature_names, "shap_value": client_values})
    positive = local.nlargest(5, "shap_value").to_dict(orient="records")
    negative = local.nsmallest(5, "shap_value").to_dict(orient="records")
    local_top = local.reindex(local["shap_value"].abs().nlargest(12).index).sort_values("shap_value")
    fig, ax = plt.subplots(figsize=(10, 6)); local_colors = np.where(local_top["shap_value"] >= 0, "#C44E52", "#4C72B0")
    ax.barh(local_top["feature"], local_top["shap_value"], color=local_colors); ax.axvline(0, color="black", linewidth=0.8)
    ax.set(title="Explication SHAP individuelle — client à risque", xlabel="Contribution au log-odds")
    fig.tight_layout(); local_path = figures / "final_shap_high_risk_client.png"
    fig.savefig(local_path, dpi=160, bbox_inches="tight"); plt.close(fig)
    return importance, {"test_position": client_position, "probability": float(probabilities[client_position]),
                        "positive_factors": positive, "negative_factors": negative}, [bar_path, summary_path, local_path], values


def run_final_ml() -> dict[str, Any]:
    """Fit once on train, evaluate once on test, explain, and serialize."""
    data = pd.read_csv(DEFAULT_DATASET)
    X, y = prepare_features(data)
    split = split_data(X, y)
    pipeline = build_final_pipeline()
    pipeline.fit(split.X_train, split.y_train)

    probabilities = pipeline.predict_proba(split.X_test)[:, 1]
    evaluations = {"0.50": _threshold_evaluation(split.y_test, probabilities, 0.50),
                   "0.30": _threshold_evaluation(split.y_test, probabilities, OPERATIONAL_THRESHOLD)}
    roc_auc = float(roc_auc_score(split.y_test, probabilities))
    average_precision = float(average_precision_score(split.y_test, probabilities))
    _, feature_names = transformed_features(pipeline, split.X_test)
    coefficients = coefficient_table(pipeline, feature_names)

    figure_paths = _plot_test_evaluation(split.y_test, probabilities, DEFAULT_FIGURES_DIR)
    figure_paths.append(_plot_coefficients(coefficients, DEFAULT_FIGURES_DIR))
    shap_importance, client, shap_paths, shap_values = _shap_analysis(
        pipeline, split.X_train, split.X_test, probabilities, feature_names, DEFAULT_FIGURES_DIR
    )
    figure_paths.extend(shap_paths)
    client["prediction"] = int(probabilities[client["test_position"]] >= OPERATIONAL_THRESHOLD)
    client["true_class"] = int(split.y_test.iloc[client["test_position"]])

    DEFAULT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, DEFAULT_MODEL_PATH)
    reloaded = joblib.load(DEFAULT_MODEL_PATH)
    before = probabilities[:20]
    after = reloaded.predict_proba(split.X_test.iloc[:20])[:, 1]
    max_difference = float(np.max(np.abs(before - after)))
    if not np.array_equal(before >= OPERATIONAL_THRESHOLD, after >= OPERATIONAL_THRESHOLD):
        raise RuntimeError("Round-trip prediction classes differ")

    metadata = {
        "model_name": "Logistic Regression", "model_version": MODEL_VERSION,
        "threshold": OPERATIONAL_THRESHOLD, "positive_class": 1,
        "feature_count": int(len(feature_names)), "training_rows": int(len(split.X_train)),
        "selected_hyperparameters": {"C": 2.0, "solver": "lbfgs", "penalty": "l2", "class_weight": None, "max_iter": 2000},
        "metrics_test": {"roc_auc": roc_auc, "average_precision": average_precision, **evaluations},
        "training_scope": "train_only", "test_rows": int(len(split.X_test)),
    }
    DEFAULT_METADATA_PATH.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    coefficients.to_csv(DEFAULT_COEFFICIENTS_PATH, index=False, float_format="%.10f")
    shap_importance.to_csv(DEFAULT_SHAP_PATH, index=False, float_format="%.10f")

    prior = json.loads(Path("reports/model_improvement.json").read_text(encoding="utf-8"))
    oof = prior["selection"]["at_candidate"]
    cv = prior["tuning"]["logistic"]["best"]
    payload = {
        "protocol": {"train_rows": len(split.X_train), "test_rows": len(split.X_test), "fit_scope": "train_only",
                     "test_set_consumed_for_final_evaluation": True, "selection_after_test": False},
        "test": {"thresholds": evaluations, "roc_auc": roc_auc, "average_precision": average_precision},
        "generalization": {
            "roc_auc_cv_mean": cv["roc_auc_mean"], "roc_auc_test": roc_auc, "roc_auc_gap_test_minus_cv": roc_auc - cv["roc_auc_mean"],
            "average_precision_cv_mean": cv["average_precision_mean"], "average_precision_test": average_precision,
            "average_precision_gap_test_minus_cv": average_precision - cv["average_precision_mean"],
            "threshold_0.30_oof": {key: oof[key] for key in ["precision", "recall", "f1"]},
            "threshold_0.30_test": {key: evaluations["0.30"][key] for key in ["precision", "recall", "f1"]},
        },
        "feature_count": len(feature_names),
        "top_coefficients": {
            "positive": coefficients.nlargest(10, "coefficient").to_dict(orient="records"),
            "negative": coefficients.nsmallest(10, "coefficient").to_dict(orient="records"),
            "absolute": coefficients.head(10).to_dict(orient="records"),
        },
        "shap": {"version": shap.__version__, "shape": list(shap_values.shape),
                 "top_global": shap_importance.head(15).to_dict(orient="records"), "high_risk_client": client},
        "serialization": {"pipeline": DEFAULT_MODEL_PATH.as_posix(), "metadata": DEFAULT_METADATA_PATH.as_posix(),
                          "round_trip_rows": 20, "max_probability_difference": max_difference},
        "figures": [path.as_posix() for path in figure_paths],
    }
    DEFAULT_REPORT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Final pipeline fitted on train only; test consumed once for final evaluation.")
    print(f"ROC-AUC={roc_auc:.6f}; AP={average_precision:.6f}; round-trip max diff={max_difference:.3g}")
    return payload


if __name__ == "__main__":
    run_final_ml()
