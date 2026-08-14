"""Train-only model improvement: weights, tuning, OOF thresholds, and selection."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_predict, cross_validate

from customer_churn_prediction.evaluation import make_cv
from customer_churn_prediction.modeling import build_model_pipeline
from customer_churn_prediction.preprocessing import DEFAULT_DATASET, prepare_features, split_data

warnings.filterwarnings(
    "ignore",
    message="'penalty' was deprecated.*",
    category=FutureWarning,
    module=r"sklearn\.linear_model.*",
)

RANDOM_STATE = 42
PRIMARY_SCORING = "average_precision"
IMPROVEMENT_SCORING = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
    "roc_auc": "roc_auc",
    "average_precision": "average_precision",
}
LOGISTIC_PARAM_GRID = [
    {
        "model__solver": ["lbfgs"],
        "model__penalty": ["l2"],
        "model__C": [0.1, 0.5, 1.0, 2.0, 10.0],
        "model__class_weight": [None, "balanced", {0: 1.0, 1: 1.5}],
    },
    {
        "model__solver": ["liblinear"],
        "model__penalty": ["l1", "l2"],
        "model__C": [0.1, 1.0, 10.0],
        "model__class_weight": [None, "balanced"],
    },
]
GRADIENT_PARAM_DISTRIBUTIONS = {
    "model__n_estimators": [75, 100, 150],
    "model__learning_rate": [0.03, 0.05, 0.1],
    "model__max_depth": [1, 2, 3],
    "model__min_samples_split": [2, 10],
    "model__min_samples_leaf": [1, 10],
    "model__subsample": [0.8, 1.0],
}
GRADIENT_N_ITER = 16
CLASS_WEIGHT_CONFIGS = {
    "None": None,
    "balanced": "balanced",
    "positive_1.5": {0: 1.0, 1: 1.5},
    "positive_2.0": {0: 1.0, 1: 2.0},
}
THRESHOLDS = np.round(np.arange(0.20, 0.701, 0.02), 2)
SELECTED_MODEL_NAME = "Logistic Regression"
SELECTED_MODEL_PARAMS = {
    "model__C": 2.0,
    "model__class_weight": None,
    "model__solver": "lbfgs",
}
SELECTED_FEATURE_ENGINEERING = True
SELECTED_THRESHOLD = 0.30

DEFAULT_JSON = Path("reports/model_improvement.json")
DEFAULT_CV_CSV = Path("reports/model_improvement_cv.csv")
DEFAULT_THRESHOLD_CSV = Path("reports/model_improvement_thresholds.csv")
DEFAULT_FIGURES = Path("reports/figures")


def _metric_summary(scores: dict[str, np.ndarray]) -> dict[str, float]:
    summary: dict[str, float] = {}
    for metric in IMPROVEMENT_SCORING:
        values = np.asarray(scores[f"test_{metric}"], dtype=float)
        summary[f"{metric}_mean"] = float(values.mean())
        summary[f"{metric}_std"] = float(values.std(ddof=0))
    return summary


def evaluate_pipeline(pipeline: object, X_train: pd.DataFrame, y_train: pd.Series) -> dict[str, float]:
    scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=make_cv(),
        scoring=IMPROVEMENT_SCORING,
        n_jobs=-1,
        error_score="raise",
    )
    return _metric_summary(scores)


def evaluate_class_weights(X_train: pd.DataFrame, y_train: pd.Series) -> dict[str, dict[str, float]]:
    results = {}
    for label, class_weight in CLASS_WEIGHT_CONFIGS.items():
        pipeline = build_model_pipeline("Logistic Regression")
        pipeline.set_params(model__class_weight=class_weight)
        results[label] = evaluate_pipeline(pipeline, X_train, y_train)
    return results


def build_logistic_search() -> GridSearchCV:
    """Exhaustive small grid containing only compatible solver/penalty pairs."""
    return GridSearchCV(
        estimator=build_model_pipeline("Logistic Regression"),
        param_grid=LOGISTIC_PARAM_GRID,
        scoring=IMPROVEMENT_SCORING,
        refit=PRIMARY_SCORING,
        cv=make_cv(),
        n_jobs=-1,
        error_score="raise",
        return_train_score=False,
    )


def build_gradient_search() -> RandomizedSearchCV:
    """Sample a bounded GB space without an excessive Cartesian grid."""
    return RandomizedSearchCV(
        estimator=build_model_pipeline("Gradient Boosting"),
        param_distributions=GRADIENT_PARAM_DISTRIBUTIONS,
        n_iter=GRADIENT_N_ITER,
        scoring=IMPROVEMENT_SCORING,
        refit=PRIMARY_SCORING,
        cv=make_cv(),
        random_state=RANDOM_STATE,
        n_jobs=-1,
        error_score="raise",
        return_train_score=False,
    )


def build_selected_candidate() -> object:
    """Build the train-selected unfitted candidate; do not serialize it yet."""
    pipeline = build_model_pipeline(
        SELECTED_MODEL_NAME,
        feature_engineering=SELECTED_FEATURE_ENGINEERING,
    )
    return pipeline.set_params(**SELECTED_MODEL_PARAMS)


def _best_search_metrics(search: object) -> dict[str, float]:
    index = int(search.best_index_)
    result = {}
    for metric in IMPROVEMENT_SCORING:
        result[f"{metric}_mean"] = float(search.cv_results_[f"mean_test_{metric}"][index])
        result[f"{metric}_std"] = float(search.cv_results_[f"std_test_{metric}"][index])
    return result


def oof_probabilities(pipeline: object, X_train: pd.DataFrame, y_train: pd.Series) -> np.ndarray:
    probabilities = cross_val_predict(
        pipeline,
        X_train,
        y_train,
        cv=make_cv(),
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]
    if probabilities.shape != (len(X_train),) or not np.isfinite(probabilities).all():
        raise RuntimeError("Invalid OOF probabilities")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise RuntimeError("OOF probabilities outside [0, 1]")
    return probabilities


def threshold_metrics(y_true: pd.Series, probabilities: np.ndarray) -> pd.DataFrame:
    rows = []
    for threshold in THRESHOLDS:
        predicted = (probabilities >= threshold).astype("int8")
        tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
        rows.append(
            {
                "threshold": float(threshold),
                "precision": precision_score(y_true, predicted, zero_division=0),
                "recall": recall_score(y_true, predicted, zero_division=0),
                "f1": f1_score(y_true, predicted, zero_division=0),
                "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            }
        )
    return pd.DataFrame(rows)


def select_threshold(table: pd.DataFrame) -> float:
    """Select max F1; ties prefer recall, then precision, then higher threshold."""
    ordered = table.sort_values(
        ["f1", "recall", "precision", "threshold"],
        ascending=[False, False, False, False],
        kind="mergesort",
    )
    return float(ordered.iloc[0]["threshold"])


def _oof_summary(y: pd.Series, probabilities: np.ndarray, threshold: float) -> dict[str, float | int]:
    predicted = (probabilities >= threshold).astype("int8")
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "average_precision": float(average_precision_score(y, probabilities)),
    }


def _plot_cv_comparison(rows: list[dict[str, object]], figures: Path) -> Path:
    frame = pd.DataFrame(rows).set_index("configuration")
    metrics = ["accuracy_mean", "precision_mean", "recall_mean", "f1_mean", "roc_auc_mean", "average_precision_mean"]
    fig, ax = plt.subplots(figsize=(13, 6))
    frame[metrics].plot(kind="bar", ax=ax, color=sns.color_palette("deep", len(metrics)))
    ax.set(title="Avant / après tuning — CV train", ylabel="Score moyen CV", xlabel="", ylim=(0, 1))
    ax.legend([name.replace("_mean", "").replace("_", " ").upper() for name in metrics], ncol=3, loc="lower right")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    path = figures / "model_improvement_cv_comparison.png"
    fig.savefig(path, dpi=160, bbox_inches="tight"); plt.close(fig)
    return path


def _plot_pr_curves(y: pd.Series, probabilities: dict[str, np.ndarray], figures: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 6))
    for label, values in probabilities.items():
        precision, recall, _ = precision_recall_curve(y, values)
        ap = average_precision_score(y, values)
        ax.plot(recall, precision, label=f"{label} (AP={ap:.3f})")
    ax.axhline(y.mean(), linestyle="--", color="grey", label=f"Prévalence={y.mean():.3f}")
    ax.set(title="Precision–Recall out-of-fold — train", xlabel="Recall", ylabel="Precision")
    ax.legend(loc="lower left"); fig.tight_layout()
    path = figures / "model_improvement_pr_curves.png"
    fig.savefig(path, dpi=160, bbox_inches="tight"); plt.close(fig)
    return path


def _plot_thresholds(table: pd.DataFrame, candidate: float, figures: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5))
    for metric in ["precision", "recall", "f1"]:
        ax.plot(table["threshold"], table[metric], marker="o", markersize=3, label=metric.capitalize())
    ax.axvline(0.5, linestyle="--", color="grey", label="Seuil 0.50")
    ax.axvline(candidate, linestyle="--", color="black", label=f"Candidat {candidate:.2f}")
    ax.set(title="Compromis Precision / Recall / F1 — OOF train", xlabel="Threshold", ylabel="Score", ylim=(0, 1))
    ax.legend(); fig.tight_layout()
    path = figures / "model_improvement_threshold_tradeoff.png"
    fig.savefig(path, dpi=160, bbox_inches="tight"); plt.close(fig)
    return path


def _plot_candidate_confusions(y: pd.Series, probabilities: np.ndarray, candidate: float, figures: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, threshold in zip(axes, [0.5, candidate]):
        matrix = confusion_matrix(y, probabilities >= threshold, labels=[0, 1])
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
        ax.set(title=f"OOF — threshold {threshold:.2f}", xlabel="Prédit", ylabel="Réel")
    fig.tight_layout()
    path = figures / "model_improvement_candidate_confusions.png"
    fig.savefig(path, dpi=160, bbox_inches="tight"); plt.close(fig)
    return path


def run_improvement(
    dataset: Path = DEFAULT_DATASET,
    output_json: Path = DEFAULT_JSON,
    output_cv: Path = DEFAULT_CV_CSV,
    output_thresholds: Path = DEFAULT_THRESHOLD_CSV,
    figures: Path = DEFAULT_FIGURES,
) -> dict[str, object]:
    """Run all improvement decisions on train only; never evaluate the test set."""
    data = pd.read_csv(dataset)
    X, y = prepare_features(data)
    split = split_data(X, y)
    X_train, y_train = split.X_train, split.y_train

    logistic_baseline = evaluate_pipeline(build_model_pipeline("Logistic Regression"), X_train, y_train)
    gradient_baseline = evaluate_pipeline(build_model_pipeline("Gradient Boosting"), X_train, y_train)
    weight_results = evaluate_class_weights(X_train, y_train)

    logistic_search = build_logistic_search().fit(X_train, y_train)
    gradient_search = build_gradient_search().fit(X_train, y_train)
    logistic_tuned = _best_search_metrics(logistic_search)
    gradient_tuned = _best_search_metrics(gradient_search)

    best_logistic = build_model_pipeline("Logistic Regression").set_params(**logistic_search.best_params_)
    best_gradient = build_model_pipeline("Gradient Boosting").set_params(**gradient_search.best_params_)

    logistic_without_fe = build_model_pipeline("Logistic Regression", feature_engineering=False)
    no_fe_params = {key: value for key, value in logistic_search.best_params_.items() if key.startswith("model__")}
    logistic_without_fe.set_params(**no_fe_params)
    fe_comparison = {
        "without": evaluate_pipeline(logistic_without_fe, X_train, y_train),
        "with": logistic_tuned,
    }
    fe_comparison["delta"] = {
        key: fe_comparison["with"][key] - fe_comparison["without"][key]
        for key in fe_comparison["with"] if key.endswith("_mean")
    }

    oof = {
        "Logistic Regression tuned": oof_probabilities(best_logistic, X_train, y_train),
        "Gradient Boosting tuned": oof_probabilities(best_gradient, X_train, y_train),
    }

    # If AP differs by <= 0.01, prefer Logistic Regression for simplicity.
    ap_gap = gradient_tuned["average_precision_mean"] - logistic_tuned["average_precision_mean"]
    final_label = "Logistic Regression tuned" if ap_gap <= 0.01 else "Gradient Boosting tuned"
    final_pipeline = best_logistic if final_label.startswith("Logistic") else best_gradient
    final_probabilities = oof[final_label]
    table = threshold_metrics(y_train, final_probabilities)
    candidate_threshold = select_threshold(table)

    threshold_default = _oof_summary(y_train, final_probabilities, 0.5)
    threshold_candidate = _oof_summary(y_train, final_probabilities, candidate_threshold)
    candidate_summaries = {
        label: _oof_summary(y_train, values, 0.5) for label, values in oof.items()
    }

    cv_rows = [
        {"configuration": "Logistic baseline", **logistic_baseline},
        {"configuration": "Logistic tuned", **logistic_tuned},
        {"configuration": "Gradient baseline", **gradient_baseline},
        {"configuration": "Gradient tuned", **gradient_tuned},
    ]
    figures.mkdir(parents=True, exist_ok=True)
    figure_paths = [
        _plot_cv_comparison(cv_rows, figures),
        _plot_pr_curves(y_train, oof, figures),
        _plot_thresholds(table, candidate_threshold, figures),
        _plot_candidate_confusions(y_train, final_probabilities, candidate_threshold, figures),
    ]

    output_json.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cv_rows).to_csv(output_cv, index=False, float_format="%.10f")
    table.to_csv(output_thresholds, index=False, float_format="%.10f")
    payload: dict[str, object] = {
        "protocol": {
            "train_rows": len(X_train), "reserved_test_rows": len(split.X_test),
            "test_set_consumed": False, "primary_scoring": PRIMARY_SCORING,
            "cv": {"type": "StratifiedKFold", "n_splits": 5, "shuffle": True, "random_state": 42},
        },
        "class_weight_experiments": weight_results,
        "baseline": {"logistic": logistic_baseline, "gradient_boosting": gradient_baseline},
        "tuning": {
            "logistic": {"method": "GridSearchCV", "best_params": logistic_search.best_params_, "best": logistic_tuned},
            "gradient_boosting": {"method": "RandomizedSearchCV", "n_iter": GRADIENT_N_ITER,
                                  "best_params": gradient_search.best_params_, "best": gradient_tuned},
        },
        "feature_engineering_comparison": fe_comparison,
        "oof_at_0_5": candidate_summaries,
        "selection": {
            "rule": "Prefer Logistic Regression when tuned AP gap versus Gradient Boosting is <= 0.01",
            "ap_gap_gradient_minus_logistic": ap_gap,
            "final_candidate": final_label,
            "feature_engineering": True,
            "candidate_threshold_rule": "Maximum OOF F1 on fixed 0.20–0.70 grid; deterministic tie-break",
            "candidate_threshold": candidate_threshold,
            "at_0_5": threshold_default,
            "at_candidate": threshold_candidate,
        },
        "figures": [path.as_posix() for path in figure_paths],
    }
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Final train-selected candidate: {final_label}; threshold={candidate_threshold:.2f}")
    print("Test set not consumed.")
    _ = final_pipeline  # Explicitly not fitted or serialized outside CV.
    return payload


if __name__ == "__main__":
    run_improvement()
