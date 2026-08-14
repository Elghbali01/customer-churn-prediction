"""Train-only cross-validation, OOF evaluation, and reporting utilities."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate

from customer_churn_prediction.modeling import MODEL_NAMES, build_model_pipeline
from customer_churn_prediction.preprocessing import DEFAULT_DATASET, prepare_features, split_data

CV_FOLDS = 5
RANDOM_STATE = 42
SCORING = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
    "roc_auc": "roc_auc",
}
DEFAULT_RESULTS_CSV = Path("reports/ml_cv_results.csv")
DEFAULT_RESULTS_JSON = Path("reports/ml_evaluation.json")
DEFAULT_FIGURES_DIR = Path("reports/figures")


def make_cv() -> StratifiedKFold:
    return StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)


def evaluate_pipeline(
    pipeline: object,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: StratifiedKFold,
) -> dict[str, float]:
    """Return mean and std for all required train-only CV metrics."""
    scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=SCORING,
        n_jobs=-1,
        error_score="raise",
    )
    result: dict[str, float] = {}
    for metric in SCORING:
        values = np.asarray(scores[f"test_{metric}"], dtype=float)
        result[f"{metric}_mean"] = float(values.mean())
        result[f"{metric}_std"] = float(values.std(ddof=0))
    return result


def compare_models(X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    """Compare all models using identical folds and full FE pipelines."""
    cv = make_cv()
    rows = []
    for model_name in MODEL_NAMES:
        result = evaluate_pipeline(build_model_pipeline(model_name), X_train, y_train, cv)
        rows.append({"model": model_name, **result})
    return pd.DataFrame(rows).set_index("model")


def compare_feature_engineering(
    X_train: pd.DataFrame, y_train: pd.Series
) -> dict[str, dict[str, float]]:
    """Controlled Logistic Regression comparison on the exact same folds."""
    cv = make_cv()
    without = evaluate_pipeline(
        build_model_pipeline("Logistic Regression", feature_engineering=False),
        X_train,
        y_train,
        cv,
    )
    with_features = evaluate_pipeline(
        build_model_pipeline("Logistic Regression", feature_engineering=True),
        X_train,
        y_train,
        cv,
    )
    delta = {
        key: with_features[key] - without[key]
        for key in with_features
        if key.endswith("_mean")
    }
    return {"without_feature_engineering": without, "with_feature_engineering": with_features, "delta": delta}


def out_of_fold_predictions(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate train-only OOF labels and positive-class probabilities."""
    predictions = cross_val_predict(
        build_model_pipeline(model_name),
        X_train,
        y_train,
        cv=make_cv(),
        method="predict",
        n_jobs=-1,
    )
    probabilities = cross_val_predict(
        build_model_pipeline(model_name),
        X_train,
        y_train,
        cv=make_cv(),
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]
    return predictions, probabilities


def _plot_model_metrics(results: pd.DataFrame, figures_dir: Path) -> Path:
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    plot_data = results[[f"{metric}_mean" for metric in metrics]].copy()
    plot_data.columns = [metric.upper().replace("_", "-") for metric in metrics]
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_data.plot(kind="bar", ax=ax, color=sns.color_palette("deep", len(metrics)))
    ax.set(title="Comparaison des modèles — CV stratifiée sur le train", ylabel="Score moyen CV", xlabel="")
    ax.set_ylim(0, 1)
    ax.legend(ncol=3, loc="lower right")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    path = figures_dir / "ml_cv_model_comparison.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_confusions(
    matrices: dict[str, list[list[int]]], figures_dir: Path
) -> Path:
    fig, axes = plt.subplots(1, len(matrices), figsize=(6 * len(matrices), 5))
    axes_array = np.atleast_1d(axes)
    for ax, (model_name, matrix) in zip(axes_array, matrices.items()):
        sns.heatmap(
            np.asarray(matrix), annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"], ax=ax,
        )
        ax.set(title=f"OOF — {model_name}", xlabel="Prédit", ylabel="Réel")
    fig.tight_layout()
    path = figures_dir / "ml_oof_confusion_matrices.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_roc(
    y_train: pd.Series,
    probabilities: dict[str, np.ndarray],
    results: pd.DataFrame,
    figures_dir: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(7, 6))
    for model_name, scores in probabilities.items():
        fpr, tpr, _ = roc_curve(y_train, scores)
        auc = results.loc[model_name, "roc_auc_mean"]
        ax.plot(fpr, tpr, label=f"{model_name} (CV AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Aléatoire")
    ax.set(title="Courbes ROC out-of-fold — train uniquement", xlabel="False Positive Rate", ylabel="True Positive Rate")
    ax.legend(loc="lower right")
    fig.tight_layout()
    path = figures_dir / "ml_oof_roc_curves.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def run_experiment(
    dataset: Path = DEFAULT_DATASET,
    results_csv: Path = DEFAULT_RESULTS_CSV,
    results_json: Path = DEFAULT_RESULTS_JSON,
    figures_dir: Path = DEFAULT_FIGURES_DIR,
) -> dict[str, object]:
    """Run the complete train-only benchmark; the test set is never consumed."""
    data = pd.read_csv(dataset)
    X, y = prepare_features(data)
    split = split_data(X, y)
    X_train, y_train = split.X_train, split.y_train

    results = compare_models(X_train, y_train)
    real_models = results.drop(index="Dummy")
    provisional_best = str(real_models["roc_auc_mean"].idxmax())
    feature_comparison = compare_feature_engineering(X_train, y_train)

    oof_models = ["Dummy", "Logistic Regression"]
    if provisional_best not in oof_models:
        oof_models.append(provisional_best)
    matrices: dict[str, list[list[int]]] = {}
    probabilities: dict[str, np.ndarray] = {}
    for model_name in oof_models:
        predicted, probability = out_of_fold_predictions(model_name, X_train, y_train)
        matrices[model_name] = confusion_matrix(y_train, predicted, labels=[0, 1]).tolist()
        probabilities[model_name] = probability

    figures_dir.mkdir(parents=True, exist_ok=True)
    figure_paths = [
        _plot_model_metrics(results, figures_dir),
        _plot_confusions(matrices, figures_dir),
        _plot_roc(y_train, probabilities, results, figures_dir),
    ]

    results_csv.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(results_csv, float_format="%.10f")
    payload: dict[str, object] = {
        "protocol": {
            "dataset": dataset.as_posix(),
            "train_rows": len(X_train),
            "reserved_test_rows": len(split.X_test),
            "test_set_consumed": False,
            "cv": {"type": "StratifiedKFold", "n_splits": CV_FOLDS, "shuffle": True, "random_state": RANDOM_STATE},
            "positive_class": 1,
            "selection_metric": "roc_auc_mean",
        },
        "cv_results": results.reset_index().to_dict(orient="records"),
        "provisional_best": provisional_best,
        "feature_engineering_comparison": feature_comparison,
        "oof_confusion_matrices": matrices,
        "figures": [path.as_posix() for path in figure_paths],
    }
    results_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Train-only ML experiment complete. Provisional best: {provisional_best}")
    print("Test set not consumed.")
    return payload


if __name__ == "__main__":
    run_experiment()
