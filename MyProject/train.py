"""Train and evaluate credit card approval prediction models."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
BASE_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / "reports" / ".matplotlib"))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
try:
    import shap
except ImportError:  # pragma: no cover - dependency is installed from requirements.txt
    shap = None
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from preprocessing import build_preprocessor, get_feature_columns, prepare_model_data


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger(__name__)

MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
PLOTS_DIR = REPORTS_DIR / "plots"


def get_models() -> dict:
    """Return candidate models for comparison."""

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(random_state=42, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(random_state=42, class_weight="balanced"),
        "Support Vector Machine": SVC(probability=True, class_weight="balanced", random_state=42),
        "KNN": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }
    if shap is None:
        LOGGER.warning("SHAP unavailable and summary plot will be skipped.")

    try:
        from xgboost import XGBClassifier

        models["XGBoost"] = XGBClassifier(
            random_state=42,
            eval_metric="logloss",
            n_estimators=100,
            learning_rate=0.05,
        )
    except Exception as exc:  # pragma: no cover - optional dependency
        LOGGER.warning("XGBoost unavailable and will be skipped: %s", exc)
    return models


def evaluate_model(name: str, pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Calculate requested evaluation metrics for a fitted model."""

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else y_pred
    return {
        "model": name,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4) if len(set(y_test)) > 1 else 0,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, zero_division=0),
    }


def create_evaluation_plots(best_pipeline: Pipeline, X_train, X_test, y_train, y_test) -> None:
    """Save ROC, precision-recall, confusion matrix, and feature importance charts."""

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    y_pred = best_pipeline.predict(X_test)
    y_proba = best_pipeline.predict_proba(X_test)[:, 1]

    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=["Rejected", "Approved"])
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "confusion_matrix.png")
    plt.close()

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(8, 5))
    plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc_score(y_test, y_proba):.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "roc_curve.png")
    plt.close()

    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    plt.figure(figsize=(8, 5))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision Recall Curve")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "precision_recall_curve.png")
    plt.close()

    model = best_pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        preprocessor = best_pipeline.named_steps["preprocessor"]
        feature_names = preprocessor.get_feature_names_out()
        importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False).head(15)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=importances.values, y=importances.index)
        plt.title("Top Feature Importances")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "feature_importance.png")
        plt.close()

    if shap is None:
        return

    try:
        transformed_train = best_pipeline.named_steps["preprocessor"].transform(X_train)
        transformed_test = best_pipeline.named_steps["preprocessor"].transform(X_test)
        feature_names = best_pipeline.named_steps["preprocessor"].get_feature_names_out()
        explainer = shap.Explainer(best_pipeline.named_steps["model"], transformed_train, feature_names=feature_names)
        shap_values = explainer(transformed_test[: min(100, len(transformed_test))])
        shap.summary_plot(shap_values, transformed_test[: min(100, len(transformed_test))], feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "shap_summary.png", bbox_inches="tight")
        plt.close()
    except Exception as exc:
        LOGGER.warning("SHAP summary plot could not be generated: %s", exc)


def tune_best_model(best_name: str, base_pipeline: Pipeline, X_train, y_train) -> Pipeline:
    """Tune tree-based models using GridSearchCV or RandomizedSearchCV."""

    LOGGER.info("Hyperparameter tuning selected model: %s", best_name)
    grids = {
        "Random Forest": {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [None, 5, 10, 20],
            "model__min_samples_split": [2, 5, 10],
        },
        "Decision Tree": {
            "model__max_depth": [None, 3, 5, 10, 20],
            "model__min_samples_split": [2, 5, 10],
            "model__criterion": ["gini", "entropy"],
        },
        "Gradient Boosting": {
            "model__n_estimators": [50, 100, 150],
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__max_depth": [2, 3, 5],
        },
        "XGBoost": {
            "model__n_estimators": [50, 100, 200],
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__max_depth": [2, 3, 5],
        },
    }
    if best_name not in grids:
        LOGGER.info("Selected model has no tuning grid; using fitted base pipeline.")
        return base_pipeline

    if best_name in {"Random Forest", "XGBoost"}:
        search = RandomizedSearchCV(
            base_pipeline,
            grids[best_name],
            n_iter=8,
            cv=3,
            scoring="f1",
            n_jobs=-1,
            random_state=42,
            error_score="raise",
        )
    else:
        search = GridSearchCV(
            base_pipeline,
            grids[best_name],
            cv=3,
            scoring="f1",
            n_jobs=-1,
            error_score="raise",
        )
    search.fit(X_train, y_train)
    LOGGER.info("Best tuning parameters: %s", search.best_params_)
    return search.best_estimator_


def train() -> None:
    """Run the full model training workflow."""

    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    X, y, full_df = prepare_model_data()
    numeric, categorical = get_feature_columns(full_df)
    preprocessor = build_preprocessor(numeric, categorical)

    stratify = y if y.nunique() > 1 and y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=stratify
    )

    results = []
    fitted_pipelines = {}
    for name, model in get_models().items():
        LOGGER.info("Training %s", name)
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
        pipeline.fit(X_train, y_train)
        cv = cross_val_score(pipeline, X, y, cv=3, scoring="f1").mean()
        metrics = evaluate_model(name, pipeline, X_test, y_test)
        metrics["cross_validation_f1"] = round(float(cv), 4)
        results.append(metrics)
        fitted_pipelines[name] = pipeline

    comparison = pd.DataFrame(results).sort_values(
        by=["f1_score", "roc_auc", "accuracy"], ascending=False
    )
    comparison.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    (REPORTS_DIR / "classification_reports.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    LOGGER.info("Model comparison:\n%s", comparison[["model", "accuracy", "precision", "recall", "f1_score", "roc_auc"]])

    best_name = comparison.iloc[0]["model"]
    tuned_pipeline = tune_best_model(best_name, fitted_pipelines[best_name], X_train, y_train)
    tuned_pipeline.fit(X_train, y_train)
    create_evaluation_plots(tuned_pipeline, X_train, X_test, y_train, y_test)

    joblib.dump(tuned_pipeline, MODELS_DIR / "best_model.pkl")
    joblib.dump(tuned_pipeline.named_steps["preprocessor"], MODELS_DIR / "preprocessor.pkl")
    joblib.dump({"best_model": best_name, "metrics": evaluate_model(best_name, tuned_pipeline, X_test, y_test)}, MODELS_DIR / "model_metadata.pkl")
    LOGGER.info("Saved best model and preprocessing artifacts in %s", MODELS_DIR)


if __name__ == "__main__":
    train()
