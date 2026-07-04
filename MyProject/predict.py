"""Prediction service helpers for the Flask application and CLI usage."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import joblib
import pandas as pd

from preprocessing import build_single_record, transform_for_prediction


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
METADATA_PATH = BASE_DIR / "models" / "model_metadata.pkl"
MINIMUM_APPROVAL_AGE = 18
LOW_INCOME_REJECTION_LIMIT = 100000


@dataclass
class PredictionResult:
    label: str
    probability: float
    confidence: float
    risk_level: str
    suggestions: list[str]


class CreditApprovalPredictor:
    """Load the trained model pipeline and serve predictions."""

    def __init__(self, model_path: Path = MODEL_PATH, allow_rule_fallback: bool = True):
        self.pipeline = None
        self.using_rule_fallback = False
        if model_path.exists():
            try:
                self.pipeline = joblib.load(model_path)
            except Exception:
                if not allow_rule_fallback:
                    raise
                self.using_rule_fallback = True
        elif allow_rule_fallback:
            self.using_rule_fallback = True
        else:
            raise FileNotFoundError(
                "Model file not found. Run `python train.py` before starting the app."
            )

    def predict_dataframe(self, raw_records: pd.DataFrame) -> pd.DataFrame:
        result = raw_records.copy()
        if self.pipeline is not None:
            features = transform_for_prediction(raw_records)
            predictions = self.pipeline.predict(features)
            probabilities = self.pipeline.predict_proba(features)[:, 1]
            result["prediction"] = ["Approved" if value == 1 else "Rejected" for value in predictions]
            result["approval_probability"] = (probabilities * 100).round(2)
        else:
            result["prediction"] = "Rejected"
            result["approval_probability"] = 0.0
        result = apply_credit_rules(result)
        return result

    def predict_form(self, form_data: dict) -> PredictionResult:
        raw_record = build_single_record(form_data)
        prediction_df = self.predict_dataframe(raw_record)
        probability = float(prediction_df.loc[0, "approval_probability"])
        label = str(prediction_df.loc[0, "prediction"])
        confidence = probability if label == "Approved" else 100 - probability
        risk_level = _risk_level(probability)
        return PredictionResult(
            label=label,
            probability=probability,
            confidence=round(confidence, 2),
            risk_level=risk_level,
            suggestions=_suggestions(label, probability, form_data),
        )


def _risk_level(probability: float) -> str:
    if probability >= 75:
        return "Low Risk"
    if probability >= 50:
        return "Moderate Risk"
    return "High Risk"


def apply_rejection_rules(result: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible wrapper for the final credit rule layer."""

    return apply_credit_rules(result)


def apply_credit_rules(result: pd.DataFrame) -> pd.DataFrame:
    """Apply transparent credit eligibility and approval rules after the ML score."""

    output = result.copy()
    default_numeric = pd.Series(0, index=output.index)
    income = pd.to_numeric(output.get("AMT_INCOME_TOTAL", default_numeric), errors="coerce").fillna(0)
    age = (-pd.to_numeric(output.get("DAYS_BIRTH", default_numeric), errors="coerce").fillna(0) / 365.25)
    years_employed = (
        -pd.to_numeric(output.get("DAYS_EMPLOYED", default_numeric), errors="coerce").fillna(365243) / 365.25
    ).clip(lower=0, upper=60)
    family_members = pd.to_numeric(
        output.get("CNT_FAM_MEMBERS", pd.Series(1, index=output.index)), errors="coerce"
    ).fillna(1).clip(lower=1)
    children = pd.to_numeric(output.get("CNT_CHILDREN", default_numeric), errors="coerce").fillna(0)
    income_type = output.get("NAME_INCOME_TYPE", pd.Series("", index=output.index)).astype(str).str.lower()
    occupation = output.get("OCCUPATION_TYPE", pd.Series("", index=output.index)).astype(str).str.lower()
    owns_car = output.get("FLAG_OWN_CAR", pd.Series("", index=output.index)).astype(str).str.upper().eq("Y")
    owns_realty = output.get("FLAG_OWN_REALTY", pd.Series("", index=output.index)).astype(str).str.upper().eq("Y")

    reject_mask = age.lt(MINIMUM_APPROVAL_AGE) | income.lt(LOW_INCOME_REJECTION_LIMIT) | income_type.isin(
        ["student", "unemployed"]
    )
    income_per_member = income / family_members
    rule_score = pd.Series(38.0, index=output.index)
    income_score = ((income - LOW_INCOME_REJECTION_LIMIT) / 20000).clip(lower=0, upper=28)
    family_income_score = ((income_per_member - 60000) / 10000).clip(lower=0, upper=7)
    rule_score += income_score
    rule_score += years_employed.ge(4.95) * 12
    rule_score += years_employed.between(2, 4.949, inclusive="both") * 8
    rule_score += years_employed.between(1, 1.999, inclusive="both") * 4
    rule_score += family_income_score
    rule_score += age.between(21, 60, inclusive="both") * 4
    rule_score += owns_realty * 5
    rule_score += owns_car * 3
    rule_score += occupation.isin(["managers", "accountants", "core staff", "drivers", "sales staff"]) * 4
    rule_score -= family_members.ge(6) * 5
    rule_score -= children.ge(3) * 5
    rule_score = rule_score.clip(lower=5, upper=95)

    rejection_score = rule_score.clip(upper=45)
    rejection_score -= age.lt(MINIMUM_APPROVAL_AGE) * 8
    rejection_score -= income.lt(LOW_INCOME_REJECTION_LIMIT) * 10
    rejection_score -= income_type.eq("student") * 6
    rejection_score -= income_type.eq("unemployed") * 9
    rejection_score = rejection_score.clip(lower=15, upper=45)

    output.loc[reject_mask, "prediction"] = "Rejected"
    output.loc[reject_mask, "approval_probability"] = rejection_score.loc[reject_mask].round(2)

    eligible_mask = ~reject_mask
    output.loc[eligible_mask, "approval_probability"] = rule_score.loc[eligible_mask].round(2)

    approve_mask = eligible_mask & rule_score.ge(55)
    output.loc[approve_mask, "prediction"] = "Approved"
    review_mask = eligible_mask & rule_score.lt(50)
    output.loc[review_mask, "prediction"] = "Rejected"
    return output


def _suggestions(label: str, probability: float, form_data: dict) -> list[str]:
    suggestions = []
    income = float(form_data.get("income", 0))
    years = float(form_data.get("years_employed", 0))
    age = float(form_data.get("age", 0))
    income_type = form_data.get("income_type", "")
    if label == "Approved":
        suggestions.append("Maintain on-time payments to preserve a healthy credit profile.")
    else:
        suggestions.append("Improve repayment history and reduce existing financial risk before reapplying.")
    if age < 18:
        suggestions.append("Applicant must be at least 18 years old before applying.")
    if income_type in {"Student", "Unemployed"}:
        suggestions.append("Stable employment or eligible income is required before reapplying.")
    if income < 150000:
        suggestions.append("Increasing stable monthly income can improve approval chances.")
    if years < 1:
        suggestions.append("A longer employment history may improve confidence in the application.")
    if probability < 60:
        suggestions.append("Consider applying with stronger supporting documents or a lower credit limit.")
    return suggestions


def predict_from_cli(data: dict) -> PredictionResult:
    predictor = CreditApprovalPredictor()
    return predictor.predict_form(data)
