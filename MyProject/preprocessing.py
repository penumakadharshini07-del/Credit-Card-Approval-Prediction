"""Data loading, cleaning, feature engineering, and preprocessing utilities."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


LOGGER = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent

TARGET_COLUMN = "TARGET"
APPROVED_LABEL = 1
REJECTED_LABEL = 0


@dataclass
class DataPaths:
    """Locations of the raw source files."""

    application_path: Path = BASE_DIR / "data" / "application_record.csv"
    credit_path: Path = BASE_DIR / "data" / "credit_record.csv"


def _log_step(message: str) -> None:
    print(f"[PREPROCESSING] {message}")
    LOGGER.info(message)


def load_raw_data(paths: DataPaths | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the application and credit record datasets."""

    paths = paths or DataPaths()
    _log_step(f"Loading application data from {paths.application_path}")
    application = pd.read_csv(paths.application_path)
    _log_step(f"Loading credit data from {paths.credit_path}")
    credit = pd.read_csv(paths.credit_path)
    return application, credit


def build_target_from_credit(credit: pd.DataFrame) -> pd.DataFrame:
    """Create a binary target from repayment status.

    TARGET = 1 means Approved/low risk.
    TARGET = 0 means Rejected/high risk.
    A customer is considered risky if any status is 2, 3, 4, or 5.
    """

    _log_step("Generating target column from credit repayment statuses")
    risky_statuses = {"2", "3", "4", "5"}
    working = credit.copy()
    working["STATUS"] = working["STATUS"].astype(str).str.upper()
    target = (
        working.assign(IS_RISKY=working["STATUS"].isin(risky_statuses).astype(int))
        .groupby("ID", as_index=False)["IS_RISKY"]
        .max()
    )
    target[TARGET_COLUMN] = np.where(target["IS_RISKY"].eq(1), REJECTED_LABEL, APPROVED_LABEL)
    return target[["ID", TARGET_COLUMN]]


def merge_datasets(application: pd.DataFrame, credit: pd.DataFrame) -> pd.DataFrame:
    """Merge application records with generated target labels."""

    target = build_target_from_credit(credit)
    _log_step("Merging datasets on ID")
    merged = application.merge(target, on="ID", how="inner")
    _log_step(f"Merged dataset shape: {merged.shape}")
    return merged


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean duplicate rows, missing values, and outliers."""

    cleaned = df.copy()
    before = len(cleaned)
    cleaned = cleaned.drop_duplicates()
    _log_step(f"Removed duplicates: {before - len(cleaned)} rows")

    if "OCCUPATION_TYPE" in cleaned.columns:
        cleaned["OCCUPATION_TYPE"] = cleaned["OCCUPATION_TYPE"].fillna("Unknown")
        _log_step("Filled missing OCCUPATION_TYPE values with 'Unknown'")

    numeric_columns = cleaned.select_dtypes(include=["number"]).columns.tolist()
    numeric_columns = [col for col in numeric_columns if col not in {"ID", TARGET_COLUMN}]
    for column in numeric_columns:
        q1 = cleaned[column].quantile(0.25)
        q3 = cleaned[column].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        clipped = cleaned[column].clip(lower=lower, upper=upper)
        outliers = int((cleaned[column] != clipped).sum())
        cleaned[column] = clipped
        _log_step(f"Detected and capped {outliers} outliers in {column}")

    _log_step("Handled missing values and numeric outliers")
    return cleaned


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create business-friendly features for credit risk prediction."""

    engineered = df.copy()
    _log_step("Creating age and employment experience features")
    engineered["AGE"] = (-engineered["DAYS_BIRTH"] / 365.25).round(1)
    engineered["YEARS_EMPLOYED"] = np.where(
        engineered["DAYS_EMPLOYED"] > 0,
        0,
        (-engineered["DAYS_EMPLOYED"] / 365.25).round(1),
    )

    engineered["AGE_GROUP"] = pd.cut(
        engineered["AGE"],
        bins=[0, 25, 35, 45, 55, 120],
        labels=["Young", "Early Career", "Mid Career", "Senior", "Retired"],
        include_lowest=True,
    ).astype(str)
    engineered["INCOME_CATEGORY"] = pd.cut(
        engineered["AMT_INCOME_TOTAL"],
        bins=[0, 100000, 200000, 350000, np.inf],
        labels=["Low", "Moderate", "High", "Very High"],
        include_lowest=True,
    ).astype(str)
    engineered["FAMILY_INCOME_RATIO"] = (
        engineered["AMT_INCOME_TOTAL"] / engineered["CNT_FAM_MEMBERS"].replace(0, 1)
    ).round(2)
    engineered["CHILDREN_CATEGORY"] = pd.cut(
        engineered["CNT_CHILDREN"],
        bins=[-1, 0, 2, np.inf],
        labels=["No Children", "Small Family", "Large Family"],
    ).astype(str)
    engineered["WORKING_STATUS"] = np.where(engineered["YEARS_EMPLOYED"] > 0, "Working", "Not Working")
    engineered["HOUSING_OWNERSHIP"] = np.where(
        engineered["NAME_HOUSING_TYPE"].str.contains("House", case=False, na=False),
        "Stable Housing",
        "Other Housing",
    )
    engineered["MONTHLY_INCOME_LEVEL"] = (engineered["AMT_INCOME_TOTAL"] / 12).round(2)
    engineered["FINANCIAL_STABILITY_SCORE"] = (
        (engineered["FLAG_OWN_REALTY"].eq("Y").astype(int) * 2)
        + engineered["FLAG_OWN_CAR"].eq("Y").astype(int)
        + np.minimum(engineered["YEARS_EMPLOYED"], 10) / 5
        + np.minimum(engineered["FAMILY_INCOME_RATIO"] / 100000, 3)
    ).round(2)
    engineered["RISK_CATEGORY"] = pd.cut(
        engineered["FINANCIAL_STABILITY_SCORE"],
        bins=[-np.inf, 2, 4, np.inf],
        labels=["High Risk", "Medium Risk", "Low Risk"],
    ).astype(str)
    engineered["CUSTOMER_SEGMENT"] = (
        engineered["AGE_GROUP"].astype(str)
        + " - "
        + engineered["INCOME_CATEGORY"].astype(str)
        + " - "
        + engineered["RISK_CATEGORY"].astype(str)
    )
    _log_step("Created requested feature engineering columns")
    return engineered


def remove_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove identifiers and raw date-offset columns after feature engineering."""

    removable = ["ID", "DAYS_BIRTH", "DAYS_EMPLOYED"]
    existing = [column for column in removable if column in df.columns]
    _log_step(f"Removing unnecessary columns: {existing}")
    return df.drop(columns=existing)


def get_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return numeric and categorical columns for preprocessing."""

    feature_df = df.drop(columns=[TARGET_COLUMN], errors="ignore")
    numeric = feature_df.select_dtypes(include=["number"]).columns.tolist()
    categorical = feature_df.select_dtypes(exclude=["number"]).columns.tolist()
    _log_step(f"Numeric columns selected for scaling: {numeric}")
    _log_step(f"Categorical columns selected for one-hot encoding: {categorical}")
    return numeric, categorical


def build_preprocessor(numeric_features: Iterable[str], categorical_features: Iterable[str]) -> ColumnTransformer:
    """Build the sklearn preprocessing transformer."""

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, list(numeric_features)),
            ("categorical", categorical_pipeline, list(categorical_features)),
        ]
    )


def prepare_model_data(paths: DataPaths | None = None) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Return X, y, and the fully engineered dataset."""

    application, credit = load_raw_data(paths)
    merged = merge_datasets(application, credit)
    cleaned = clean_data(merged)
    engineered = engineer_features(cleaned)
    final_df = remove_unnecessary_columns(engineered)

    _log_step("Running correlation analysis on numeric features")
    correlation = final_df.select_dtypes(include=["number"]).corr(numeric_only=True)
    if TARGET_COLUMN in correlation.columns:
        print(correlation[TARGET_COLUMN].sort_values(ascending=False))

    X = final_df.drop(columns=[TARGET_COLUMN])
    y = final_df[TARGET_COLUMN]
    _log_step(f"Final feature matrix shape: {X.shape}")
    return X, y, final_df


def build_single_record(form_data: dict) -> pd.DataFrame:
    """Convert web form data into the same raw columns used by the model."""

    age = float(form_data["age"])
    years_employed = float(form_data["years_employed"])
    income = float(form_data["income"])
    return pd.DataFrame(
        [
            {
                "ID": 0,
                "CODE_GENDER": form_data["gender"],
                "FLAG_OWN_CAR": form_data["car_ownership"],
                "FLAG_OWN_REALTY": form_data["real_estate_ownership"],
                "CNT_CHILDREN": int(form_data["children"]),
                "AMT_INCOME_TOTAL": income,
                "NAME_INCOME_TYPE": form_data["income_type"],
                "NAME_EDUCATION_TYPE": form_data["education"],
                "NAME_FAMILY_STATUS": form_data["marital_status"],
                "NAME_HOUSING_TYPE": form_data["housing_type"],
                "DAYS_BIRTH": int(-age * 365.25),
                "DAYS_EMPLOYED": int(-years_employed * 365.25) if years_employed > 0 else 365243,
                "OCCUPATION_TYPE": form_data["occupation"],
                "CNT_FAM_MEMBERS": int(form_data["family_members"]),
                TARGET_COLUMN: APPROVED_LABEL,
            }
        ]
    )


def transform_for_prediction(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Apply cleaning and feature engineering to new prediction records."""

    cleaned = clean_data(raw_df)
    engineered = engineer_features(cleaned)
    final_df = remove_unnecessary_columns(engineered)
    return final_df.drop(columns=[TARGET_COLUMN], errors="ignore")
