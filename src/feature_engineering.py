# src/feature_engineering.py

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from src.config import NUMERIC_FEATURES, CATEGORICAL_FEATURES

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Custom transformer to add engineered features."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Pulse Pressure
        X["Pulse_Pressure"] = X["Systolic_BP"] - X["Diastolic_BP"]

        # Mean Arterial Pressure
        X["MAP"] = X["Diastolic_BP"] + (
            (X["Systolic_BP"] - X["Diastolic_BP"]) / 3
        )

        # Activity Score
        X["Activity_Score"] = (
            X["Exercise Hours Per Week"]
            + X["Physical Activity Days Per Week"] * 0.5
            - X["Sedentary Hours Per Day"] * 0.3
        )

        # RiskFactor Count
        risk_cols = [
            "Smoking", "Diabetes", "Obesity",
            "Previous Heart Problems", "Family History", "Medication Use"
        ]
        X["RiskFactor_Count"] = X[risk_cols].sum(axis=1)

        # Activity Ratio
        X["Activity_Ratio"] = X["Exercise Hours Per Week"] / X["Sedentary Hours Per Day"]

        # Optional features
        X["Metabolic_Core"] = X["Diabetes"] + X["Obesity"]
        X["CardiacHistory_Core"] = X["Previous Heart Problems"] + X["Medication Use"]

        return X


def build_preprocessor():
    """
    Build preprocessing pipeline:
    - Custom feature engineering
    - Numeric: median imputation + scaling
    - Categorical: most frequent imputation + one-hot encoding
    """
    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = Pipeline(steps=[
        ("features", FeatureEngineer()),   # <-- add engineered features first
        ("transform", ColumnTransformer(
            transformers=[
                ("num", num_pipe, NUMERIC_FEATURES + [
                    "Pulse_Pressure", "MAP", "Activity_Score",
                    "RiskFactor_Count", "Activity_Ratio",
                    "Metabolic_Core", "CardiacHistory_Core"
                ]),
                ("cat", cat_pipe, CATEGORICAL_FEATURES),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        ))
    ])

    return preprocessor