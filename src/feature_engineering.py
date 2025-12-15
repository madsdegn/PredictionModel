# Predicting Heart Attacks Using Machine Learning
# A project submitted for the Subject Module Project in Computer Science

# Mads Degn, Julia Lundager, Daniel Holst Pedersen, Jonas Pheiffer, Magnus Stilling Østergaard
# 18/12-25

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MinMaxScaler
from src.config import NUMERIC_FEATURES, CATEGORICAL_FEATURES


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom transformer that adds engineered medical risk features to the dataset.

    Inputs
    X : pandas.DataFrame
        A dataset containing numeric and categorical patient features,
        including systolic and diastolic blood pressure and cholesterol.

    Processing steps
    1. Compute Pulse Pressure as the difference between systolic and diastolic blood pressure.
    2. Compute Mean Arterial Pressure (MAP) using systolic and diastolic values.
    3. Create a binary flag for high blood pressure (systolic ≥ 130 or diastolic ≥ 80).
    4. Create a binary flag for high cholesterol (cholesterol ≥ 240).
    5. Compute a composite Risk Score by summing high blood pressure, high cholesterol, and smoking status.

    Outputs
    X : pandas.DataFrame
        A copy of the input dataset with five new engineered columns:
        Pulse_Pressure, MAP, High Blood Pressure, High Cholesterol, Risk_Score.
    """

    def fit(self, X, y=None):
        # No fitting required, transformer only adds features
        return self

    def transform(self, X):
        X = X.copy()

        # Engineered features based on medical risk factors
        X["Pulse_Pressure"] = X["Systolic_BP"] - X["Diastolic_BP"]
        X["MAP"] = X["Diastolic_BP"] + ((X["Systolic_BP"] - X["Diastolic_BP"]) / 3)
        X["High Blood Pressure"] = ((X["Systolic_BP"] >= 130) | (X["Diastolic_BP"] >= 80)).astype(int)
        X["High Cholesterol"] = (X["Cholesterol"] >= 240).astype(int)
        X["Risk_Score"] = X["High Blood Pressure"] + X["High Cholesterol"] + X["Smoking"]

        return X


def build_preprocessor_scaled():
    """
    Builds a preprocessing pipeline with scaling applied to numeric features.

    Inputs
    None directly; uses NUMERIC_FEATURES and CATEGORICAL_FEATURES defined in config.py.

    Processing steps
    1. Apply FeatureEngineer to add engineered medical risk features.
    2. Scale all numeric features (original + engineered) using MinMaxScaler.
    3. Encode categorical features:
       - Continent and Country with one-hot encoding.
       - Diet, Sex, Hemisphere with ordinal encoding.

    Outputs
    preprocessor : sklearn.Pipeline
        A pipeline that transforms raw patient data into scaled numeric features
        and encoded categorical features, ready for model training.
    """
    num_pipe = Pipeline(steps=[("scaler", MinMaxScaler())])

    cat_transform = ColumnTransformer(
        transformers=[
            #("continent", OneHotEncoder(handle_unknown="ignore"), ["Continent"]),
            #("country", OneHotEncoder(handle_unknown="ignore"), ["Country"]),
            ("diet", OrdinalEncoder(), ["Diet"]),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    preprocessor = Pipeline(steps=[
        ("features", FeatureEngineer()),
        ("transform", ColumnTransformer(
            transformers=[
                ("num", num_pipe, NUMERIC_FEATURES + [
                    "Pulse_Pressure", "MAP",
                    "High Blood Pressure", "High Cholesterol", "Risk_Score"
                ]),
                ("cat", cat_transform, CATEGORICAL_FEATURES),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        ))
    ])

    return preprocessor


def build_preprocessor_not_scaled():
    """
    Builds a preprocessing pipeline without scaling numeric features.

    Inputs
    None directly; uses NUMERIC_FEATURES and CATEGORICAL_FEATURES defined in config.py.

    Processing steps
    1. Apply FeatureEngineer to add engineered medical risk features.
    2. Pass numeric features through without scaling.
    3. Encode categorical features:
       - Continent and Country with one-hot encoding.
       - Diet, Sex, Hemisphere with ordinal encoding.

    Outputs
    preprocessor : sklearn.Pipeline
        A pipeline that transforms raw patient data into unscaled numeric features
        and encoded categorical features, ready for model training.
    """
    num_pipe = "passthrough"

    cat_transform = ColumnTransformer(
        transformers=[
            #("continent", OneHotEncoder(handle_unknown="ignore"), ["Continent"]),
            #("country", OneHotEncoder(handle_unknown="ignore"), ["Country"]),
            ("diet", OrdinalEncoder(), ["Diet"]),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    preprocessor = Pipeline(steps=[
        ("features", FeatureEngineer()),
        ("transform", ColumnTransformer(
            transformers=[
                ("num", num_pipe, NUMERIC_FEATURES + [
                    "Pulse_Pressure", "MAP",
                    "High Blood Pressure", "High Cholesterol", "Risk_Score"
                ]),
                ("cat", cat_transform, CATEGORICAL_FEATURES),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        ))
    ])

    return preprocessor