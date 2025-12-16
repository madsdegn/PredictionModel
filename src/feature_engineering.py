# Predicting Heart Attacks Using Machine Learning
# A project submitted for the Subject Module Project in Computer Science

# Mads Degn, Julia Lundager, Daniel Holst Pedersen, Jonas Pheiffer, Magnus Stilling Østergaard
# 18/12-25

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from src.config import CONTINUOUS_FEATURES, BINARY_FEATURES, ORDINAL_FEATURES


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom transformer that adds engineered medical risk features to the dataset.
    """

    def fit(self, X, y=None):
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
    Preprocessor for models that require scaling (LogReg, KNN).
    - Scales continuous numeric features
    - Leaves binary numeric features untouched
    - Keeps ordinal features (already encoded in data_cleaning)
    - Adds engineered medical risk features
    """
    num_pipe = Pipeline(steps=[("scaler", MinMaxScaler())])

    preprocessor = Pipeline(steps=[
        ("features", FeatureEngineer()),
        ("transform", ColumnTransformer(
            transformers=[
                ("num_scaled", num_pipe, CONTINUOUS_FEATURES + [
                    "Pulse_Pressure", "MAP"
                ]),
                ("num_binary", "passthrough", BINARY_FEATURES + [
                    "High Blood Pressure", "High Cholesterol", "Risk_Score"
                ]),
                ("ordinal", "passthrough", ORDINAL_FEATURES),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        ))
    ])

    return preprocessor


def build_preprocessor_not_scaled():
    """
    Preprocessor for tree-based models (DT, RF, XGB).
    - Passes continuous, binary, and ordinal features unchanged
    - Adds engineered medical risk features
    """
    preprocessor = Pipeline(steps=[
        ("features", FeatureEngineer()),
        ("transform", ColumnTransformer(
            transformers=[
                ("num", "passthrough", CONTINUOUS_FEATURES + BINARY_FEATURES + ORDINAL_FEATURES + [
                    "Pulse_Pressure", "MAP",
                    "High Blood Pressure", "High Cholesterol", "Risk_Score"
                ]),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        ))
    ])

    return preprocessor