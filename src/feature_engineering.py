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

    Methods
    fit(X, y=None):
        Required by scikit-learn, but does nothing here since no fitting is needed.
    transform(X):
        Creates new medical risk features based on blood pressure, cholesterol, and smoking.

    Inputs
    X : pandas.DataFrame
        Dataset containing patient features.

    Outputs
    X : pandas.DataFrame
        Copy of the dataset with additional engineered features:
        - Pulse_Pressure: difference between systolic and diastolic blood pressure
        - MAP: mean arterial pressure
        - High Blood Pressure: binary indicator for hypertension
        - High Cholesterol: binary indicator for cholesterol ≥ 240
        - Risk_Score: combined risk score based on blood pressure, cholesterol, and smoking
    """

    def fit(self, X, y=None):
        # No fitting required, so return self unchanged
        return self

    def transform(self, X):
        # Make a copy of the dataset to avoid modifying the original
        X = X.copy()

        # Calculate pulse pressure (systolic - diastolic)
        X["Pulse_Pressure"] = X["Systolic_BP"] - X["Diastolic_BP"]

        # Calculate mean arterial pressure (MAP)
        X["MAP"] = X["Diastolic_BP"] + ((X["Systolic_BP"] - X["Diastolic_BP"]) / 3)

        # Create binary indicator for high blood pressure (≥130 systolic or ≥80 diastolic)
        X["High Blood Pressure"] = ((X["Systolic_BP"] >= 130) | (X["Diastolic_BP"] >= 80)).astype(int)

        # Create binary indicator for high cholesterol (≥240)
        X["High Cholesterol"] = (X["Cholesterol"] >= 240).astype(int)

        # Create combined risk score (sum of hypertension, high cholesterol, and smoking indicators)
        X["Risk_Score"] = X["High Blood Pressure"] + X["High Cholesterol"] + X["Smoking"]

        # Return the dataset with new engineered features
        return X


def build_preprocessor_scaled():
    """
    Builds a preprocessing pipeline for models that require scaling (Logistic Regression, KNN).

    Processing steps
    1. Scale continuous numeric features using MinMaxScaler.
    2. Pass binary features unchanged.
    3. Pass ordinal features unchanged.
    4. Add engineered medical risk features from FeatureEngineer.

    Inputs
    None directly; uses feature lists from config.py.

    Outputs
    preprocessor : sklearn.Pipeline
        Pipeline that applies feature engineering and scaling transformations.
    """

    # Define pipeline for scaling continuous features
    num_pipe = Pipeline(steps=[("scaler", MinMaxScaler())])

    # Build preprocessing pipeline
    preprocessor = Pipeline(steps=[
        ("features", FeatureEngineer()),  # Add engineered medical risk features
        ("transform", ColumnTransformer(
            transformers=[
                # Scale continuous features (excluding raw blood pressure, but including engineered ones)
                ("num_scaled", num_pipe,
                    [f for f in CONTINUOUS_FEATURES if f not in ["Systolic_BP", "Diastolic_BP"]] + [
                        "Pulse_Pressure", "MAP"
                    ]
                ),
                # Pass binary features and engineered binary indicators unchanged
                ("num_binary", "passthrough", BINARY_FEATURES + [
                    "High Blood Pressure", "High Cholesterol", "Risk_Score"
                ]),
                # Pass ordinal features unchanged
                ("ordinal", "passthrough", ORDINAL_FEATURES),
            ],
            remainder="drop",                # Drop any columns not listed above
            verbose_feature_names_out=False, # Keep original column names
        ))
    ])

    return preprocessor


def build_preprocessor_not_scaled():
    """
    Builds a preprocessing pipeline for tree-based models (Decision Tree, Random Forest, XGBoost).

    Processing steps
    1. Pass continuous, binary, and ordinal features unchanged.
    2. Add engineered medical risk features from FeatureEngineer.

    Inputs
    None directly; uses feature lists from config.py.

    Outputs
    preprocessor : sklearn.Pipeline
        Pipeline that applies feature engineering without scaling.
    """

    # Build preprocessing pipeline
    preprocessor = Pipeline(steps=[
        ("features", FeatureEngineer()),  # Add engineered medical risk features
        ("transform", ColumnTransformer(
            transformers=[
                # Pass all features unchanged (continuous, binary, ordinal, engineered)
                ("num", "passthrough", CONTINUOUS_FEATURES + BINARY_FEATURES + ORDINAL_FEATURES + [
                    "Pulse_Pressure", "MAP",
                    "High Blood Pressure", "High Cholesterol", "Risk_Score"
                ]),
            ],
            remainder="drop",                # Drop any columns not listed above
            verbose_feature_names_out=False, # Keep original column names
        ))
    ])

    return preprocessor