# Predicting Heart Attacks Using Machine Learning
# A project submitted for the Subject Module Project in Computer Science

# Mads Degn, Julia Lundager, Daniel Holst Pedersen, Jonas Pheiffer, Magnus Stilling Østergaard
# 18/12-25

import pandas as pd
import joblib
from src.config import ARTIFACTS_DIR, DEFAULT_MODEL, TARGET

def load_model(model_name: str = DEFAULT_MODEL):
    model_path = ARTIFACTS_DIR / f"{model_name}_model.pkl"
    pipeline = joblib.load(model_path)
    print(f"Loaded model from {model_path}")
    return pipeline

def predict(model_name=DEFAULT_MODEL, input_path=None, data=None):
    """Load a trained model and make predictions on new data."""
    pipeline = load_model(model_name)

    # Load input data
    if input_path:
        # Auto-detect delimiter: try semicolon first, fall back to comma
        try:
            df = pd.read_csv(input_path, sep=";")
            if df.shape[1] == 1:  # if it still collapsed into one column
                df = pd.read_csv(input_path, sep=",")
        except Exception:
            df = pd.read_csv(input_path, sep=",")
    elif data:
        df = pd.DataFrame([data])
    else:
        raise ValueError("Provide either input_path or data dictionary")

    # Predict
    y_pred = pipeline.predict(df)
    y_proba = pipeline.predict_proba(df)[:, 1] if hasattr(pipeline, "predict_proba") else None

    # Output
    print("\nPredictions:")
    for i, pred in enumerate(y_pred):
        risk = "High Risk" if pred == 1 else "Low Risk"
        prob = f" (probability: {y_proba[i]:.2f})" if y_proba is not None else ""
        print(f"Patient {i+1}: {risk}{prob}")

    return y_pred, y_proba

if __name__ == "__main__":
    # Predict from a CSV of new patients
    predict(input_path="data/PredictHeartAttackData.csv")