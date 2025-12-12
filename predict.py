# Predicting Heart Attacks Using Machine Learning
# A project submitted for the Subject Module Project in Computer Science

# Mads Degn, Julia Lundager, Daniel Holst Pedersen, Jonas Pheiffer, Magnus Stilling Østergaard
# 18/12-25

import pandas as pd
import joblib
from src.config import ARTIFACTS_DIR, DEFAULT_MODEL, TARGET

def load_model(model_name: str = DEFAULT_MODEL):
    """
    Loads a trained model pipeline from the artifacts directory.

    Inputs
    model_name : str, optional
        The name of the model to load. Defaults to DEFAULT_MODEL.

    Processing steps
    1. Construct the path to the saved model file in ARTIFACTS_DIR.
    2. Load the pipeline object using joblib.
    3. Print confirmation of the loaded model path.

    Outputs
    pipeline : sklearn.Pipeline
        The trained model pipeline ready for prediction.
    """
    model_path = ARTIFACTS_DIR / f"{model_name}_model.pkl"
    pipeline = joblib.load(model_path)
    print(f"Loaded model from {model_path}")
    return pipeline


def predict(model_name=DEFAULT_MODEL, input_path=None, data=None):
    """
    Loads a trained model and makes predictions on new patient data.

    Inputs
    model_name : str, optional
        The name of the model to use for prediction. Defaults to DEFAULT_MODEL.
    input_path : str or Path, optional
        Path to a CSV file containing new patient data.
    data : dict, optional
        A dictionary of patient features for prediction. Used if no CSV path is provided.

    Processing steps
    1. Load the trained model pipeline using load_model.
    2. Load input data:
       - If input_path is provided, attempt to read CSV with semicolon delimiter,
         falling back to comma if necessary.
       - If data dictionary is provided, convert it into a DataFrame.
       - Raise an error if neither input_path nor data is provided.
    3. Use the pipeline to predict class labels and probabilities.
    4. Print predictions for each patient, including risk classification
       and probability if available.

    Outputs
    y_pred : numpy.ndarray
        Array of predicted class labels (0 = Low Risk, 1 = High Risk).
    y_proba : numpy.ndarray or None
        Array of predicted probabilities for the positive class, if available.
    """
    pipeline = load_model(model_name)

    # Load input data from CSV or dictionary
    if input_path:
        try:
            df = pd.read_csv(input_path, sep=";")
            if df.shape[1] == 1:  # If file collapsed into one column, retry with comma
                df = pd.read_csv(input_path, sep=",")
        except Exception:
            df = pd.read_csv(input_path, sep=",")
    elif data:
        df = pd.DataFrame([data])
    else:
        raise ValueError("Provide either input_path or data dictionary")

    # Generate predictions and probabilities
    y_pred = pipeline.predict(df)
    y_proba = pipeline.predict_proba(df)[:, 1] if hasattr(pipeline, "predict_proba") else None

    # Print predictions for each patient
    print("\nPredictions:")
    for i, pred in enumerate(y_pred):
        risk = "High Risk" if pred == 1 else "Low Risk"
        prob = f" (probability: {y_proba[i]:.2f})" if y_proba is not None else ""
        print(f"Patient {i+1}: {risk}{prob}")

    return y_pred, y_proba


if __name__ == "__main__":
    # Run prediction from a CSV file when executed directly
    predict(input_path="data/PredictHeartAttackData.csv")