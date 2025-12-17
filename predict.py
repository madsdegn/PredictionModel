import pandas as pd
import joblib
from src.config import ARTIFACTS_DIR, DEFAULT_MODEL
from src.data_cleaning import clean_data

def load_model(model_name: str = DEFAULT_MODEL):
    """
    Loads a trained model pipeline from the artifacts directory.

    Parameters
    model_name : str, optional
        Name of the model to load. Defaults to DEFAULT_MODEL.

    Processing steps
    1. Construct the file path for the saved model.
    2. Load the pipeline using joblib.
    3. Print confirmation of the loaded model.

    Returns
    pipeline : sklearn.Pipeline
        The trained model pipeline including preprocessing and classifier.
    """

    # Construct path to the saved model file
    model_path = ARTIFACTS_DIR / f"{model_name}_model.pkl"

    # Load the pipeline from disk
    pipeline = joblib.load(model_path)

    # Print confirmation message
    print(f"Loaded model from {model_path}")

    # Return the loaded pipeline
    return pipeline


def predict(model_name=DEFAULT_MODEL, input_path=None, data=None):
    """
    Generates predictions for new patient data using a trained model.

    Parameters
    model_name : str, optional
        Name of the model to use for prediction. Defaults to DEFAULT_MODEL.
    input_path : str, optional
        Path to a CSV file containing patient data.
    data : dict, optional
        Dictionary of patient features for prediction.

    Processing steps
    1. Load the trained model pipeline.
    2. Read input data from CSV or dictionary.
       - Handle both semicolon and comma separators for CSV files.
    3. Clean the input data using the same cleaning function as training.
    4. Generate predictions and probabilities.
    5. Print results for each patient.

    Returns
    y_pred : numpy.ndarray
        Predicted class labels (0 = no risk, 1 = risk).
    y_proba : numpy.ndarray or None
        Predicted probabilities for risk, if available.
    """

    # Load the trained model pipeline
    pipeline = load_model(model_name)

    # Load input data from CSV file if provided
    if input_path:
        try:
            # First try reading with semicolon separator
            df = pd.read_csv(input_path, sep=";")

            # If file collapsed into one column, retry with comma separator
            if df.shape[1] == 1:
                df = pd.read_csv(input_path, sep=",")
        except Exception:
            # Fallback: read with comma separator
            df = pd.read_csv(input_path, sep=",")

    # If data dictionary is provided, convert to DataFrame
    elif data:
        df = pd.DataFrame([data])

    # If neither input_path nor data is provided, raise an error
    else:
        raise ValueError("Provide either input_path or data dictionary")

    # Clean the data before prediction (same cleaning as training)
    df = clean_data(df)

    # Generate predictions (class labels)
    y_pred = pipeline.predict(df)

    # Generate probabilities if model supports predict_proba
    y_proba = pipeline.predict_proba(df)[:, 1] if hasattr(pipeline, "predict_proba") else None

    # Print predictions for each patient
    print("\nPredictions:")
    for i, pred in enumerate(y_pred):
        risk = "Risk" if pred == 1 else "No Risk"
        prob = f" (probability: {y_proba[i]:.2f})" if y_proba is not None else ""
        print(f"Patient {i+1}: {risk}{prob}")

    # Return predictions and probabilities
    return y_pred, y_proba


if __name__ == "__main__":
    # Run prediction from a CSV file when executed directly
    predict(input_path="data/PredictHeartAttackData.csv")