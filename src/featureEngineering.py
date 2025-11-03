import pandas as pd
import os

def make_features(input_path: str, output_path: str):
    """
    Performs feature engineering on the cleaned heart attack dataset.

    Steps:
    - Load cleaned dataset
    - Separate target column
    - Create derived features (examples included)
    - Save processed dataset
    """
    # Load cleaned dataset
    df = pd.read_csv(input_path)

    # Separate target
    target_col = "Heart Attack Risk"
    y = df[target_col]
    X = df.drop(columns=[target_col])

    # --- Example derived features ---
    if "Systolic_BP" in X.columns and "Diastolic_BP" in X.columns:
        X["BP_Ratio"] = X["Systolic_BP"] / (X["Diastolic_BP"] + 1e-6)

    if "Cholesterol" in X.columns and "Triglycerides" in X.columns:
        X["Chol_Tri_Ratio"] = X["Cholesterol"] / (X["Triglycerides"] + 1e-6)

    if "Stress Level" in X.columns and "Sleep Hours Per Day" in X.columns:
        X["Stress_Sleep_Ratio"] = X["Stress Level"] / (X["Sleep Hours Per Day"] + 1e-6)

    # Recombine features + target
    df_fe = pd.concat([X, y], axis=1)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to CSV
    df_fe.to_csv(output_path, index=False)
    print(f"Feature-engineered dataset saved as {output_path}")

    return df_fe