# Predicting Heart Attacks Using Machine Learning
# A project submitted for the Subject Module Project in Computer Science

# Mads Degn, Julia Lundager, Daniel Holst Pedersen, Jonas Pheiffer, Magnus Stilling Østergaard
# 18/12-25

import pandas as pd
from src.config import TARGET, ID_COLUMNS

def clean_data(input_path, output_path=None):
    """
    Cleans the heart attack dataset to prepare it for feature engineering and model training.

    Inputs
    input_path : str or Path
        Path to the raw dataset CSV file that contains patient records.
    output_path : str or Path, optional
        If provided, the cleaned dataset will be saved to this location.

    Processing steps
    1. Read the dataset from the given input path.
    2. Remove patient identifiers listed in ID_COLUMNS, since they are not predictive.
    3. If a combined 'Blood Pressure' column exists, split it into two numeric features:
       Systolic_BP and Diastolic_BP.
    4. Convert the split values into numeric types, handling any non-numeric entries.
    5. Drop the original 'Blood Pressure' column once the new features are created.
    6. If an output path is provided, save the cleaned dataset and print a preview.

    Outputs
    df : pandas.DataFrame
        The cleaned dataset with identifiers removed and blood pressure split into
        systolic and diastolic values. Returned for further processing.
    """
    # Read the raw dataset from the input path
    df = pd.read_csv(input_path)

    # Remove patient ID columns if present
    df.drop(columns=[c for c in ID_COLUMNS if c in df.columns],
            inplace=True, errors="ignore")

    # Split blood pressure into systolic and diastolic values
    if "Blood Pressure" in df.columns:
        bp_split = df["Blood Pressure"].str.split("/", expand=True)
        df["Systolic_BP"] = pd.to_numeric(bp_split[0], errors="coerce")
        df["Diastolic_BP"] = pd.to_numeric(bp_split[1], errors="coerce")
        df.drop(columns=["Blood Pressure"], inplace=True)

    # Save cleaned dataset if an output path is provided
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Cleaned dataset saved as {output_path}")
        print("Preview of cleaned dataset:")
        print(df.head())

    # Return the cleaned dataframe for further use
    return df