# Predicting Heart Attacks Using Machine Learning
# A project submitted for the Subject Module Project in Computer Science

# Mads Degn, Julia Lundager, Daniel Holst Pedersen, Jonas Pheiffer, Magnus Stilling Østergaard
# 18/12-25

import pandas as pd
from pathlib import Path
from src.config import ID_COLUMNS

def clean_data(input, output_path=None):
    """
    Cleans the heart attack dataset to prepare it for feature engineering and model training.

    Parameters
    input : str, Path, or pandas.DataFrame
        Either a path to the raw dataset CSV file or a DataFrame containing patient records.
    output_path : str or Path, optional
        If provided, the cleaned dataset will be saved to this location as a CSV file.

    Processing steps
    1. Read the dataset from the given input path (if a string/Path is provided).
       If a DataFrame is passed directly, make a copy to avoid modifying the original.
    2. Remove patient identifier columns listed in ID_COLUMNS, since they are not predictive.
    3. Encode categorical variables:
       - Sex: Male → 0, Female → 1
       - Hemisphere: Northern Hemisphere → 0, Southern Hemisphere → 1
       - Diet: Unhealthy → 0, Average → 1, Healthy → 2
    4. Drop unused categorical columns such as Country and Continent.
    5. If a combined 'Blood Pressure' column exists, split it into two numeric features:
       Systolic_BP and Diastolic_BP.
    6. Convert the split values into numeric types, handling any non-numeric entries.
    7. Drop the original 'Blood Pressure' column once the new features are created.
    8. If an output path is provided, save the cleaned dataset and print a preview.

    Returns
    df : pandas.DataFrame
        The cleaned dataset with identifiers removed, categorical variables encoded,
        and blood pressure split into systolic and diastolic values.
    """

    # If input is a file path, read the CSV into a DataFrame
    if isinstance(input, (str, Path)):
        df = pd.read_csv(input)
    else:
        # If input is already a DataFrame, make a copy to avoid modifying the original
        df = input.copy()

    # Remove patient ID columns if present (non-predictive identifiers)
    df.drop(columns=[c for c in ID_COLUMNS if c in df.columns],
            inplace=True, errors="ignore")

    # Encode categorical variables into numeric values
    df["Sex"] = df["Sex"].map({"Male": 0, "Female": 1})
    df["Hemisphere"] = df["Hemisphere"].map({"Northern Hemisphere": 0, "Southern Hemisphere": 1})

    # Encode ordinal 'Diet' feature with a natural order
    df["Diet"] = df["Diet"].map({"Unhealthy": 0, "Average": 1, "Healthy": 2})

    # Drop unused categorical columns that are not needed for prediction
    df.drop(columns=["Country"], inplace=True, errors="ignore")
    df.drop(columns=["Continent"], inplace=True, errors="ignore")

    # If 'Blood Pressure' column exists, split into systolic and diastolic values
    if "Blood Pressure" in df.columns:
        bp_split = df["Blood Pressure"].str.split("/", expand=True)
        df["Systolic_BP"] = pd.to_numeric(bp_split[0], errors="coerce")
        df["Diastolic_BP"] = pd.to_numeric(bp_split[1], errors="coerce")
        df.drop(columns=["Blood Pressure"], inplace=True)

    # If an output path is provided, save the cleaned dataset to CSV and show a preview
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Cleaned dataset saved as {output_path}")
        print("Preview of cleaned dataset:")
        print(df.head())

    # Return the cleaned DataFrame for further use in training or prediction
    return df