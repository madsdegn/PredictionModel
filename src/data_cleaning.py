# Predicting Heart Attacks Using Machine Learning
# A project submitted for the Subject Module Project in Computer Science

# Mads Degn, Julia Lundager, Daniel Holst Pedersen, Jonas Pheiffer, Magnus Stilling Østergaard
# 18/12-25

import pandas as pd
from src.config import TARGET, ID_COLUMNS

def clean_data(input_path, output_path=None):
    """
    Cleans the heart attack dataset.
    
    Steps:
    - Drop Patient ID
    - Split blood pressure into systolic/diastolic
    - Normalize target column to 0/1
    - Standardize missing values
    - Save cleaned dataset (optional)
    """
    df = pd.read_csv(input_path)

    # Drop Patient ID
    df.drop(columns=[c for c in ID_COLUMNS if c in df.columns], inplace=True, errors="ignore")

    # Split BP
    if "Blood Pressure" in df.columns:
        bp_split = df["Blood Pressure"].str.split("/", expand=True)
        df["Systolic_BP"] = pd.to_numeric(bp_split[0], errors="coerce")
        df["Diastolic_BP"] = pd.to_numeric(bp_split[1], errors="coerce")
        df.drop(columns=["Blood Pressure"], inplace=True)

    # Normalize target
    if TARGET in df.columns and df[TARGET].dtype == object:
        df[TARGET] = df[TARGET].str.lower().map({"yes": 1, "no": 0})

    # Save if requested
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Cleaned dataset saved as {output_path}")
        print("Preview of cleaned dataset:")
        print(df.head())

    return df