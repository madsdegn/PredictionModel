import pandas as pd
from sklearn.preprocessing import StandardScaler

def clean_data(input_path: str, output_path: str):
    """
    Cleans and preprocesses the heart attack dataset.
    
    Steps:
    - Drop Patient ID
    - Split blood pressure into systolic/diastolic
    - Encode categorical variables
    - Scale continuous features
    - Save cleaned dataset
    """
    df = pd.read_csv(input_path)

    # Drop ID
    df.drop(columns=['Patient ID'], inplace=True)

    # Split BP
    bp_split = df["Blood Pressure"].str.split("/", expand=True)
    df["Systolic_BP"] = pd.to_numeric(bp_split[0], errors="coerce")
    df["Diastolic_BP"] = pd.to_numeric(bp_split[1], errors="coerce")
    df.drop(columns=["Blood Pressure"], inplace=True)

    # Encode categorical
    df["Sex"] = df["Sex"].map({"Female": 0, "Male": 1})
    diet_map = {"Healthy": 2, "Average": 1, "Unhealthy": 0}
    df["Diet"] = df["Diet"].map(diet_map)
    df = pd.get_dummies(df, columns=["Country", "Continent", "Hemisphere"], drop_first=True)

    # Identify numeric cols
    numeric_cols = df.select_dtypes(include=['number']).columns
    binary_cols = [col for col in numeric_cols if set(df[col].dropna().unique()).issubset({0,1})]
    continuous_cols = numeric_cols.difference(binary_cols)

    # Scale continuous
    scaler = StandardScaler()
    df[continuous_cols] = scaler.fit_transform(df[continuous_cols])

    # Save
    df.to_csv(output_path, index=False)
    print(f"Preprocessed dataset saved as {output_path}")
    print("Preview of cleaned dataset:")
    print(df.head())
    return df