# main.py

from src.dataCleaning import clean_data
from src.featureEngineering import make_features

def main():
    # Step 1: Clean the raw data
    cleaned_path = "data/HeartAttackDataCleaned.csv"
    clean_data("data/HeartAttackDataRaw.csv", cleaned_path)

    # Step 2: Feature engineering
    features_path = "data/HeartAttackDataFE.csv"
    make_features(cleaned_path, features_path)

    # Step 3: Train model (placeholder)

    # Step 4: Evaluate model (placeholder)

if __name__ == "__main__":
    main()