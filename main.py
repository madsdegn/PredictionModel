# Predicting Heart Attacks Using Machine Learning
# A project submitted for the Subject Module Project in Computer Science

# Mads Degn, Julia Lundager, Daniel Holst Pedersen, Jonas Pheiffer, Magnus Stilling Østergaard
# 18/12-25

import argparse
from pathlib import Path

from src.data_cleaning import clean_data
from src.train import train_model
from predict import predict
from src.config import DATA_DIR, TRAIN_FILE, ARTIFACTS_DIR, DEFAULT_MODEL

def main():
    # Create argument parser with description of the project
    parser = argparse.ArgumentParser(description="Heart Attack Risk Prediction Project")

    # Add argument to specify which step to run (clean, train, predict)
    parser.add_argument("--step", choices=["clean", "train", "predict"], default="train",
                        help="Which step to run: clean, train, or predict")

    # Add argument to specify which model to use for training or prediction
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="Model name (log_reg, knn, dt, rf, xgb)")

    # Add argument to specify input CSV file for prediction
    parser.add_argument("--input", type=str, help="Path to input CSV for prediction")

    # Parse the command-line arguments
    args = parser.parse_args()

    # If step is 'clean', run data cleaning on raw dataset and save to TRAIN_FILE
    if args.step == "clean":
        raw_path = DATA_DIR / "HeartAttackData.csv"
        clean_data(raw_path, TRAIN_FILE)

    # If step is 'train', train the selected model
    elif args.step == "train":
        train_model(model_name=args.model)

    # If step is 'predict', run prediction using the selected model and input CSV
    elif args.step == "predict":
        if not args.input:
            raise ValueError("Please provide --input CSV for prediction")
        predict(model_name=args.model, input_path=args.input)

# Run the main function when this script is executed directly
if __name__ == "__main__":
    main()