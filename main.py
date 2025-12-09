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
    parser = argparse.ArgumentParser(description="Heart Attack Risk Prediction Project")
    parser.add_argument("--step", choices=["clean", "train", "predict"], default="train",
                        help="Which step to run: clean, train, or predict")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="Model name (log_reg, knn, dt, rf, xgb)")
    parser.add_argument("--input", type=str, help="Path to input CSV for prediction")
    args = parser.parse_args()

    if args.step == "clean":
        raw_path = DATA_DIR / "HeartAttackData.csv"
        clean_data(raw_path, TRAIN_FILE)

    elif args.step == "train":
        train_model(model_name=args.model)

    elif args.step == "predict":
        if not args.input:
            raise ValueError("Please provide --input CSV for prediction")
        predict(model_name=args.model, input_path=args.input)

if __name__ == "__main__":
    main()