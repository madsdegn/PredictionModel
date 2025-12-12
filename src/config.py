# Predicting Heart Attacks Using Machine Learning
# A project submitted for the Subject Module Project in Computer Science

# Mads Degn, Julia Lundager, Daniel Holst Pedersen, Jonas Pheiffer, Magnus Stilling Østergaard
# 18/12-25

from pathlib import Path

# Paths
# These define where data and model artifacts are stored.
# DATA_DIR points to the folder containing raw and cleaned datasets.
# ARTIFACTS_DIR is where trained models and outputs are saved.
# TRAIN_FILE is the main dataset used for training.
DATA_DIR = Path("data")
ARTIFACTS_DIR = Path("artifacts")
TRAIN_FILE = DATA_DIR / "HeartAttackData.csv"

# Target and identifier columns
# TARGET is the column we want to predict (binary classification: 0/1).
# ID_COLUMNS are identifiers that should be removed during preprocessing.
TARGET = "Heart Attack Risk"
ID_COLUMNS = ["Patient ID"]

# Numeric features
# These are continuous or binary numeric features used in training.
# Includes engineered features created during cleaning (Systolic_BP and Diastolic_BP).
NUMERIC_FEATURES = [
    "Age",
    "Cholesterol",
    "Heart Rate",
    "Diabetes",
    "Family History",
    "Smoking",
    "Obesity",
    "Alcohol Consumption",
    "Exercise Hours Per Week",
    "Previous Heart Problems",
    "Medication Use",
    "Stress Level",
    "Sedentary Hours Per Day",
    "Income",
    "BMI",
    "Triglycerides",
    "Physical Activity Days Per Week",
    "Sleep Hours Per Day",
    "Systolic_BP",     # created in data_cleaning.py
    "Diastolic_BP",    # created in data_cleaning.py
]

# Categorical features
# These are categorical variables that will be encoded during preprocessing.
CATEGORICAL_FEATURES = [
    "Sex",
    "Diet",
    "Country",
    "Continent",
    "Hemisphere",
]

# Training setup
# SEED ensures reproducibility.
# TRAIN_SIZE, VALIDATION_SIZE, and TEST_SIZE define how the dataset is split.
SEED = 42
TRAIN_SIZE = 0.6
VALIDATION_SIZE = 0.2
TEST_SIZE = 0.2

# Default model
# Specifies which model to train if none is provided.
# Options are: "log_reg", "knn", "dt", "rf", "xgb".
DEFAULT_MODEL = "xgb"