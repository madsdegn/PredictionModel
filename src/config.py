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
TRAIN_FILE = DATA_DIR / "CleanedHeartAttackData.csv"

# Target and identifier columns
# TARGET is the column we want to predict (binary classification: 0/1).
# ID_COLUMNS are identifiers that should be removed during preprocessing.
TARGET = "Heart Attack Risk"
ID_COLUMNS = ["Patient ID"]

CONTINUOUS_FEATURES = [
    "Age",
    "Cholesterol",
    "Heart Rate",
    "Exercise Hours Per Week",
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

BINARY_FEATURES = [
    "Sex",             # encoded in data_cleaning.py
    "Hemisphere",      # encoded in data_cleaning.py
    "Diabetes",
    "Family History",
    "Smoking",
    "Obesity",
    "Alcohol Consumption",
    "Previous Heart Problems",
    "Medication Use",
    #"Country",        # encoded in data_cleaning.py
    #"Continent",      # encoded in data_cleaning.py
]

ORDINAL_FEATURES = [
    "Diet",
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