# Predicting Heart Attacks Using Machine Learning
# A project submitted for the Subject Module Project in Computer Science

# Mads Degn, Julia Lundager, Daniel Holst Pedersen, Jonas Pheiffer, Magnus Stilling Østergaard
# 18/12-25

from pathlib import Path

# Paths
# DATA_DIR points to the folder containing raw and cleaned datasets
# ARTIFACTS_DIR is where trained models and outputs are saved
# TRAIN_FILE is the main dataset used for training
DATA_DIR = Path("data")
ARTIFACTS_DIR = Path("artifacts")
TRAIN_FILE = DATA_DIR / "CleanedHeartAttackData.csv"

# Target and identifier columns
# TARGET is the column we want to predict (binary classification: 0/1)
# ID_COLUMNS are identifiers that should be removed during preprocessing
TARGET = "Heart Attack Risk"
ID_COLUMNS = ["Patient ID"]

# Continuous features
# These are numeric variables measured on a continuous scale
# Systolic_BP and Diastolic_BP are created in data_cleaning.py by splitting the Blood Pressure column
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
    "Systolic_BP",
    "Diastolic_BP",
]

# Binary features
# These are categorical variables encoded as 0/1
# Sex and Hemisphere are encoded in data_cleaning.py
BINARY_FEATURES = [
    "Sex",
    "Hemisphere",
    "Diabetes",
    "Family History",
    "Smoking",
    "Obesity",
    "Alcohol Consumption",
    "Previous Heart Problems",
    "Medication Use",
]

# Ordinal features
# These are categorical variables with a natural order
# Diet is encoded in data_cleaning.py as Unhealthy=0, Average=1, Healthy=2
ORDINAL_FEATURES = [
    "Diet",
]

# Training setup
# SEED ensures reproducibility
# TRAIN_SIZE, VALIDATION_SIZE, and TEST_SIZE define how the dataset is split
SEED = 42
TRAIN_SIZE = 0.6
VALIDATION_SIZE = 0.2
TEST_SIZE = 0.2

# Default model
# Specifies which model to train if none is provided
# Options are: "log_reg", "knn", "dt", "rf", "xgb"
DEFAULT_MODEL = "xgb"