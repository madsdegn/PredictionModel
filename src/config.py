# Predicting Heart Attacks Using Machine Learning
# A project submitted for the Subject Module Project in Computer Science

# Mads Degn, Julia Lundager, Daniel Holst Pedersen, Jonas Pheiffer, Magnus Stilling Østergaard
# 18/12-25

from pathlib import Path

# Paths
DATA_DIR = Path("data")
ARTIFACTS_DIR = Path("artifacts")
TRAIN_FILE = DATA_DIR / "HeartAttackData.csv"

# Columns
TARGET = "Heart Attack Risk"
ID_COLUMNS = ["Patient ID"]

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

CATEGORICAL_FEATURES = [
    "Sex",
    "Diet",
    "Country",
    "Continent",
    "Hemisphere",
]

# Training setup
SEED = 42
TRAIN_SIZE = 0.6
VALIDATION_SIZE = 0.2
TEST_SIZE = 0.2

# Choose one of: "log_reg", "knn", "dt", "rf", "xgb"
DEFAULT_MODEL = "xgb"