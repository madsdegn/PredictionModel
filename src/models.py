# Predicting Heart Attacks Using Machine Learning
# A project submitted for the Subject Module Project in Computer Science

# Mads Degn, Julia Lundager, Daniel Holst Pedersen, Jonas Pheiffer, Magnus Stilling Østergaard
# 18/12-25

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

from src.feature_engineering import build_preprocessor

def log_reg():
    pre = build_preprocessor()
    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs"
        )
    return Pipeline([("pre", pre), ("model", clf)])

def knn():
    pre = build_preprocessor()
    clf = KNeighborsClassifier(
        n_neighbors=5,
        weights="uniform"
        )
    return Pipeline([("pre", pre), ("model", clf)])

def dt():
    pre = build_preprocessor()
    clf = DecisionTreeClassifier(
        max_depth=6,
        class_weight="balanced",
        random_state=42
        )
    return Pipeline([("pre", pre), ("model", clf)])

def rf():
    pre = build_preprocessor()
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        min_samples_leaf=10,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    return Pipeline([("pre", pre), ("model", clf)])

def xgb():
    pre = build_preprocessor()
    clf = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3
    )
    return Pipeline([("pre", pre), ("model", clf)])

MODEL_REGISTRY = {
    "log_reg": log_reg,
    "knn": knn,
    "dt": dt,
    "rf": rf,
    "xgb": xgb,
}