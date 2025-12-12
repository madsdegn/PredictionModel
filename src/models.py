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

from src.feature_engineering import build_preprocessor_scaled, build_preprocessor_not_scaled


def log_reg():
    """
    Builds a pipeline with logistic regression as the classifier.

    Inputs
    None directly; uses the scaled preprocessing pipeline from feature_engineering.

    Processing steps
    1. Apply preprocessing with scaling to numeric features and encoding to categorical features.
    2. Train a logistic regression model with balanced class weights to handle imbalance.

    Outputs
    pipeline : sklearn.Pipeline
        A pipeline combining preprocessing and logistic regression.
    """
    pre = build_preprocessor_scaled()
    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs"
    )
    return Pipeline([("pre", pre), ("model", clf)])


def knn():
    """
    Builds a pipeline with k-nearest neighbors as the classifier.

    Inputs
    None directly; uses the scaled preprocessing pipeline from feature_engineering.

    Processing steps
    1. Apply preprocessing with scaling to numeric features and encoding to categorical features.
    2. Train a KNN classifier with 5 neighbors and uniform weighting.

    Outputs
    pipeline : sklearn.Pipeline
        A pipeline combining preprocessing and KNN.
    """
    pre = build_preprocessor_scaled()
    clf = KNeighborsClassifier(
        n_neighbors=5,
        weights="uniform"
    )
    return Pipeline([("pre", pre), ("model", clf)])


def dt():
    """
    Builds a pipeline with decision tree as the classifier.

    Inputs
    None directly; uses the non-scaled preprocessing pipeline from feature_engineering.

    Processing steps
    1. Apply preprocessing without scaling to numeric features and encoding to categorical features.
    2. Train a decision tree classifier with maximum depth of 6 and balanced class weights.

    Outputs
    pipeline : sklearn.Pipeline
        A pipeline combining preprocessing and decision tree.
    """
    pre = build_preprocessor_not_scaled()
    clf = DecisionTreeClassifier(
        max_depth=6,
        class_weight="balanced",
        random_state=42
    )
    return Pipeline([("pre", pre), ("model", clf)])


def rf():
    """
    Builds a pipeline with random forest as the classifier.

    Inputs
    None directly; uses the non-scaled preprocessing pipeline from feature_engineering.

    Processing steps
    1. Apply preprocessing without scaling to numeric features and encoding to categorical features.
    2. Train a random forest classifier with 200 trees, maximum depth of 5,
       minimum of 10 samples per leaf, and balanced class weights.

    Outputs
    pipeline : sklearn.Pipeline
        A pipeline combining preprocessing and random forest.
    """
    pre = build_preprocessor_not_scaled()
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


def xgb(scale_pos_weight=None):
    """
    Builds a pipeline with XGBoost as the classifier.

    Inputs
    scale_pos_weight : float, optional
        Weighting factor to handle class imbalance. If not provided, defaults to 1.

    Processing steps
    1. Apply preprocessing without scaling to numeric features and encoding to categorical features.
    2. Train an XGBoost classifier with 300 estimators, learning rate of 0.05,
       maximum depth of 3, and scale_pos_weight for imbalance handling.

    Outputs
    pipeline : sklearn.Pipeline
        A pipeline combining preprocessing and XGBoost.
    """
    pre = build_preprocessor_not_scaled()
    clf = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        scale_pos_weight=scale_pos_weight if scale_pos_weight else 1
    )
    return Pipeline([("pre", pre), ("model", clf)])


# Registry mapping model names to their builder functions
MODEL_REGISTRY = {
    "log_reg": log_reg,
    "knn": knn,
    "dt": dt,
    "rf": rf,
    "xgb": xgb,
}