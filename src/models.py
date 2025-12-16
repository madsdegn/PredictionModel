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

    Processing steps
    1. Apply preprocessing with scaling to numeric features and encoding to categorical features.
    2. Train a logistic regression model with balanced class weights to handle class imbalance.

    Inputs
    None directly; uses the scaled preprocessing pipeline from feature_engineering.

    Outputs
    pipeline : sklearn.Pipeline
        A pipeline combining preprocessing and logistic regression.
    """

    # Preprocessing pipeline with scaling
    pre = build_preprocessor_scaled()

    # Logistic Regression classifier with balanced class weights
    clf = LogisticRegression(
        max_iter=1000,          # Allow more iterations for convergence
        class_weight="balanced",# Handle class imbalance
        solver="lbfgs"          # Optimization algorithm
    )

    # Combine preprocessing and model into a pipeline
    return Pipeline([("pre", pre), ("model", clf)])


def knn():
    """
    Builds a pipeline with k-nearest neighbors as the classifier.

    Processing steps
    1. Apply preprocessing with scaling to numeric features and encoding to categorical features.
    2. Train a KNN classifier with 5 neighbors and uniform weighting.

    Inputs
    None directly; uses the scaled preprocessing pipeline from feature_engineering.

    Outputs
    pipeline : sklearn.Pipeline
        A pipeline combining preprocessing and KNN.
    """

    # Preprocessing pipeline with scaling
    pre = build_preprocessor_scaled()

    # KNN classifier with 5 neighbors
    clf = KNeighborsClassifier(
        n_neighbors=5,          # Number of neighbors to consider
        weights="uniform"       # Equal weight for all neighbors
    )

    # Combine preprocessing and model into a pipeline
    return Pipeline([("pre", pre), ("model", clf)])


def dt():
    """
    Builds a pipeline with decision tree as the classifier.

    Processing steps
    1. Apply preprocessing without scaling to numeric features and encoding to categorical features.
    2. Train a decision tree classifier with maximum depth of 6 and balanced class weights.

    Inputs
    None directly; uses the non-scaled preprocessing pipeline from feature_engineering.

    Outputs
    pipeline : sklearn.Pipeline
        A pipeline combining preprocessing and decision tree.
    """

    # Preprocessing pipeline without scaling
    pre = build_preprocessor_not_scaled()

    # Decision Tree classifier
    clf = DecisionTreeClassifier(
        max_depth=6,            # Limit depth to prevent overfitting
        class_weight="balanced",# Handle class imbalance
        random_state=42         # Ensure reproducibility
    )

    # Combine preprocessing and model into a pipeline
    return Pipeline([("pre", pre), ("model", clf)])


def rf():
    """
    Builds a pipeline with random forest as the classifier.

    Processing steps
    1. Apply preprocessing without scaling to numeric features and encoding to categorical features.
    2. Train a random forest classifier with 200 trees, maximum depth of 5,
       minimum of 10 samples per leaf, and balanced class weights.

    Inputs
    None directly; uses the non-scaled preprocessing pipeline from feature_engineering.

    Outputs
    pipeline : sklearn.Pipeline
        A pipeline combining preprocessing and random forest.
    """

    # Preprocessing pipeline without scaling
    pre = build_preprocessor_not_scaled()

    # Random Forest classifier
    clf = RandomForestClassifier(
        n_estimators=200,       # Number of trees
        max_depth=5,            # Limit depth
        min_samples_leaf=10,    # Minimum samples per leaf
        max_features="sqrt",    # Use square root of features at each split
        class_weight="balanced",# Handle class imbalance
        random_state=42,        # Ensure reproducibility
        n_jobs=-1               # Use all CPU cores for training
    )

    # Combine preprocessing and model into a pipeline
    return Pipeline([("pre", pre), ("model", clf)])


def xgb(scale_pos_weight=None):
    """
    Builds a pipeline with XGBoost as the classifier.

    Processing steps
    1. Apply preprocessing without scaling to numeric features and encoding to categorical features.
    2. Train an XGBoost classifier with 300 estimators, learning rate of 0.05,
       maximum depth of 3, and scale_pos_weight for imbalance handling.

    Inputs
    scale_pos_weight : float, optional
        Weighting factor to handle class imbalance. If not provided, defaults to 1.

    Outputs
    pipeline : sklearn.Pipeline
        A pipeline combining preprocessing and XGBoost.
    """

    # Preprocessing pipeline without scaling
    pre = build_preprocessor_not_scaled()

    # XGBoost classifier
    clf = XGBClassifier(
        objective="binary:logistic",          # Binary classification
        eval_metric="logloss",                # Evaluation metric
        random_state=42,                      # Ensure reproducibility
        n_estimators=300,                     # Number of boosting rounds
        learning_rate=0.05,                   # Step size shrinkage
        max_depth=3,                          # Limit depth of trees
        scale_pos_weight=scale_pos_weight if scale_pos_weight else 1
    )

    # Combine preprocessing and model into a pipeline
    return Pipeline([("pre", pre), ("model", clf)])


# Registry mapping model names to their builder functions
MODEL_REGISTRY = {
    "log_reg": log_reg,
    "knn": knn,
    "dt": dt,
    "rf": rf,
    "xgb": xgb,
}