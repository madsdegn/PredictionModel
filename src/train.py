# Predicting Heart Attacks Using Machine Learning
# A project submitted for the Subject Module Project in Computer Science

# Mads Degn, Julia Lundager, Daniel Holst Pedersen, Jonas Pheiffer, Magnus Stilling Østergaard
# 18/12-25

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import joblib

from src.config import (
    TRAIN_FILE, TARGET,
    TRAIN_SIZE, VALIDATION_SIZE, TEST_SIZE,
    SEED, DEFAULT_MODEL, ARTIFACTS_DIR
)
from src.models import MODEL_REGISTRY
from src.evaluation import plot_evaluation


def train_model(model_name: str = DEFAULT_MODEL):
    """
    Trains a machine learning model on the heart attack dataset and evaluates its performance.

    Inputs
    model_name : str, optional
        The name of the model to train. Defaults to the value in DEFAULT_MODEL.
        Options include "log_reg", "knn", "dt", "rf", "xgb".

    Processing steps
    1. Load the dataset from TRAIN_FILE and separate features (X) and target (y).
    2. Perform stratified splitting into training, validation, and test sets
       using the proportions defined in TRAIN_SIZE, VALIDATION_SIZE, and TEST_SIZE.
    3. Build the model pipeline from MODEL_REGISTRY. If XGBoost is selected,
       calculate scale_pos_weight to handle class imbalance.
    4. Fit the pipeline on the training data.
    5. Evaluate the model on train, validation, and test sets, computing metrics
       such as accuracy, precision, recall, F1, ROC-AUC, and PR-AUC.
    6. Save the trained model to ARTIFACTS_DIR for later use.
    7. Generate evaluation plots for the test set.

    Outputs
    metrics : dict
        A dictionary containing evaluation metrics for train, validation, and test sets.
    """
    # Load dataset and separate features and target
    df = pd.read_csv(TRAIN_FILE)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # Stratified split into train and temporary (validation + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        train_size=TRAIN_SIZE,
        random_state=SEED,
        stratify=y
    )

    # Stratified split of temporary set into validation and test
    val_ratio = VALIDATION_SIZE / (VALIDATION_SIZE + TEST_SIZE)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        train_size=val_ratio,
        random_state=SEED,
        stratify=y_temp
    )

    # Build model pipeline from registry
    model_fn = MODEL_REGISTRY[model_name]

    if model_name == "xgb":
        # Calculate scale_pos_weight for imbalance handling
        pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
        pipeline = model_fn(scale_pos_weight=pos_weight)
    else:
        pipeline = model_fn()

    # Train the model pipeline
    pipeline.fit(X_train, y_train)

    # Helper function to evaluate model performance on a given split
    def evaluate(split_name, X_split, y_split):
        y_pred = pipeline.predict(X_split)
        print(f"\n--- {split_name} ---")
        print(classification_report(y_split, y_pred, digits=3))

        metrics = {
            f"{split_name}_Accuracy": accuracy_score(y_split, y_pred),
            f"{split_name}_Precision": precision_score(y_split, y_pred, zero_division=0),
            f"{split_name}_Recall": recall_score(y_split, y_pred, zero_division=0),
            f"{split_name}_F1": f1_score(y_split, y_pred, zero_division=0),
        }

        # Compute ROC-AUC and PR-AUC if probabilities are available
        if hasattr(pipeline, "predict_proba"):
            y_proba = pipeline.predict_proba(X_split)[:, 1]
            metrics[f"{split_name}_ROC_AUC"] = roc_auc_score(y_split, y_proba)
            metrics[f"{split_name}_PR_AUC"] = average_precision_score(y_split, y_proba)
            print(f"ROC-AUC: {metrics[f'{split_name}_ROC_AUC']:.3f}")
            print(f"PR-AUC: {metrics[f'{split_name}_PR_AUC']:.3f}")

        return metrics

    # Collect metrics for train, validation, and test splits
    metrics = {}
    metrics.update(evaluate("Train", X_train, y_train))
    metrics.update(evaluate("Validation", X_val, y_val))
    metrics.update(evaluate("Test", X_test, y_test))

    # Save the trained model to artifacts directory
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    model_path = ARTIFACTS_DIR / f"{model_name}_model.pkl"
    joblib.dump(pipeline, model_path)
    print(f"\nModel saved to {model_path}")

    # Generate evaluation plots for the test set
    plot_evaluation(pipeline, X_test, y_test, model_name)

    # Return collected metrics
    return metrics


if __name__ == "__main__":
    # Run training with the default model when executed directly
    train_model()