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
from src.evaluation import plot_evaluation   # <-- import plotting helper


def train_model(model_name: str = DEFAULT_MODEL):
    # Load dataset
    df = pd.read_csv(TRAIN_FILE)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # Stratified split: train vs temp (val+test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        train_size=TRAIN_SIZE,
        random_state=SEED,
        stratify=y
    )

    # Stratified split: temp into validation and test
    val_ratio = VALIDATION_SIZE / (VALIDATION_SIZE + TEST_SIZE)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        train_size=val_ratio,
        random_state=SEED,
        stratify=y_temp
    )

    # Build model pipeline
    model_fn = MODEL_REGISTRY[model_name]
    pipeline = model_fn()

    # Train
    pipeline.fit(X_train, y_train)

    # Evaluation helper
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

        if hasattr(pipeline, "predict_proba"):
            y_proba = pipeline.predict_proba(X_split)[:, 1]
            metrics[f"{split_name}_ROC_AUC"] = roc_auc_score(y_split, y_proba)
            metrics[f"{split_name}_PR_AUC"] = average_precision_score(y_split, y_proba)
            print(f"ROC-AUC: {metrics[f'{split_name}_ROC_AUC']:.3f}")
            print(f"PR-AUC: {metrics[f'{split_name}_PR_AUC']:.3f}")

        return metrics

    # Collect metrics for all splits
    metrics = {}
    metrics.update(evaluate("Train", X_train, y_train))
    metrics.update(evaluate("Validation", X_val, y_val))
    metrics.update(evaluate("Test", X_test, y_test))

    # Save model
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    model_path = ARTIFACTS_DIR / f"{model_name}_model.pkl"
    joblib.dump(pipeline, model_path)
    print(f"\nModel saved to {model_path}")

    # Generate evaluation plots for the test set
    plot_evaluation(pipeline, X_test, y_test, model_name)

    return metrics


if __name__ == "__main__":
    train_model()