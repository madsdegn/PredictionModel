# Predicting Heart Attacks Using Machine Learning
# A project submitted for the Subject Module Project in Computer Science

# Mads Degn, Julia Lundager, Daniel Holst Pedersen, Jonas Pheiffer, Magnus Stilling Østergaard
# 18/12-25

import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay, ConfusionMatrixDisplay


def plot_evaluation(pipeline, X_test, y_test, model_name):
    """
    Generates evaluation plots for a trained model on the test dataset.

    Inputs
    pipeline : sklearn.Pipeline
        The trained model pipeline that includes preprocessing and the classifier.
    X_test : pandas.DataFrame or numpy.ndarray
        Test set features used for evaluation.
    y_test : pandas.Series or numpy.ndarray
        True labels for the test set.
    model_name : str
        Name of the model being evaluated, used in plot titles.

    Processing steps
    1. Predict class labels for the test set.
    2. Predict class probabilities for the test set.
    3. Generate and display the ROC curve.
    4. Generate and display the Precision-Recall curve.
    5. Generate and display the confusion matrix.

    Outputs
    None
        The function does not return values. It produces plots displayed
        interactively using matplotlib.
    """
    # Predict class labels and probabilities
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    # Plot ROC curve
    RocCurveDisplay.from_predictions(y_test, y_proba)
    plt.title(f"ROC Curve - {model_name}")
    plt.show()

    # Plot Precision-Recall curve
    PrecisionRecallDisplay.from_predictions(y_test, y_proba)
    plt.title(f"Precision-Recall Curve - {model_name}")
    plt.show()

    # Plot confusion matrix
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.show()