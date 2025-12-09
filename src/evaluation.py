# Predicting Heart Attacks Using Machine Learning
# A project submitted for the Subject Module Project in Computer Science

# Mads Degn, Julia Lundager, Daniel Holst Pedersen, Jonas Pheiffer, Magnus Stilling Østergaard
# 18/12-25

import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay, ConfusionMatrixDisplay

def plot_evaluation(pipeline, X_test, y_test, model_name):
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    # ROC curve
    RocCurveDisplay.from_predictions(y_test, y_proba)
    plt.title(f"ROC Curve - {model_name}")
    plt.show()

    # Precision-Recall curve
    PrecisionRecallDisplay.from_predictions(y_test, y_proba)
    plt.title(f"Precision-Recall Curve - {model_name}")
    plt.show()

    # Confusion matrix
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.show()