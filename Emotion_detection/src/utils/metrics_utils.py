import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, log_loss
)

def get_predictions(model, x, model_type):
    """Return predictions and predicted probabilities."""
    
    if model_type == "sklearn":
        y_pred = model.predict(x)
        y_prob = model.predict_proba(x)[:, 1]

    elif model_type == "tensorflow":
        y_prob = model.predict(x, verbose=0).flatten()
        y_pred = (y_prob > 0.5).astype(int)

    else:
        raise ValueError("model_type must be 'sklearn' or 'tensorflow'")

    return y_pred, y_prob


def compute_metrics(y_true, y_pred, y_prob):
    """Compute accuracy, F1 (macro), precision, recall, and log loss."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average="macro"),
        "precision": precision_score(y_true, y_pred, average="macro"),
        "recall": recall_score(y_true, y_pred, average="macro"),
        "logloss": log_loss(y_true, y_prob)
    }