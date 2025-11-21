from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    log_loss, roc_auc_score
)

def get_predictions(model, x):
    """Return predictions and predicted probabilities."""
    
    y_pred = model.predict(x)

    # If model has no predict_proba
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(x)[:, 1]
        print("Using predict_proba for probability estimates.")
    else:
        # Convert decision function output into pseudo-probabilities using sigmoid
        from scipy.special import expit
        y_prob = expit(model.decision_function(x))
        print("Using decision_function with sigmoid for probability estimates.")

    return y_pred, y_prob


def compute_metrics(y_true, y_pred, y_prob):
    """Compute accuracy, F1, precision, recall, logloss, and roc-auc."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "logloss": log_loss(y_true, y_prob),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }
