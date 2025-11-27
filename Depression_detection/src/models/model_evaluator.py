from Depression_detection.src.logger.train_logger import get_logger
from Depression_detection.src.utils.metrics_utils import get_predictions, compute_metrics
from Depression_detection.src.utils.plot_utils import plot_confusion_matrix, plot_metrics, plot_roc_curve
from Depression_detection.src.utils.report_utils import print_report
import torch

LOGGER = get_logger()

def get_report(model, x_train, y_train, x_test, y_test, save_path=None):
    """
    Generate evaluation report for both sklearn and TensorFlow models, with plots.

    Parameters:
    -----------
    model : LogisticRegression or navie bayes or SVC
        Trained model to evaluate.
    x_train, y_train : array-like
        Training data and labels.
    x_test, y_test : array-like
        Test data and labels.
    save_path : str or None
        Path to save the plot. If None, plot is shown but not saved.

    Returns:
    --------
    metrics_dict : dict
        Dictionary with train/test accuracy, F1, and log loss (if applicable).
    """

    # 1) predictions
    y_pred_train, prob_train = get_predictions(model, x_train)
    y_pred_test, prob_test = get_predictions(model, x_test)

    # 2) metrics
    train_m = compute_metrics(y_train, y_pred_train, prob_train)
    test_m = compute_metrics(y_test, y_pred_test, prob_test)

    # 3) print
    print_report(train_m, test_m, y_test, y_pred_test)

    # 4) plot
    plot_metrics(train_m, test_m, save_path)
    plot_confusion_matrix(y_test, y_pred_test, classes=('Negative', 'Positive'),
                          normalize=True, title=f'Confusion Matrix', save_path=save_path)
    plot_roc_curve(y_test, prob_test, save_path=save_path)

    # 5) logging
    LOGGER.info(
        f"{type(model).__name__} model - Test Acc: {test_m['accuracy']:.3f},"
        f"F1: {test_m['f1']:.3f}, LogLoss: {test_m['logloss']:.3f}."
        f"Test AUC: {test_m['roc_auc']:.3f}"
    )

    return {
        "train": train_m,
        "test": test_m
    }


def compute_bert_metrics(pred):
    """
    Compute metrics for BERT model during training.
    """
    from sklearn.metrics import (
        accuracy_score,
        precision_recall_fscore_support,
        roc_auc_score
    )
    
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary'
    )
    acc = accuracy_score(labels, preds)

    # AUC-ROC
    probs = torch.softmax(torch.tensor(pred.predictions), dim=1)[:, 1].numpy()
    auc = roc_auc_score(labels, probs)

    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'auc_roc': auc
    }