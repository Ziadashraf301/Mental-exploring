from sklearn.metrics import classification_report

def print_report(train_m, test_m, y_test, y_pred_test):
    """Printable evaluation output for the model with the six metrics."""
    # Accuracy
    print(f"Train Accuracy : {train_m['accuracy']*100:.2f}%")
    print(f"Test Accuracy  : {test_m['accuracy']*100:.2f}%")

    # F1
    print(f"Train F1 Score : {train_m['f1']*100:.2f}%")
    print(f"Test F1 Score  : {test_m['f1']*100:.2f}%")

    # Precision
    print(f"Train Precision: {train_m['precision']*100:.2f}%")
    print(f"Test Precision : {test_m['precision']*100:.2f}%")

    # Recall
    print(f"Train Recall   : {train_m['recall']*100:.2f}%")
    print(f"Test Recall    : {test_m['recall']*100:.2f}%")

    # Log Loss
    print(f"Train Log Loss : {train_m['logloss']:.4f}")
    print(f"Test Log Loss  : {test_m['logloss']:.4f}")

    # Auc-ROC
    print(f"Train ROC-AUC  : {train_m['roc_auc']*100:.2f}%")
    print(f"Test ROC-AUC   : {test_m['roc_auc']*100:.2f}%")

    # Classification Report
    print("Classification Report:")
    print(classification_report(
        y_test, 
        y_pred_test, 
        target_names=['Negative', 'Positive'])
    )
