import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score

def plot_metrics(train_m, test_m, save_path=None):
    """Plot accuracy, F1, precision, recall, log loss, and ROC AUC."""
    
    metric_names = ["accuracy", "f1", "precision", "recall", "logloss", "roc_auc"]
    titles = [
        "Accuracy (%)", 
        "F1 Score (%)", 
        "Precision (%)", 
        "Recall (%)", 
        "Log Loss",
        "ROC AUC (%)"
    ]

    # Convert to % except logloss
    train_vals = [
        train_m[m] * 100 if m != "logloss" else train_m[m]
        for m in metric_names
    ]
    test_vals = [
        test_m[m] * 100 if m != "logloss" else test_m[m]
        for m in metric_names
    ]

    # Create 6 subplots
    fig, axes = plt.subplots(1, 6, figsize=(22, 4))

    for idx, ax in enumerate(axes):

        ax.bar(["Train", "Test"],
               [train_vals[idx], test_vals[idx]],
               color=["skyblue", "salmon"],
               edgecolor="black",
               alpha=0.7)

        ax.set_title(titles[idx], fontsize=13, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

        # Add annotation values
        for i, v in enumerate([train_vals[idx], test_vals[idx]]):
            offset = 0.02 if metric_names[idx] == "logloss" else 2
            ax.text(i, v + offset, f"{v:.2f}", ha="center")

        # Set y-limits for percentage-based plots
        if metric_names[idx] != "logloss":
            ax.set_ylim(0, 100)

    plt.tight_layout()

    if save_path:
        plt.savefig(f"{save_path}_metrics.png", dpi=150)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, classes=('Negative', 'Positive'),
                          normalize=False, title='Confusion Matrix',
                          cmap='Blues', save_path=None):
    """
    Plot a nicely formatted confusion matrix with optional normalization.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        classes: Class names for axis labels
        normalize: If True, normalize counts by row
        title: Title of the plot
        cmap: Color map for the heatmap
        save_path: If provided, save the image to this path
    """

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Normalize if required
    if normalize:
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        display_matrix = cm_norm
        fmt = ".2f"
    else:
        display_matrix = cm
        fmt = "d"

    # Prepare percentage labels (always useful)
    group_names = ["True Neg", "False Pos", "False Neg", "True Pos"]
    group_percentages = ["{0:.2%}".format(value)
                         for value in cm.flatten() / np.sum(cm)]

    labels = [f"{name}\n{perc}"
              for name, perc in zip(group_names, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)

    # Plotting
    plt.figure(figsize=(6, 6))
    sns.heatmap(
        display_matrix,
        annot=labels,
        fmt="",
        cmap=cmap,
        xticklabels=classes,
        yticklabels=classes,
        cbar=True
    )

    plt.xlabel("Predicted", fontsize=14)
    plt.ylabel("Actual", fontsize=14)
    plt.title(title, fontsize=18, pad=20)
    plt.tight_layout()

    # Save to file if needed
    if save_path:
        plt.savefig(f"{save_path}_confusion_matrix.png", dpi=150)

    plt.close()

def plot_roc_curve(y_true, y_prob, save_path=None):
    """
    Compute and plot ROC curve from true labels and predicted probabilities.

    Args:
        y_true: Ground truth labels
        y_prob: Predicted probabilities for the positive class
        save_path: Optional path to save the figure (without extension)
    """

    # Compute ROC values
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score = roc_auc_score(y_true, y_prob)

    # Plot
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Guessing")

    plt.xlabel("False Positive Rate", fontsize=14)
    plt.ylabel("True Positive Rate", fontsize=14)
    plt.title("ROC Curve", fontsize=18, pad=20)

    plt.legend(loc="lower right", fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # Save for Airflow / MLflow
    if save_path:
        plt.savefig(f"{save_path}_roc_curve.png", dpi=150)

    plt.close()

    return auc_score
