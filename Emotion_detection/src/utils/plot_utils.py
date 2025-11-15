import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

def plot_metrics(train_m, test_m, save_path=None):
    """Plot accuracy, F1, precision, recall, and log loss."""
    
    metric_names = ["accuracy", "f1", "precision", "recall", "logloss"]
    titles = [
        "Accuracy (%)", 
        "F1 Score (%)", 
        "Precision (%)", 
        "Recall (%)", 
        "Log Loss"
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

    # Create 5 subplots
    fig, axes = plt.subplots(1, 5, figsize=(22, 4))

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


def plot_training_history(history, save_path=None, title_prefix="CNN"):
    """
    Plot training history (accuracy & loss) for TensorFlow/Keras models.

    Parameters
    ----------
    history : tf.keras.callbacks.History
        History object returned from model.fit()
    save_path : str or None
        Where to save the generated plot. If None, only displays.
    title_prefix : str
        Title label prefix for identifying the model.
    """

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Accuracy ---
    axes[0].plot(history.history['accuracy'], label='Train', linewidth=2, marker='o')
    axes[0].plot(history.history['val_accuracy'], label='Validation', linewidth=2, marker='s')
    axes[0].set_title(f'{title_prefix} Training Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # --- Loss ---
    axes[1].plot(history.history['loss'], label='Train', linewidth=2, marker='o')
    axes[1].plot(history.history['val_loss'], label='Validation', linewidth=2, marker='s')
    axes[1].set_title(f'{title_prefix} Training Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save if needed
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.close()
