from images.image_loader import load_images, load_labels
from utils.pipeline_logger import pipeline_logger

def prepare_data_for_sklearn(images, labels):
    """
    Prepare image and label data for traditional ML models (sklearn).

    This includes:
    - Flattening images from (H, W, C) → (H*W*C)
    - Normalizing pixel values to [0, 1]
    - Converting labels to integers

    Parameters
    ----------
    images : numpy.ndarray
        Array of images with shape (n_samples, height, width, channels).
    labels : numpy.ndarray
        Array of labels.

    Returns
    -------
    tuple
        X_sklearn : numpy.ndarray
            Flattened and normalized feature matrix.
        y : numpy.ndarray
            Integer label vector.
    """
    pipeline_logger.info("Preparing data for sklearn...")

    n_samples = images.shape[0]
    X_flat = images.reshape(n_samples, -1)

    X_normalized = X_flat / 255.0
    y = labels.astype(int)

    pipeline_logger.info(
        f"Sklearn: X shape={X_normalized.shape}, y shape={y.shape}, "
        f"pixel_range=[{X_normalized.min():.3f}, {X_normalized.max():.3f}]"
    )

    return X_normalized, y


def prepare_data_for_tensorflow(images, labels):
    """
    Prepare image and label data for TensorFlow/Keras CNN models.

    This includes:
    - Keeping image structure (H, W, C)
    - Normalizing to [0, 1]
    - Ensuring labels are integers

    Parameters
    ----------
    images : numpy.ndarray
        Raw image array.
    labels : numpy.ndarray
        Array of labels.

    Returns
    -------
    tuple
        X_tf : numpy.ndarray
            Normalized image data.
        y : numpy.ndarray
            Integer label vector.
    """
    pipeline_logger.info("Preparing data for TensorFlow...")

    X_normalized = images / 255.0
    y = labels.astype(int)

    pipeline_logger.info(
        f"TensorFlow: X shape={X_normalized.shape}, y shape={y.shape}, "
        f"pixel_range=[{X_normalized.min():.3f}, {X_normalized.max():.3f}]"
    )

    return X_normalized, y


# ==================== MAIN LOADING PIPELINE ====================

def load_and_prepare_data(images_path, labels_path, verify=True):
    """
    Complete pipeline to load images, load labels, and prepare datasets
    for multiple ML frameworks (sklearn + TensorFlow).

    Parameters
    ----------
    images_path : str or Path
        Directory containing image files.
    labels_path : str
        Path to CSV file containing labels (second column = label).
    verify : bool, optional
        Whether to validate equal number of images and labels.

    Returns
    -------
    dict
        {
            'sklearn': {'X': X_sklearn, 'y': y_sklearn},
            'tensorflow': {'X': X_tf, 'y': y_tf},
            'raw': {'images': images, 'labels': labels}
        }
    """
    pipeline_logger.info("=" * 70)
    pipeline_logger.info(f"LOADING DATA FROM: {images_path}")
    pipeline_logger.info("=" * 70)

    # Step 1: Load images
    pipeline_logger.info("Step 1: Loading images...")
    images, image_ids = load_images(images_path)
    pipeline_logger.info(f"Loaded {len(images)} images")

    # Step 2: Load labels
    pipeline_logger.info("Step 2: Loading labels...")
    labels = load_labels(labels_path)
    pipeline_logger.info(f"Loaded {len(labels)} labels")

    # Optional validation
    if verify and len(images) != len(labels):
        pipeline_logger.warning(
            f"Mismatch: {len(images)} images vs {len(labels)} labels"
        )

    # Step 3: Prepare transformed data
    pipeline_logger.info("Step 3: Preparing data for ML models...")
    X_sklearn, y_sklearn = prepare_data_for_sklearn(images, labels)
    X_tf, y_tf = prepare_data_for_tensorflow(images, labels)

    pipeline_logger.info("Data preparation completed successfully.")

    return {
        'sklearn': {'X': X_sklearn, 'y': y_sklearn},
        'tensorflow': {'X': X_tf, 'y': y_tf},
        'raw': {'images': images, 'labels': labels}
    }
