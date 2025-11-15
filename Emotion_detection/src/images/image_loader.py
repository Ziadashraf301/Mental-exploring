import cv2
import os
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from src.logger.train_logger import pipeline_logger


def load_image(filepath):
    """
    Load a single image using OpenCV.

    Parameters
    ----------
    filepath : str or Path
        Path to the image file.

    Returns
    -------
    numpy.ndarray or None
        Loaded image array, or None if loading fails.
    """
    img = cv2.imread(str(filepath))
    if img is None:
        pipeline_logger.warning(f"Failed to load image: {filepath}")
    return img


def load_images(path):
    """
    Load all images from a directory using parallel processing.

    Parameters
    ----------
    path : str or Path
        Directory containing image files.

    Returns
    -------
    tuple
        valid_images : numpy.ndarray
            Array of loaded images.
        image_ids : list
            Names of all image files found.
    """
    path = Path(path)
    image_files = list(path.glob("*"))
    image_ids = [f.name for f in image_files]

    pipeline_logger.info(f"Found {len(image_files)} files in {path}")

    # Parallel load
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        images = list(executor.map(load_image, image_files))

    # Keep only successfully loaded images
    valid_images = [img for img in images if img is not None]

    pipeline_logger.info(
        f"Loaded {len(valid_images)} / {len(image_files)} images successfully."
    )

    return np.array(valid_images), image_ids


def load_labels(path):
    """
    Load labels from a CSV using pandas.

    Assumes labels are in the second column.

    Parameters
    ----------
    path : str
        Path to the CSV file.

    Returns
    -------
    numpy.ndarray
        Array of labels from the second column.
    """
    import pandas as pd

    pipeline_logger.info(f"Loading labels from: {path}")

    labels = pd.read_csv(path).values

    pipeline_logger.info(f"Loaded {labels.shape[0]} labels.")

    return labels[:, 1]
