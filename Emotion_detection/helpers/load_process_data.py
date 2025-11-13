import cv2
import os
import numpy as np
import pandas as pd
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# ==================== OPTIMIZED LOADING FUNCTIONS ====================
def load_image(filepath):
    """Helper function to load a single image"""
    img = cv2.imread(str(filepath))
    return img

def load_images(path):
    """Optimized parallel image loading"""
    path = Path(path)
    image_files = list(path.glob('*'))
    images_ids = [f.name for f in image_files]
    
    # Use ThreadPoolExecutor for parallel loading
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        images = list(executor.map(load_image, image_files))
    
    # Filter out None values (failed loads)
    valid_images = [img for img in images if img is not None]
    
    return np.array(valid_images), images_ids

def load_labels(path):
    """Optimized label loading using pandas"""
    import pandas as pd
    
    # pandas is much faster for CSV files
    labels = pd.read_csv(path).values
    
    return labels[:,1]

# ==================== DATA PREPARATION FUNCTIONS ====================

def prepare_data_for_sklearn(images, labels):
    """
    Prepare data for Logistic Regression and MLP (sklearn)
    - Flatten images
    - Normalize to [0, 1]
    - Ensure labels are integers
    """
    # Flatten images: (n_samples, height, width, channels) -> (n_samples, height*width*channels)
    n_samples = images.shape[0]
    X_flat = images.reshape(n_samples, -1)
    
    # Normalize to [0, 1]
    X_normalized = X_flat / 255.0
    
    # Ensure labels are integers
    y = labels.astype(int)
    
    print(f"\n  Sklearn preparation:")
    print(f"    Data shape: {X_normalized.shape}")
    print(f"    Labels shape: {y.shape}")
    print(f"    Data range: [{X_normalized.min():.3f}, {X_normalized.max():.3f}]")
    
    return X_normalized, y

def prepare_data_for_tensorflow(images, labels):
    """
    Prepare data for TensorFlow/Keras CNN
    - Keep image structure (height, width, channels)
    - Normalize to [0, 1]
    - Ensure labels are integers
    """
    # Normalize to [0, 1]
    X_normalized = images / 255.0
    
    # Ensure labels are integers
    y = labels.astype(int)
    
    print(f"\n  TensorFlow preparation:")
    print(f"    Data shape: {X_normalized.shape}")
    print(f"    Labels shape: {y.shape}")
    print(f"    Data range: [{X_normalized.min():.3f}, {X_normalized.max():.3f}]")
    
    return X_normalized, y

# ==================== MAIN LOADING PIPELINE ====================

def load_and_prepare_data(images_path, labels_path, verify=True):
    """
    Complete pipeline to load and prepare data with proper pandas join
    """
    print(f"\n{'='*70}")
    print(f"LOADING DATA FROM: {images_path}")
    print(f"{'='*70}")
    
    # Step 1: Load images
    print(f"\nStep 1: Loading images...")
    images, image_ids = load_images(images_path)
    print(f"   Loaded {len(images)} images")
    
    # Step 2: Load labels
    print(f"\nStep 2: Loading labels...")
    labels = load_labels(labels_path)
    print(f"   Loaded {len(labels)} labels")
    
    # Step 3: Prepare both versions
    print(f"\nStep 3: Preparing data for different models...")
    X_sklearn, y_sklearn = prepare_data_for_sklearn(images, labels)
    X_tf, y_tf = prepare_data_for_tensorflow(images, labels)
    
    return {
        'sklearn': {'X': X_sklearn, 'y': y_sklearn},
        'tensorflow': {'X': X_tf, 'y': y_tf},
        'raw': {'images': images, 'labels': labels}
    }