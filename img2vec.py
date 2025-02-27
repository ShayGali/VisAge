import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image

import os
import numpy as np
from typing import List
from PIL import Image
from sklearn.decomposition import PCA
import joblib

# Global variable to store PCA objects
_PCA_CACHE = {}


def rgb2flatPCA(img_paths: List[str], n_components: int = 256, img_size=(128, 128)):
    """
    Convert RGB images to single flattened vectors and then reduce the dimensionality using PCA.
    Uses a cached PCA model when available to handle small batches while maintaining n_components=256.

    :param img_paths: List of paths to images.
    :param n_components: Number of PCA components to retain (default is 256).
    :param img_size: Size to resize images to (default is 128x128).
    :return: A NumPy array of shape (num_images, n_components) where each row is a reduced feature vector.
    """
    global _PCA_CACHE

    # Unique cache key for this configuration
    cache_key = f"rgb_{n_components}_{img_size[0]}x{img_size[1]}"
    pca_file = f"pca_cache_{cache_key}.joblib"

    flattened_images = []
    for path in img_paths:
        try:
            # Open image and ensure it is RGB
            img = Image.open(path).convert('RGB')
            img = img.resize(img_size)
            # Convert to numpy array and flatten to a 1D vector
            arr = np.array(img)  # shape: (img_size[0], img_size[1], 3)
            flat = arr.flatten()  # shape: (img_size[0]*img_size[1]*3,)
            flattened_images.append(flat)
        except Exception as e:
            print(f"Error processing image {path}: {e}")
            # Add a zero vector as placeholder for failed images
            flattened_images.append(np.zeros(img_size[0] * img_size[1] * 3))

    # Stack all flattened images into a matrix
    X = np.stack(flattened_images, axis=0)

    num_samples, n_features = X.shape
    min_dim = min(num_samples, n_features)

    # Check if we have a cached PCA model
    if cache_key in _PCA_CACHE:
        pca = _PCA_CACHE[cache_key]
        # Use the cached PCA model to transform the data
        X_reduced = pca.transform(X)
    elif os.path.exists(pca_file):
        # Load PCA from file if it exists
        pca = joblib.load(pca_file)
        _PCA_CACHE[cache_key] = pca
        X_reduced = pca.transform(X)
    else:
        # If batch is too small for requested components, handle specially
        if n_components > min_dim:
            print(f"Warning: Batch size {num_samples} too small for {n_components} components.")
            print(f"Will use a partial transformation and pad with zeros.")

            # Fit PCA with maximum possible components for this batch
            actual_components = max(1, min_dim - 1)
            temp_pca = PCA(n_components=actual_components)
            X_temp = temp_pca.fit_transform(X)

            # Create output array of correct size and fill with transformed data
            X_reduced = np.zeros((num_samples, n_components))
            X_reduced[:, :actual_components] = X_temp

            return X_reduced
        else:
            # Normal case: fit PCA and cache it for future use
            pca = PCA(n_components=n_components)
            X_reduced = pca.fit_transform(X)
            _PCA_CACHE[cache_key] = pca
            # Save the PCA model to disk for future runs
            joblib.dump(pca, pca_file)

    return X_reduced


# Helper function to pre-train PCA on a large dataset
def pretrain_pca(image_paths, n_components=256, img_size=(128, 128)):
    """
    Pre-train PCA on a larger dataset to ensure we can use desired n_components.

    :param image_paths: List of paths to images to use for training
    :param n_components: Number of components to use
    :param img_size: Image size to use
    :return: Trained PCA object
    """
    print(f"Pre-training PCA with {len(image_paths)} images...")

    # Use a sufficient sample size to ensure we can extract n_components
    sample_size = min(1000, len(image_paths))
    sample_paths = image_paths[:sample_size]

    # For RGB images
    flattened_images = []
    for i, path in enumerate(sample_paths):
        if i % 100 == 0:
            print(f"Processing image {i}/{sample_size}...")
        try:
            img = Image.open(path).convert('RGB')
            img = img.resize(img_size)
            arr = np.array(img)
            flat = arr.flatten()
            flattened_images.append(flat)
        except Exception as e:
            print(f"Error: {e}")
            flattened_images.append(np.zeros(img_size[0] * img_size[1] * 3))

    X = np.stack(flattened_images, axis=0)

    # Fit PCA
    pca = PCA(n_components=n_components)
    pca.fit(X)

    # Save to cache
    global _PCA_CACHE
    cache_key = f"rgb_{n_components}_{img_size[0]}x{img_size[1]}"
    _PCA_CACHE[cache_key] = pca

    # Save to disk
    pca_file = f"pca_cache_{cache_key}.joblib"
    joblib.dump(pca, pca_file)

    print(f"PCA pre-training complete. Saved to {pca_file}")
    return pca


def gray2emb(img_paths: List[str]) -> np.ndarray:
    """
    Convert images to grayscale and then to embeddings using ResNet50.
    Assumes that the images are in RGB format and have the same size of 128x128
    :param img_paths:
    :return: embeddings matrix of shape (n_images, 2048)
    """
    # Load ResNet50 without the top classification layer:
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    # Add a global average pooling layer to convert features to a vector
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D()
    ])

    imgs = []

    for img_path in img_paths:
        # Preprocess and load your grayscale image:
        img = image.load_img(img_path, target_size=(128, 128), color_mode='grayscale')
        x = image.img_to_array(img)  # shape: (128,128,1)
        # Replicate the grayscale channel three times to simulate an RGB image:
        x = np.repeat(x, 3, axis=-1)  # shape: (128,128,3)
        imgs.append(x)

    batch = np.array(imgs)
    batch = preprocess_input(batch)

    # Get the vector representation:
    feature_vector = model.predict(batch)
    return feature_vector


def rgb2emb(img_paths: List[str]) -> np.ndarray:
    """
    Convert RGB images to embeddings using ResNet50.
    Assumes that the images are already in RGB format and have the same size of 128x128
    :param img_paths:
    :return: embeddings matrix of shape (n_images, 2048)
    """
    # Load ResNet50 without the top classification layer:
    model = ResNet50(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    # Add a global average pooling layer to convert features to a vector
    model = tf.keras.Sequential([
        model,
        tf.keras.layers.GlobalAveragePooling2D()
    ])

    imgs = []

    for img_path in img_paths:
        img = image.load_img(img_path, target_size=(128, 128))
        x = image.img_to_array(img)
        imgs.append(x)

    batch = np.array(imgs)
    batch = preprocess_input(batch)

    # Get the vector representation:
    feature_vector = model.predict(batch)
    return feature_vector
