import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image

import os
import numpy as np
from typing import List
from PIL import Image
from sklearn.decomposition import PCA, IncrementalPCA
import joblib

# Global variables to store models and PCA objects
_RESNET_MODEL = None
_PCA_CACHE = {}


def reg2emb_one_image(load_img)->np.ndarray:
    """
    Convert RGB image to embedding using ResNet50 model.
    :param load_img: image that load with keras.preprocessing.image.load_img(path, target_size=(224, 224))
    :return: Array of embeddings with shape (1, 2048)
    """
    model = get_resnet_model()
    x = image.img_to_array(load_img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x, verbose=0)
    return features

def get_resnet_model():
    """
    Returns a cached ResNet50 model to avoid reloading it.
    """
    global _RESNET_MODEL
    if _RESNET_MODEL is None:
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        _RESNET_MODEL = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D()
        ])
    return _RESNET_MODEL


def rgb2emb(img_paths: List[str], batch_size=32) -> np.ndarray:
    """
    Convert RGB images to embeddings using ResNet50 with batch processing.

    :param img_paths: List of paths to images
    :param batch_size: Size of batches to process at once
    :return: Array of embeddings with shape (n_images, 2048)
    """
    model = get_resnet_model()

    all_features = []
    valid_indices = []
    valid_paths = []

    # First check which images exist and can be loaded
    for i, img_path in enumerate(img_paths):
        try:
            if not os.path.exists(img_path):
                print(f"Warning: File not found: {img_path}")
                continue

            valid_indices.append(i)
            valid_paths.append(img_path)
        except Exception as e:
            print(f"Error checking image {img_path}: {e}")

    # Process valid images in batches
    for i in range(0, len(valid_paths), batch_size):
        batch_paths = valid_paths[i:i + batch_size]
        imgs = []

        for img_path in batch_paths:
            try:
                img = image.load_img(img_path, target_size=(224, 224))
                x = image.img_to_array(img)
                imgs.append(x)
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
                # Add a placeholder instead
                imgs.append(np.zeros((224, 224, 3)))

        batch = np.array(imgs)
        batch = preprocess_input(batch)
        features = model.predict(batch, verbose=0)
        all_features.append(features)

    if not all_features:
        return np.array([])

    # Combine all batches
    all_features_array = np.vstack(all_features)

    # Create result array with same length as input, with zeros for failed images
    result = np.zeros((len(img_paths), all_features_array.shape[1]))

    # Fill in the features for valid images
    for idx, valid_idx in enumerate(valid_indices):
        if idx < len(all_features_array):
            result[valid_idx] = all_features_array[idx]

    return result


def grayscale2emb(img_paths: List[str], batch_size=32) -> np.ndarray:
    """
    Convert images to embeddings using ResNet50 with batch processing.
    :param img_paths: List of paths to images
    :param batch_size: Size of batches to process at once
    :return: Array of embeddings with shape (n_images, 2048)
    """
    model = get_resnet_model()

    all_features = []
    valid_indices = []
    valid_paths = []

    # First check which images exist and can be loaded
    for i, img_path in enumerate(img_paths):
        try:
            if not os.path.exists(img_path):
                print(f"Warning: File not found: {img_path}")
                continue

            valid_indices.append(i)
            valid_paths.append(img_path)
        except Exception as e:
            print(f"Error checking image {img_path}: {e}")

    # Process valid images in batches
    for i in range(0, len(valid_paths), batch_size):
        batch_paths = valid_paths[i:i + batch_size]
        imgs = []

        for img_path in batch_paths:
            try:
                # Load image with specified color mode
                img = image.load_img(img_path, target_size=(224, 224), color_mode='grayscale')
                x = image.img_to_array(img)
                x = np.repeat(x, 3, axis=2)  # Duplicate the single channel to all 3 RGB channels

                imgs.append(x)
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
                # Add a placeholder instead
                imgs.append(np.zeros((224, 224, 3)))

        batch = np.array(imgs)
        batch = preprocess_input(batch)
        features = model.predict(batch, verbose=0)
        all_features.append(features)

    if not all_features:
        return np.array([])

    # Combine all batches
    all_features_array = np.vstack(all_features)

    # Create result array with same length as input, with zeros for failed images
    result = np.zeros((len(img_paths), all_features_array.shape[1]))

    # Fill in the features for valid images
    for idx, valid_idx in enumerate(valid_indices):
        if idx < len(all_features_array):
            result[valid_idx] = all_features_array[idx]

    return result

def rgb2flatPCA(img_paths: List[str], n_components: int = 256, img_size=(224, 224)):
    """
    Convert RGB images to single flattened vectors and then reduce the dimensionality using PCA.
    Uses a cached PCA model when available to handle small batches while maintaining n_components=256.

    :param img_paths: List of paths to images.
    :param n_components: Number of PCA components to retain (default is 256).
    :param img_size: Size to resize images to (default is 224x224).
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


def pretrain_pca(image_paths, n_components=256, img_size=(224, 224), batch_size=100):
    """Use IncrementalPCA to handle larger datasets with less memory"""
    print(f"Pre-training Incremental PCA with {len(image_paths)} images...")

    # Ensure n_components is not greater than batch_size
    if n_components > batch_size:
        print(f"Reducing n_components from {n_components} to {batch_size} to match batch size.")
        n_components = batch_size

    ipca = IncrementalPCA(n_components=n_components)

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        print(f"Processing batch {i // batch_size + 1}/{len(image_paths) // batch_size + 1}...")

        flattened_images = []
        for path in batch_paths:
            try:
                img = Image.open(path).convert('RGB')
                img = img.resize(img_size)
                arr = np.array(img)
                flat = arr.flatten()
                flattened_images.append(flat)
            except Exception as e:
                print(f"Error: {e}")

        if flattened_images:
            X = np.stack(flattened_images, axis=0)
            ipca.partial_fit(X)

    # Save to cache and disk
    cache_key = f"rgb_{n_components}_{img_size[0]}x{img_size[1]}"
    _PCA_CACHE[cache_key] = ipca
    pca_file = f"pca_cache_{cache_key}.joblib"
    joblib.dump(ipca, pca_file)

    print(f"Incremental PCA pre-training complete. Saved to {pca_file}")
    return ipca
