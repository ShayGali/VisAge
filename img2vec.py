from typing import List

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA


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


def rgb2flatPCA(img_paths: List[str], n_components: int = 256):
    """
    Convert RGB images to single flattened vectors and then reduce the dimensionality using PCA.
    Assumes that the images are already in RGB format and have the same size of 128x128.

    :param img_paths: List of paths to images.
    :param n_components: Number of PCA components to retain (default is 256).
    :return: A NumPy array of shape (num_images, n_components) where each row is a reduced feature vector.
    """
    flattened_images = []

    for path in img_paths:
        # Open image and ensure it is RGB
        img = Image.open(path).convert('RGB')

        # Convert to numpy array and flatten to a 1D vector
        arr = np.array(img)  # shape: (128, 128, 3)
        flat = arr.flatten()  # shape: (128*128*3,) = (49152,)
        flattened_images.append(flat)

    # Stack all flattened images into a matrix of shape (num_images, 49152)
    X = np.stack(flattened_images, axis=0)

    num_samples, n_features = X.shape
    min_dim = min(num_samples, n_features)
    if n_components > min_dim:
        raise ValueError(f"n_components={n_components} must be between 0 and min(n_samples, n_features)={min_dim}")

    # Fit PCA to reduce the dimensionality
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X)

    return X_reduced


def grayscale2flatPCA(img_paths: List[str], n_components: int = 256):
    """
    Convert images to grayscale, flatten them into vectors, and then reduce the dimensionality using PCA.
    Takes images of any format, converts them to grayscale, and resizes to 128x128 before processing.

    :param img_paths: List of paths to images.
    :param n_components: Number of PCA components to retain (default is 256).
    :return: A NumPy array of shape (num_images, n_components) where each row is a reduced feature vector.
    """
    flattened_images = []

    for path in img_paths:
        # Open image and convert to grayscale
        img = Image.open(path).convert('L')  # 'L' mode is grayscale

        # Convert to numpy array and flatten to a 1D vector
        arr = np.array(img)  # shape: (128, 128)
        flat = arr.flatten()  # shape: (128*128,) = (16384,)
        flattened_images.append(flat)

    # Stack all flattened images into a matrix of shape (num_images, 16384)
    X = np.stack(flattened_images, axis=0)

    # Check if we have enough samples and features for the requested components
    num_samples, n_features = X.shape
    min_dim = min(num_samples, n_features)
    if n_components > min_dim:
        raise ValueError(f"n_components={n_components} must be between 1 and min(n_samples, n_features)={min_dim}")

    # Fit PCA to reduce the dimensionality
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X)

    return X_reduced  # Return only the reduced vectors


def rgb2flat(img_paths: List[str]):
    """
    Convert images to RGB and flatten them into vectors.
    Takes images of any format, converts them to RGB, and resizes to 128x128 before flattening.

    :param img_paths: List of paths to images.
    :return: A NumPy array of shape (num_images, 49152) where each row is a flattened RGB image.
             Each flattened vector contains all R values, followed by all G values, then all B values.
    """
    flattened_images = []

    for path in img_paths:
        # Open image and ensure it is RGB
        img = Image.open(path).convert('RGB')

        # Resize if necessary (enforce 128x128)
        if img.size != (128, 128):
            img = img.resize((128, 128))

        # Convert to numpy array and flatten to a 1D vector
        arr = np.array(img)  # shape: (128, 128, 3)
        flat = arr.flatten()  # shape: (128*128*3,) = (49152,)
        flattened_images.append(flat)

    # Stack all flattened images into a matrix of shape (num_images, 49152)
    X = np.stack(flattened_images, axis=0)

    return X


def grayscale2flat(img_paths: List[str]):
    """
    Convert images to grayscale and flatten them into vectors.
    Takes images of any format, converts them to grayscale, and resizes to 128x128 before flattening.

    :param img_paths: List of paths to images.
    :return: A NumPy array of shape (num_images, 16384) where each row is a flattened grayscale image.
    """
    flattened_images = []

    for path in img_paths:
        # Open image and convert to grayscale
        img = Image.open(path).convert('L')  # 'L' mode is grayscale

        # Convert to numpy array and flatten to a 1D vector
        arr = np.array(img)  # shape: (128, 128)
        flat = arr.flatten()  # shape: (128*128,) = (16384,)
        flattened_images.append(flat)

    # Stack all flattened images into a matrix of shape (num_images, 16384)
    X = np.stack(flattened_images, axis=0)

    return X
