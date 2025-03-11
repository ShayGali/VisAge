import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image

import numpy as np


# Global variables to store models and PCA objects
_RESNET_MODEL = None

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