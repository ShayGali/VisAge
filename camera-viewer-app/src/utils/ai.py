import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import os

from .img2vec import reg2emb_one_image

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")

MODEL_TO_PATH = {
    "comb_softmax_rgb": os.path.join(MODELS_DIR, "softmax_model_rgb.pkl"),
    "comb_mlp_rgb": os.path.join(MODELS_DIR, "mlp_model_rgb.h5"),
}

MODELS = {}

COMBINED_ENCODER = joblib.load(os.path.join(MODELS_DIR, "combined_encoder.pkl"))
COMBINED_SCALER = joblib.load(os.path.join(MODELS_DIR, "combined_scaler.pkl"))


def load_model(model_name):
    global MODELS

    if model_name in MODELS:
        return MODELS[model_name]

    model_path = MODEL_TO_PATH[model_name]

    if model_path.endswith(".h5"):
        model = tf.keras.models.load_model(model_path)

    else:
        model = joblib.load(model_path)

    MODELS[model_name] = model
    return model


def predict_image(model_name, loaded_img):
    model = load_model(model_name)
    features = reg2emb_one_image(loaded_img)
    features_scaled = COMBINED_SCALER.transform(features)

    is_keras_model = model_name.endswith(".h5")

    if is_keras_model:
        probs = model.predict(features_scaled)[0]
        predicted_class_idx = np.argmax(probs)
        confidence = float(probs[predicted_class_idx])
    else:
        predicted_class_idx = model.predict(features_scaled)[0]

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(features_scaled)[0]
            confidence = float(probs[predicted_class_idx])
        else:
            confidence = None

    predicted_class = COMBINED_ENCODER.classes_[predicted_class_idx]
    age, gender = predicted_class.split("_")
    return {
        "age": age,
        "gender": gender,
        "confidence": confidence,
        "class_index": int(predicted_class_idx),
    }


if __name__ == "__main__":
    image_path = "./mad.jpg"
    load_img = image.load_img(image_path, target_size=(224, 224))
    # Test with different models
    result = predict_image("comb_softmax_rgb", load_img)
    print(
        f"Softmax prediction: age={result['age']}, prediction={result['gender']} (Confidence: {result['confidence']:.2f})"
    )

    # # Display the image with prediction
    img = plt.imread(image_path)
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.title(
        f"Prediction:  age={result['age']}, prediction={result['gender']} (Confidence: {result['confidence']:.2f})"
    )
    plt.axis("off")
    plt.show()
