import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import os
import sys

project_root = os.path.abspath(os.getcwd())
if project_root not in sys.path:
    sys.path.append(project_root)
    
from img2vec import reg2emb_one_image

def predict_image(model_name, image_path):
    """
    Predict age/gender class for a single image using the specified model.
    
    Parameters:
    -----------
    model_name : str
        Name of the model file in the 'models' directory (without path)
    image_path : str
        Path to the image file
    
    Returns:
    --------
    dict
        Dictionary containing prediction class, confidence, and model info
    """
    # Construct full model path
    model_path = os.path.join('models', model_name)
    
    # Load the model
    if model_name.endswith('.h5'):
        # For Keras models
        model = tf.keras.models.load_model(model_path)
    else:
        # For scikit-learn models
        model = joblib.load(model_path)
    
    # Load label encoder and scaler
    combined_encoder = joblib.load(os.path.join('models', 'combined_encoder.pkl'))
    combined_scaler = joblib.load(os.path.join('models', 'combined_scaler.pkl'))
    
    # Extract features using rgb2emb
    img = image.load_img(image_path, target_size=(224, 224))
    features = reg2emb_one_image(img)
    
    # Preprocess features using the scaler
    features_scaled = combined_scaler.transform(features)
    
    # Make prediction
    is_keras_model = model_name.endswith('.h5')
    
    if is_keras_model:
        # For Keras models
        probs = model.predict(features_scaled)[0]
        predicted_class_idx = np.argmax(probs)
        confidence = float(probs[predicted_class_idx])
    else:
        # For sklearn models
        predicted_class_idx = model.predict(features_scaled)[0]
        
        # Get confidence (probability)
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(features_scaled)[0]
            confidence = float(probs[predicted_class_idx])
        else:
            # For models without probability estimates
            confidence = None
            
    # Get the class label
    predicted_class = combined_encoder.classes_[predicted_class_idx]
    age, gender = predicted_class.split('_')
    return {
        'age': age,
        'gender':gender,
        'confidence': confidence,
        'class_index': int(predicted_class_idx)
    }



image_path = './imgs/mad.jpg'
# Test with different models
# result1 = predict_image('softmax_model_rgb.pkl', image_path)
result1 = predict_image('mlp_model_rgb.h5', image_path)
print(f"Softmax prediction: age={result1['age']}, prediction={result1['gender']} (Confidence: {result1['confidence']:.2f})")

# If you have other models:
# result2 = predict_image('random_forest_model_rgb.pkl', image_path)
# result3 = predict_image('cnn_model_rgb.h5', image_path)

# # Display the image with prediction
img = plt.imread(image_path)
plt.figure(figsize=(8, 8))
plt.imshow(img)
plt.title(f"Prediction:  age={result1['age']}, prediction={result1['gender']} (Confidence: {result1['confidence']:.2f})")
plt.axis('off')
plt.show()