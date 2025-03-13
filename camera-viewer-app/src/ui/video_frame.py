import cv2
import os
import numpy as np
import time
from tkinter import Frame, Label
from PIL import Image, ImageTk
import sys

# Add parent directory to path to import from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.ai import load_model, predict_image
from tensorflow.keras.preprocessing import image as keras_image

model_dir_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")

class VideoFrame(Frame):
    def __init__(self, parent, video_source=0):
        super().__init__(parent)
        
        self.video_source = video_source
        self.vid = None
        
        self.frame_count = 0
        self.process_every_n_frames = 5  # Process only every 3rd frame
        
        # Store the detected faces and predictions
        self.detected_faces = []  # List of (x, y, w, h, age, gender, confidence) tuples
        self.last_detection_time = 0
        self.detection_timeout = 2.0  # Clear detections after 2 seconds of no updates
   
        # Create a label for displaying the video (full screen)
        self.label = Label(self)
        self.label.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Load the face cascade
        cascade_path = os.path.join(model_dir_path, 'haarcascade_frontalface_default.xml')
        if not os.path.exists(cascade_path):
            print(f"Warning: Face cascade file not found at {cascade_path}")
            print("Face detection will not work.")
            self.face_cascade = None
        else:
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # Load machine learning models for age and gender prediction
        self.load_ml_models()
        
        # Open the video source
        self.open_video_source()
        
        # Start updating the video frames
        self.update()
    
    def load_ml_models(self):
        """Load all required ML models and encoders once at initialization"""
        try:
            # Using the model from ai.py, just verify it can be loaded
            _ = load_model('comb_softmax_rgb')
            self.models_loaded = True
            print("ML model 'comb_softmax_rgb' loaded successfully")
        except Exception as e:
            self.models_loaded = False
            print(f"Error loading ML model: {str(e)}")
    
    def predict_from_frame(self, frame):
        """
        Predict age and gender from a frame (OpenCV image)
        
        Returns:
            tuple: (age, gender, confidence)
        """
        if not self.models_loaded:
            return None, None, 0
        
        try:
            # Resize frame if it's too large (faster processing)
            h, w = frame.shape[:2]
            if w > 224:  # Only resize if larger than needed
                frame = cv2.resize(frame, (224, 224))
                
            # Convert OpenCV BGR frame to RGB (PIL uses RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Create PIL Image from numpy array
            pil_img = Image.fromarray(rgb_frame)
            
            # Resize to the target size expected by the model
            pil_img = pil_img.resize((224, 224), Image.LANCZOS)
            
            # Use predict_image from ai.py
            result = predict_image('comb_softmax_rgb', pil_img)
            
            # Extract results
            age = result['age']
            gender = result['gender']
            confidence = result['confidence'] if result['confidence'] is not None else 0.0
            
            return age, gender, confidence
            
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            return None, None, 0
    
    def open_video_source(self):
        try:
            self.vid = cv2.VideoCapture(self.video_source)
            if not self.vid.isOpened():
                raise ValueError(f"Unable to open video source {self.video_source}")
        except Exception as e:
            print(f"Error opening video source: {e}")
            self.vid = None
    
    def update(self):
        # Update the video frame
        if self.vid and self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                # Process frame for face detection
                self.frame_count += 1
                if self.frame_count % self.process_every_n_frames == 0 and self.face_cascade is not None:
                    # Run actual detection on this frame
                    self.detect_faces(frame)
                    self.last_detection_time = time.time()
                else:
                    # Apply stored detections to this frame without running detection
                    self.apply_stored_detections(frame)
                
                # Convert the frame from OpenCV BGR format to RGB for display
                self.photo = self.convert_frame_to_photo(frame)
                self.label.config(image=self.photo)
                    
                    # Clear old detections if we haven't updated in a while
                current_time = time.time()
                if current_time - self.last_detection_time > self.detection_timeout:
                    self.detected_faces = []
        
        # Call update again after 15 milliseconds
        self.after(15, self.update)
    
    def detect_faces(self, frame):
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Reduce resolution for face detection (much faster)
        height, width = gray.shape
        if width > 640:  # Don't process huge frames at full resolution
            scale = 640 / width
            small_gray = cv2.resize(gray, (int(width * scale), int(height * scale)))
            scale_factor = 1/scale
        else:
            small_gray = gray
            scale_factor = 1.0
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            small_gray,
            scaleFactor=1.2,
            minNeighbors=4,
            minSize=(30, 30)
        )

        # Scale coordinates back to original size if needed
        if scale_factor != 1.0:
            faces = [(int(x * scale_factor), int(y * scale_factor), 
                 int(w * scale_factor), int(h * scale_factor)) for x, y, w, h in faces]
        
        # Clear previous detections
        self.detected_faces = []
        
        # Draw rectangles around detected faces
        for (x, y, w, h) in faces:
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            age, gender, confidence = None, None, 0
            
            if self.models_loaded:
                # Extract the face for prediction
                face_img = frame[y:y+h, x:x+w].copy()
                
                # Ensure the face is large enough for prediction
                if face_img.shape[0] > 20 and face_img.shape[1] > 20:
                    # Predict age and gender
                    age, gender, confidence = self.predict_from_frame(face_img)
                    
                    if age is not None and gender is not None:
                        # Format the text with confidence score
                        prediction_text = f"Age: {age}, Gender: {gender}"
                        conf_text = f"Confidence: {confidence:.2f}"
                        
                        # Display predictions above the face
                        cv2.putText(frame, prediction_text, (x, y-25), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 4)
                        # Then draw white text on top
                        cv2.putText(frame, prediction_text, (x, y-25), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
                        cv2.putText(frame, conf_text, (x, y-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 4)
                        cv2.putText(frame, conf_text, (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Store the detection result for future frames
            self.detected_faces.append((x, y, w, h, age, gender, confidence))
    
    def apply_stored_detections(self, frame):
        """Apply stored face detections to the current frame"""
        for (x, y, w, h, age, gender, confidence) in self.detected_faces:
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Display the stored prediction if available
            if age is not None and gender is not None:
                # Format the text with confidence score
                prediction_text = f"Age: {age}, Gender: {gender}"
                conf_text = f"Confidence: {confidence:.2f}"
                
                # Display predictions above the face
                cv2.putText(frame, prediction_text, (x, y-25), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 4)
                # Then draw white text on top
                cv2.putText(frame, prediction_text, (x, y-25), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                cv2.putText(frame, conf_text, (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 4)
                cv2.putText(frame, conf_text, (x, y-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    def convert_frame_to_photo(self, frame):
        # Convert the frame from BGR to RGB format
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert the frame to PIL image
        image = Image.fromarray(frame)
        
        # Convert the PIL image to PhotoImage
        return ImageTk.PhotoImage(image=image)
    
    def __del__(self):
        # Release the video source when the object is destroyed
        if hasattr(self, 'vid') and self.vid and self.vid.isOpened():
            self.vid.release()