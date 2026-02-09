"""
Missing Person Search System - Real-time Recognition (Improved)
Uses trained CNN model for face identification
"""

import cv2
import numpy as np
from keras.models import load_model
import pickle
import os

print("=" * 60)
print("MISSING PERSON SEARCH SYSTEM - RECOGNITION")
print("=" * 60)

# Load the trained model
model_files = ['best_model.h5', 'facial_recognition_model.h5', 'facial_recognition_model3.h5']
model = None

for model_file in model_files:
    if os.path.exists(model_file):
        print(f"Loading model: {model_file}")
        model = load_model(model_file)
        print("✓ Model loaded successfully")
        break

if model is None:
    print("ERROR: No model found!")
    print("Run train_cnn.py first!")
    exit(1)

# Load label encoder
try:
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    print(f"✓ Loaded {len(label_encoder.classes_)} identities")
except FileNotFoundError:
    print("WARNING: label_encoder.pkl not found!")
    print("Creating dummy labels...")
    # If no encoder, create numbered labels
    num_classes = model.output_shape[1]


    class names =[f"Person_{i}" for i in range(num_classes)]


    class LabelEncoder:
        def __init__(self, classes):
            self.classes_ = classes

        def inverse_transform(self, indices):
            return [self.classes_[i] for i in indices]


    label_encoder = LabelEncoder(class_names)

# Load Haar cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Image dimensions (match training)
image_width = 128
image_height = 128

# Open webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("ERROR: Could not open webcam!")
    exit(1)

print("\n" + "=" * 60)
print("RECOGNITION ACTIVE")
print("=" * 60)
print("Instructions:")
print("  - Green box = High confidence match")
print("  - Yellow box = Medium confidence match")
print("  - Red box = Low confidence/Unknown")
print("  - Press 'q' to quit")
print("=" * 60 + "\n")

# Performance optimization
frame_count = 0
PROCESS_EVERY_N_FRAMES = 3
last_predictions = []

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    frame_count += 1

    # Only process every Nth frame for speed
    if frame_count % PROCESS_EVERY_N_FRAMES == 0:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )

        predictions = []

        for (x, y, w, h) in faces:
            # Extract and preprocess face
            face_img = frame[y:y + h, x:x + w]
            face_img_resized = cv2.resize(face_img, (image_width, image_height))
            face_img_normalized = face_img_resized.astype('float32') / 255.0
            face_img_input = np.expand_dims(face_img_normalized, axis=0)

            # Predict
            prediction = model.predict(face_img_input, verbose=0)
            confidence = np.max(prediction)
            predicted_class = np.argmax(prediction)

            # Get person name
            person_name = label_encoder.inverse_transform([predicted_class])[0]

            # Determine color based on confidence
            if confidence > 0.8:
                color = (0, 255, 0)  # Green - high confidence
                status = "MATCH"
            elif confidence > 0.6:
                color = (0, 255, 255)  # Yellow - medium
                status = "LIKELY"
            else:
                color = (0, 0, 255)  # Red - low confidence
                status = "UNKNOWN"
                person_name = "Unknown"

            predictions.append({
                'box': (x, y, w, h),
                'name': person_name,
                'confidence': confidence,
                'color': color,
                'status': status
            })

        last_predictions = predictions

    # Draw predictions (using last known results for smooth display)
    for pred in last_predictions:
        x, y, w, h = pred['box']
        name = pred['name']
        confidence = pred['confidence']
        color = pred['color']

        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)

        # Draw label background
        label_text = f"{name} ({confidence * 100:.1f}%)"
        label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(frame, (x, y - 35), (x + label_size[0] + 10, y), color, -1)

        # Draw label text
        cv2.putText(frame, label_text, (x + 5, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Display info
    cv2.putText(frame, f"Faces Detected: {len(last_predictions)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Show frame
    cv2.imshow('Missing Person Search System', frame)

    # Print predictions to console
    if frame_count % (PROCESS_EVERY_N_FRAMES * 10) == 0 and last_predictions:
        print("\nCurrent Detections:")
        for pred in last_predictions:
            print(f"  - {pred['name']}: {pred['confidence'] * 100:.1f}% ({pred['status']})")

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("\n✓ Recognition stopped")