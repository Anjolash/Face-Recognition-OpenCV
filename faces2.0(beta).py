import cv2
import numpy as np
from keras.models import load_model
import os

# Load the Keras model
model = load_model('facial_recognition_model2.h5')

# Load the Haar cascade classifiers
face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('cascades/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('cascades/haarcascade_smile.xml')

# Set the dimensions for resizing the input images
image_width = 200
image_height = 200

# Define the labels array
labels = ['emilia', 'Gal Gadot', 'kit', 'Najwa Nimri', 'nikolaj', 'zoe saldana']  # Replace with your actual labels

cap = cv2.VideoCapture(0)

predictions = []
has_match_above_threshold = False  # Flag to keep track of matches above the threshold

# Specify the directory to store the unrecognized faces
unrecognized_faces_dir = 'unrecognized_faces'

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # Resize the ROI to match the input shape of the model
        roi_resized = cv2.resize(roi_gray, (image_width, image_height), interpolation=cv2.INTER_AREA)

        # Convert grayscale to 3-channel image
        roi_resized = cv2.cvtColor(roi_resized, cv2.COLOR_GRAY2BGR)

        # Add a batch dimension to the ROI
        roi_norm = np.expand_dims(roi_resized, axis=0)

        # Normalize the ROI
        roi_norm = roi_norm.astype('float32') / 255.0

        # Predict the class probabilities for the ROI
        preds = model.predict(roi_norm)
        class_id = np.argmax(preds[0])

        # Map the class ID back to the original label
        class_name = labels[class_id]

        # Get the predicted probability for the class
        probability = preds[0][class_id] * 100

        # Draw the bounding box
        color = (212, 0, 0)  # BGR
        stroke = 4
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)

        # Display the prediction if the probability is above 10%
        if probability > 80:
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = f"{class_name} ({probability:.2f}%)"
            color = (255, 255, 255)
            stroke = 2
            cv2.putText(frame, name, (x, y), font, 1, color, stroke, cv2.LINE_AA)

        # Check if there is a match above the threshold
        if probability > 80:
            has_match_above_threshold = True

        # Append class name and probability to the predictions list
        predictions.append((class_name, probability))

    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# Save the image of unrecognized face if no match above the threshold occurred
if not has_match_above_threshold:
    unrecognized_face_path = os.path.join(unrecognized_faces_dir, f"unrecognized_face_{len(predictions)}.jpg")
    cv2.imwrite(unrecognized_face_path, roi_color)

cap.release()
cv2.destroyAllWindows()

# Print the predictions
for prediction in predictions:
    print(f"Class: {prediction[0]}, Probability: {prediction[1]}")

print("Has match above threshold:", has_match_above_threshold)
