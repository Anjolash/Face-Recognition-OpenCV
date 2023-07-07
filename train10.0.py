import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Constants
dataset_path = 'C:/Users/Anjola Lash/Documents/python proj/face rec final proj kpi/images'
image_height = 200  # Define the desired height of the resized images
image_width = 200  # Define the desired width of the resized images

# Load the dataset
images = []
labels = []
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

for person_name in os.listdir(dataset_path):
    person_dir = os.path.join(dataset_path, person_name)
    for image_name in os.listdir(person_dir):
        image_path = os.path.join(person_dir, image_name)
        image = cv2.imread(image_path)
        
        # Apply Viola-Jones face detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # Process detected faces
        for (x, y, w, h) in faces:
            face_image = image[y:y+h, x:x+w]
            face_image = cv2.resize(face_image, (image_width, image_height))  # Resize the face image
            
            images.append(face_image)
            labels.append(person_name)

# Convert the images and labels to numpy arrays
images = np.array(images)
labels = np.array(labels)

# Encode the labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)
print(labels)
original_labels = label_encoder.inverse_transform(labels)
print(original_labels)
#np.savetxt('original_label.txt', original_labels, fmt='%s')

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Normalize the pixel values
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Define the number of classes
num_classes = len(label_encoder.classes_)

# Create the CNN model
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(image_height, image_width, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=40)

# Save the model
model.save('facial_recognition_model3.h5')
