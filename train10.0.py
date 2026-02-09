"""
Missing Person Search System - Training Script (Improved)
Uses CNN for face recognition with data augmentation
"""

import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
import pickle

# Constants
dataset_path = 'images'  # Relative path
image_height = 128  # Smaller = faster (was 200)
image_width = 128
MIN_IMAGES_PER_PERSON = 10  # Require at least 10 images per person

print("=" * 60)
print("MISSING PERSON SEARCH SYSTEM - TRAINING")
print("=" * 60)
print(f"Dataset: {dataset_path}")
print(f"Image size: {image_width}x{image_height}\n")

# Check if dataset exists
if not os.path.exists(dataset_path):
    print(f"ERROR: Dataset path not found: {dataset_path}")
    print("\nCreate folder structure:")
    print("  images/")
    print("    person1/")
    print("      photo1.jpg")
    print("      photo2.jpg")
    print("    person2/")
    print("      photo1.jpg")
    exit(1)

# Load the dataset
images = []
labels = []
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

print("Loading and processing images...")
person_count = {}

for person_name in os.listdir(dataset_path):
    person_dir = os.path.join(dataset_path, person_name)

    if not os.path.isdir(person_dir):
        continue

    person_images = 0

    for image_name in os.listdir(person_dir):
        if not image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        image_path = os.path.join(person_dir, image_name)

        try:
            image = cv2.imread(image_path)

            if image is None:
                continue

            # Apply Viola-Jones face detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            # Process detected faces
            for (x, y, w, h) in faces:
                face_image = image[y:y + h, x:x + w]
                face_image = cv2.resize(face_image, (image_width, image_height))

                images.append(face_image)
                labels.append(person_name)
                person_images += 1
        except Exception as e:
            print(f"  Error processing {image_path}: {e}")

    if person_images > 0:
        person_count[person_name] = person_images

# Filter out people with too few images
filtered_images = []
filtered_labels = []
for img, label in zip(images, labels):
    if person_count[label] >= MIN_IMAGES_PER_PERSON:
        filtered_images.append(img)
        filtered_labels.append(label)

images = np.array(filtered_images)
labels = np.array(filtered_labels)

print(f"\n✓ Loaded {len(images)} face images")
print(f"✓ Found {len(set(labels))} people:\n")
for person, count in sorted(person_count.items(), key=lambda x: x[1], reverse=True):
    if count >= MIN_IMAGES_PER_PERSON:
        print(f"  - {person}: {count} images")

if len(images) == 0:
    print("\nERROR: No images found!")
    print("Use collect_images.py to gather training data")
    exit(1)

# Encode the labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Save label encoder for later use
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)
print(f"\n✓ Saved label encoder")

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    images, labels_encoded,
    test_size=0.2,
    random_state=42,
    stratify=labels_encoded  # Ensure balanced split
)

# Normalize the pixel values
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

print(f"\nTraining set: {len(X_train)} images")
print(f"Testing set: {len(X_test)} images")

# Define the number of classes
num_classes = len(label_encoder.classes_)

# Create IMPROVED CNN model
print(f"\nBuilding CNN model for {num_classes} classes...")

model = Sequential([
    # Block 1
    Conv2D(32, (3, 3), activation='relu', input_shape=(image_height, image_width, 3)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    # Block 2
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    # Block 3
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    # Flatten and Dense layers
    Flatten(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),

    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),

    Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

print("\n✓ Model compiled")
print(f"\nModel summary:")
model.summary()

# Data augmentation for better generalization
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

# Callbacks
early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=10,
    restore_best_weights=True
)

model_checkpoint = ModelCheckpoint(
    'best_model.h5',
    monitor='val_accuracy',
    save_best_only=True
)

# Train the model
print("\n" + "=" * 60)
print("TRAINING MODEL")
print("=" * 60)
print("This may take 5-15 minutes depending on dataset size...\n")

history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    validation_data=(X_test, y_test),
    epochs=50,
    callbacks=[early_stopping, model_checkpoint],
    verbose=1
)

# Evaluate
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

# Save the final model
model.save('facial_recognition_model.h5')

print("\n" + "=" * 60)
print("✓ TRAINING COMPLETE!")
print("=" * 60)
print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")
print(f"\nFiles created:")
print(f"  - facial_recognition_model.h5 (final model)")
print(f"  - best_model.h5 (best model during training)")
print(f"  - label_encoder.pkl (label mappings)")
print(f"\nNext step: Run the recognition script!")
print("=" * 60)