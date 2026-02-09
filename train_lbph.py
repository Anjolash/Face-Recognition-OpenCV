"""
LBPH Face Recognition Training - Optimized for Large Datasets
Handles 100+ people with 100+ images each efficiently
"""

import cv2
import numpy as np
import os
import pickle
from pathlib import Path
import time

def train_lbph_model(dataset_path='images', model_output='lbph_model.yml',
                     label_output='labels.pkl'):
    """
    Train LBPH face recognition model on large dataset

    Args:
        dataset_path: Path to images folder (organized by person)
        model_output: Output file for trained model
        label_output: Output file for label mappings
    """

    print("=" * 70)
    print("LBPH FACE RECOGNITION TRAINING - LARGE SCALE")
    print("=" * 70)

    # Initialize face detector and recognizer
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=1,           # Default: 1
        neighbors=8,        # Default: 8
        grid_x=8,           # Default: 8
        grid_y=8            # Default: 8
    )

    # Load dataset
    print("\n📂 Loading dataset...")
    start_time = time.time()

    faces = []
    labels = []
    label_names = {}
    current_label = 0

    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        print(f"❌ Error: Dataset path '{dataset_path}' not found!")
        print(f"   Run 'python download_dataset.py' first!")
        return

    # Get all person directories
    person_dirs = [d for d in dataset_path.iterdir() if d.is_dir()]
    total_people = len(person_dirs)

    print(f"   Found {total_people} people in dataset")

    total_images = 0
    faces_detected = 0
    faces_skipped = 0

    # Process each person
    for person_idx, person_dir in enumerate(person_dirs, 1):
        person_name = person_dir.name
        label_names[current_label] = person_name

        person_images = list(person_dir.glob('*.jpg')) + list(person_dir.glob('*.png'))

        print(f"\n[{person_idx}/{total_people}] Processing {person_name}...")
        print(f"   Images found: {len(person_images)}")

        person_faces_count = 0

        for img_path in person_images:
            total_images += 1

            # Read image
            img = cv2.imread(str(img_path))

            if img is None:
                faces_skipped += 1
                continue

            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Detect faces
            detected_faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )

            # Use the largest detected face
            if len(detected_faces) > 0:
                # Sort by area (w*h) and take largest
                detected_faces = sorted(detected_faces, key=lambda x: x[2] * x[3], reverse=True)
                x, y, w, h = detected_faces[0]

                face_roi = gray[y:y+h, x:x+w]

                # Resize to standard size for consistency
                face_roi = cv2.resize(face_roi, (200, 200))

                faces.append(face_roi)
                labels.append(current_label)

                faces_detected += 1
                person_faces_count += 1
            else:
                faces_skipped += 1

            # Progress indicator every 50 images
            if total_images % 50 == 0:
                print(f"   Processed {total_images} images... ({faces_detected} faces detected)")

        print(f"   ✓ Extracted {person_faces_count} faces for {person_name}")
        current_label += 1

    load_time = time.time() - start_time

    print(f"\n{'=' * 70}")
    print("DATASET STATISTICS")
    print("=" * 70)
    print(f"  Total people: {total_people}")
    print(f"  Total images processed: {total_images}")
    print(f"  Faces detected: {faces_detected}")
    print(f"  Faces skipped (no detection): {faces_skipped}")
    print(f"  Average faces per person: {faces_detected // total_people if total_people > 0 else 0}")
    print(f"  Loading time: {load_time:.2f} seconds")

    if faces_detected == 0:
        print("\n❌ No faces detected! Check your dataset.")
        return

    # Train the model
    print(f"\n🧠 Training LBPH model on {faces_detected} faces...")
    train_start = time.time()

    recognizer.train(faces, np.array(labels))

    train_time = time.time() - train_start

    # Save the model
    print(f"\n💾 Saving model...")
    try:
        # Delete old model if exists
        model_path = Path(model_output)
        if model_path.exists():
            print(f"   Removing old model file...")
            model_path.unlink()

        # Save with write() method
        recognizer.write(model_output)

        # Verify file size
        file_size = model_path.stat().st_size
        file_size_mb = file_size / 1024 / 1024
        print(f"   ✓ Model saved: {file_size_mb:.2f} MB")

        if file_size_mb > 500:
            print(f"   ⚠️ WARNING: Model is unusually large ({file_size_mb:.0f} MB)!")
            print(f"   Expected size: 5-50 MB for LBPH")
            print(f"   The file may be corrupted. Please report this issue.")

    except Exception as e:
        print(f"   ❌ Error saving model: {e}")
        return

    # Save label mappings
    try:
        with open(label_output, 'wb') as f:
            pickle.dump(label_names, f)
        print(f"   ✓ Labels saved")
    except Exception as e:
        print(f"   ❌ Error saving labels: {e}")
        return

    print(f"\n{'=' * 70}")
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"  Training time: {train_time:.2f} seconds")
    print(f"  Total time: {load_time + train_time:.2f} seconds")
    print(f"\n  Model saved: {model_output}")
    print(f"  Labels saved: {label_output}")
    print(f"\n{'=' * 70}")
    print("READY FOR RECOGNITION!")
    print("=" * 70)
    print(f"\nNext step:")
    print(f"  Run: python recognize_lbph.py")
    print()

    # Show sample of people in dataset
    print("Sample of people in dataset:")
    sample_names = list(label_names.values())[:10]
    for name in sample_names:
        print(f"  - {name}")
    if len(label_names) > 10:
        print(f"  ... and {len(label_names) - 10} more")
    print()


if __name__ == "__main__":
    train_lbph_model()