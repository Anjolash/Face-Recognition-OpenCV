"""
LBPH Model Evaluation - Test Existing Trained Model
Evaluates the already-trained lbph_model.yml on test images
"""

import cv2
import numpy as np
import pickle
from pathlib import Path
from collections import defaultdict
import time


def evaluate_existing_model(model_path='lbph_model.yml', labels_path='labels.pkl',
                           dataset_path='images', test_percentage=20, confidence_threshold=70):
    """
    Evaluate existing trained LBPH model

    Args:
        model_path: Path to trained model (.yml)
        labels_path: Path to label mappings (.pkl)
        dataset_path: Path to images folder
        test_percentage: Percentage of images to use for testing (20 = last 20% of each person's images)
        confidence_threshold: Confidence threshold for predictions
    """

    print("=" * 70)
    print("LBPH MODEL EVALUATION - USING EXISTING MODEL")
    print("=" * 70)
    print(f"Model: {model_path}")
    print(f"Labels: {labels_path}")
    print(f"Test percentage: {test_percentage}%")
    print(f"Confidence threshold: {confidence_threshold}")
    print()

    # Load trained model
    print("📂 Loading trained model...")

    # Check if model file exists
    if not Path(model_path).exists():
        print(f"❌ Model file not found: {model_path}")
        print(f"   Make sure you've run 'python train_lbph.py' first!")
        print(f"   Current directory: {Path.cwd()}")
        return

    # Check file size
    file_size = Path(model_path).stat().st_size
    print(f"   Model file size: {file_size / 1024 / 1024:.2f} MB")

    if file_size < 1000:
        print(f"❌ Model file seems too small ({file_size} bytes)")
        print(f"   The model may be corrupted. Please retrain:")
        print(f"   python train_lbph.py")
        return

    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(model_path)
        print(f"✓ Model loaded successfully from: {model_path}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print(f"\nThis usually means the model file is corrupted.")
        print(f"Solution: Delete the corrupted file and retrain:")
        print(f"  1. Delete: {model_path}")
        print(f"  2. Run: python train_lbph.py")
        print(f"  3. Run: python evaluate_existing_model.py")
        return

    # Load label mappings
    try:
        with open(labels_path, 'rb') as f:
            label_names = pickle.load(f)
        print(f"✓ Labels loaded: {len(label_names)} people")
    except Exception as e:
        print(f"❌ Error loading labels: {e}")
        return

    # Reverse mapping (name -> label)
    name_to_label = {name: label for label, name in label_names.items()}

    # Initialize face detector
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    # Load test images (last X% of each person's images)
    print(f"\n📂 Loading test images (last {test_percentage}% of each person)...")

    test_faces = []
    test_labels = []
    test_names = []

    dataset_path = Path(dataset_path)
    person_dirs = [d for d in dataset_path.iterdir() if d.is_dir()]

    for person_dir in person_dirs:
        person_name = person_dir.name

        # Skip if person not in trained model
        if person_name not in name_to_label:
            continue

        person_label = name_to_label[person_name]

        # Get all images for this person
        person_images = sorted(list(person_dir.glob('*.jpg')) + list(person_dir.glob('*.png')))

        # Take last X% for testing
        start_idx = int(len(person_images) * (1 - test_percentage / 100))
        test_imgs = person_images[start_idx:]

        # Process test images
        for img_path in test_imgs:
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

            if len(faces) > 0:
                x, y, w, h = faces[0]
                face_roi = gray[y:y+h, x:x+w]
                face_roi = cv2.resize(face_roi, (200, 200))

                test_faces.append(face_roi)
                test_labels.append(person_label)
                test_names.append(person_name)

    total_test = len(test_faces)

    print(f"✓ Test images loaded:")
    print(f"  Total people: {len(label_names)}")
    print(f"  Test images: {total_test}")
    print()

    if total_test == 0:
        print("❌ No test images found!")
        return

    # Test model
    print("🧪 Testing model on test set...")
    test_start = time.time()

    correct = 0
    incorrect = 0
    unknown = 0

    # Per-class metrics
    true_positives = defaultdict(int)
    false_positives = defaultdict(int)
    false_negatives = defaultdict(int)

    predictions = []
    confidences = []

    # Track confusion (what was it confused with)
    confusion_examples = []

    for idx, (face, true_label, true_name) in enumerate(zip(test_faces, test_labels, test_names)):
        predicted_label, confidence = recognizer.predict(face)
        predicted_name = label_names.get(predicted_label, "Unknown")

        predictions.append(predicted_label)
        confidences.append(confidence)

        # Check if prediction is confident enough
        if confidence > confidence_threshold:
            unknown += 1
            false_negatives[true_label] += 1
        elif predicted_label == true_label:
            correct += 1
            true_positives[true_label] += 1
        else:
            incorrect += 1
            false_positives[predicted_label] += 1
            false_negatives[true_label] += 1
            # Store confusion example
            if len(confusion_examples) < 10:  # Keep first 10 confusions
                confusion_examples.append((true_name, predicted_name, confidence))

        # Progress
        if (idx + 1) % 100 == 0:
            print(f"  Tested {idx + 1}/{total_test} images...")

    test_time = time.time() - test_start

    # Calculate metrics
    total_predictions = correct + incorrect + unknown
    accuracy = (correct / total_predictions * 100) if total_predictions > 0 else 0
    precision = (correct / (correct + incorrect) * 100) if (correct + incorrect) > 0 else 0

    avg_confidence_correct = np.mean([100 - c for c in [confidences[i] for i in range(len(predictions)) if predictions[i] == test_labels[i] and confidences[i] <= confidence_threshold]])

    print()
    print("=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    print("\n📊 OVERALL METRICS:")
    print(f"  Accuracy: {accuracy:.2f}%")
    print(f"  Precision: {precision:.2f}%")
    print(f"  Avg Confidence (correct): {avg_confidence_correct:.2f}%")
    print()
    print(f"  Total test images: {total_predictions}")
    print(f"  Correct predictions: {correct} ({correct/total_predictions*100:.1f}%)")
    print(f"  Incorrect predictions: {incorrect} ({incorrect/total_predictions*100:.1f}%)")
    print(f"  Unknown (low confidence): {unknown} ({unknown/total_predictions*100:.1f}%)")
    print()
    print(f"  Test time: {test_time:.2f} seconds")
    print(f"  Speed: {total_predictions/test_time:.1f} predictions/second")

    # Confusion examples
    if confusion_examples:
        print("\n❌ EXAMPLE CONFUSIONS:")
        for true_name, pred_name, conf in confusion_examples:
            print(f"  True: {true_name:25s} → Predicted: {pred_name:25s} (conf: {100-conf:.1f}%)")

    # Per-class performance (top 10 best)
    print("\n🎯 TOP 10 BEST PERFORMING PEOPLE:")

    person_accuracies = []
    for label in sorted(true_positives.keys()):
        tp = true_positives[label]
        fn = false_negatives[label]
        total = tp + fn
        acc = (tp / total * 100) if total > 0 else 0
        person_accuracies.append((label_names[label], acc, tp, total))

    # Sort by accuracy
    person_accuracies.sort(key=lambda x: x[1], reverse=True)

    for i, (name, acc, tp, total) in enumerate(person_accuracies[:10], 1):
        print(f"  {i:2d}. {name:30s} - {acc:5.1f}% ({tp}/{total})")

    # Top 10 worst
    print("\n🎯 TOP 10 WORST PERFORMING PEOPLE:")
    for i, (name, acc, tp, total) in enumerate(person_accuracies[-10:], 1):
        print(f"  {i:2d}. {name:30s} - {acc:5.1f}% ({tp}/{total})")

    # Confidence distribution
    print("\n📈 CONFIDENCE DISTRIBUTION:")
    confident = sum(1 for c in confidences if c < 50)
    good = sum(1 for c in confidences if 50 <= c < confidence_threshold)
    uncertain = sum(1 for c in confidences if c >= confidence_threshold)

    print(f"  Very Confident (< 50): {confident} ({confident/len(confidences)*100:.1f}%)")
    print(f"  Good (50-{confidence_threshold}): {good} ({good/len(confidences)*100:.1f}%)")
    print(f"  Uncertain (> {confidence_threshold}): {uncertain} ({uncertain/len(confidences)*100:.1f}%)")

    print("\n" + "=" * 70)
    print("SUMMARY FOR RESUME")
    print("=" * 70)
    print(f"✓ Model tested on {total_test} test images")
    print(f"✓ Accuracy: {accuracy:.2f}%")
    print(f"✓ Precision: {precision:.2f}%")
    print(f"✓ Testing speed: {total_predictions/test_time:.1f} predictions/second")
    print("=" * 70)
    print()

    # Save results
    with open('evaluation_results.txt', 'w') as f:
        f.write(f"LBPH Model Evaluation Results\n")
        f.write(f"{'=' * 70}\n\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Accuracy: {accuracy:.2f}%\n")
        f.write(f"Precision: {precision:.2f}%\n")
        f.write(f"Total test images: {total_predictions}\n")
        f.write(f"Correct: {correct}\n")
        f.write(f"Incorrect: {incorrect}\n")
        f.write(f"Unknown: {unknown}\n")
        f.write(f"Testing speed: {total_predictions/test_time:.1f} predictions/second\n")
        f.write(f"\nPer-person performance:\n")
        for name, acc, tp, total in person_accuracies:
            f.write(f"  {name}: {acc:.1f}% ({tp}/{total})\n")

    print("✓ Results saved to: evaluation_results.txt")
    print()

    return accuracy, precision


if __name__ == "__main__":
    evaluate_existing_model(
        model_path='lbph_model.yml',
        labels_path='labels.pkl',
        dataset_path='images',
        test_percentage=20,  # Use last 20% of each person's images for testing
        confidence_threshold=70
    )