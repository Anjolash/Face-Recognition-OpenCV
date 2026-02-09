import cv2
import os

"""
Simple script to collect training images from your webcam.
Run this to capture 30-50 photos of each person you want to recognize.
"""

# Configuration
PERSON_NAME = input("Enter person's name: ").strip()
NUM_IMAGES = 50  # Number of images to capture
SAVE_DIR = f"images/{PERSON_NAME.replace(' ', '_').lower()}"

# Create directory
os.makedirs(SAVE_DIR, exist_ok=True)

# Load cascade
try:
    face_cascade = cv2.CascadeClassifier('./cascades/haarcascade_frontalface_alt2.xml')
except:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam!")
    exit(1)

print(f"\n{'=' * 50}")
print(f"COLLECTING TRAINING IMAGES FOR: {PERSON_NAME}")
print(f"{'=' * 50}")
print(f"Target: {NUM_IMAGES} images")
print(f"Saving to: {SAVE_DIR}")
print("\nInstructions:")
print("- Look at the camera")
print("- Press SPACE to capture an image")
print("- Try different angles, expressions, and lighting")
print("- Press 'q' to quit early")
print(f"{'=' * 50}\n")

count = 0

while count < NUM_IMAGES:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Face detected", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show progress
    cv2.putText(frame, f"Images: {count}/{NUM_IMAGES}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame, "Press SPACE to capture", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow('Collect Training Images', frame)

    key = cv2.waitKey(1) & 0xFF

    # Press SPACE to save image
    if key == ord(' '):
        if len(faces) > 0:
            filename = os.path.join(SAVE_DIR, f"{PERSON_NAME}_{count:03d}.jpg")
            cv2.imwrite(filename, frame)
            count += 1
            print(f"✓ Captured image {count}/{NUM_IMAGES}")
        else:
            print("✗ No face detected! Please face the camera.")

    # Press 'q' to quit
    elif key == ord('q'):
        print("\nQuitting early...")
        break

cap.release()
cv2.destroyAllWindows()

print(f"\n{'=' * 50}")
print(f"✓ COLLECTION COMPLETE!")
print(f"{'=' * 50}")
print(f"Captured {count} images for {PERSON_NAME}")
print(f"Saved to: {SAVE_DIR}")
print(f"\nNext steps:")
print(f"1. Repeat this process for other people you want to recognize")
print(f"2. Run 'python train.py' to train the model")
print(f"3. Run 'python recognizer.py' to test recognition")
print(f"{'=' * 50}\n")