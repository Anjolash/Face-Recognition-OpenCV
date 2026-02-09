"""
Missing Person Search System - Real-time Recognition (ArcFace + Smoothing)
Uses ArcFace embeddings with temporal voting for stable identity
"""

import cv2
import numpy as np
import pickle
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
from collections import deque, Counter

print("=" * 60)
print("MISSING PERSON SEARCH SYSTEM - ARCFACE (SMOOTH)")
print("=" * 60)

# ---------- Load embeddings ----------
with open("embeddings.pkl", "rb") as f:
    known_embeddings, known_labels = pickle.load(f)

known_embeddings = np.array(known_embeddings)
print(f"✓ Loaded {len(known_labels)} identities")

# ---------- Load ArcFace ----------
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))  # use -1 for CPU

# ---------- Webcam ----------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ---------- Params ----------
PROCESS_EVERY_N_FRAMES = 3
THRESHOLD_HIGH = 0.65
THRESHOLD_MED = 0.50

# Smoothing
HISTORY = 7
name_history = deque(maxlen=HISTORY)
conf_history = deque(maxlen=HISTORY)

frame_count = 0
last_predictions = []

print("\nRecognition Active (ArcFace + Smoothing)")
print("Press 'q' to quit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    if frame_count % PROCESS_EVERY_N_FRAMES == 0:
        faces = app.get(frame)
        predictions = []

        for face in faces:
            emb = face.embedding.reshape(1, -1)

            sims = cosine_similarity(emb, known_embeddings)[0]
            best_idx = np.argmax(sims)
            best_score = float(sims[best_idx])
            raw_name = known_labels[best_idx]

            # Thresholding
            if best_score > THRESHOLD_HIGH:
                color = (0, 255, 0)
                status = "MATCH"
                display_name = raw_name
            elif best_score > THRESHOLD_MED:
                color = (0, 255, 255)
                status = "LIKELY"
                display_name = raw_name
            else:
                color = (0, 0, 255)
                status = "UNKNOWN"
                display_name = "Unknown"

            # ---- Temporal smoothing ----
            name_history.append(display_name)
            conf_history.append(best_score)

            final_name = Counter(name_history).most_common(1)[0][0]
            final_conf = sum(conf_history) / len(conf_history)

            x1, y1, x2, y2 = map(int, face.bbox)

            predictions.append({
                "box": (x1, y1, x2, y2),
                "name": final_name,
                "confidence": final_conf,
                "color": color,
                "status": status
            })

        last_predictions = predictions

    # ---------- Draw ----------
    for pred in last_predictions:
        x1, y1, x2, y2 = pred["box"]
        name = pred["name"]
        conf = pred["confidence"]
        color = pred["color"]

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

        label = f"{name} ({conf*100:.1f}%)"
        size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(frame, (x1, y1-35), (x1+size[0]+10, y1), color, -1)
        cv2.putText(frame, label, (x1+5, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.putText(frame, f"Faces Detected: {len(last_predictions)}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.imshow("Missing Person Search System (ArcFace Smooth)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("\n✓ Recognition stopped")
