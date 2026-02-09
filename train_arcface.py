import cv2
import numpy as np
import os
import pickle
from pathlib import Path
from insightface.app import FaceAnalysis

DATASET_PATH = "images"
OUTPUT_FILE = "embeddings.pkl"

app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))  # ctx_id=0 = GPU, use -1 for CPU

embeddings = []
labels = []

dataset_path = Path(DATASET_PATH)
person_dirs = [d for d in dataset_path.iterdir() if d.is_dir()]

print(f"Found {len(person_dirs)} people")

for person_dir in person_dirs:
    label = person_dir.name
    image_paths = list(person_dir.glob("*.jpg")) + list(person_dir.glob("*.png"))

    print(f"Processing {label} ({len(image_paths)} images)")

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        faces = app.get(img)
        if len(faces) == 0:
            continue

        face = max(faces, key=lambda f: f.bbox[2]*f.bbox[3])
        emb = face.embedding

        embeddings.append(emb)
        labels.append(label)

print(f"Total embeddings: {len(embeddings)}")

with open(OUTPUT_FILE, "wb") as f:
    pickle.dump((embeddings, labels), f)

print("Embeddings saved to", OUTPUT_FILE)
