---

title: Face Recognition System
emoji: 🎭
colorFrom: purple
colorTo: blue
sdk: docker
app_port: 7860
--------------

# 🎭 Face Recognition System

A multi‑approach face recognition project containing **two independent pipelines**:

1. **Classical Computer Vision (OpenCV + LBPH + LFW dataset)** — offline, real‑time recognition from webcam
2. **Deep Learning (ArcFace / InsightFace Web App)** — high‑accuracy embedding based recognition

This repository demonstrates both traditional and modern face recognition techniques and compares stability, accuracy, and scalability.

---

# 🧠 Part 1 — Real‑Time Face Recognition (OpenCV + LFW)

A lightweight real‑time recognition system trained on the **Labeled Faces in the Wild (LFW)** dataset using Local Binary Pattern Histograms (LBPH).

## Key Features

* Real‑time webcam recognition
* Named celebrities (from LFW dataset)
* Temporal stabilization (no flickering labels)
* Face tracking & identity locking
* Confidence smoothing across frames
* Works fully offline
* Runs on CPU in real time

---

## How It Works

### Training Pipeline

1. Load LFW dataset using `sklearn.datasets.fetch_lfw_people`
2. Filter identities with enough images
3. Preprocess faces

   * grayscale
   * resize
   * histogram equalization
4. Train LBPH recognizer
5. Save:

   * `trainer.yml`
   * `labels.pickle`

### Recognition Pipeline

For each webcam frame:

1. Detect faces (Haar Cascade)
2. Track faces between frames
3. Predict identity (LBPH)
4. Apply temporal smoothing
5. Lock identity for stability
6. Display result

---

## Temporal Stabilization (Important)

The system does NOT rely on single‑frame predictions.

Instead it uses:

* position tracking
* rolling prediction buffer
* majority vote
* confidence averaging
* identity lock timer

This prevents flickering and creates professional‑quality tracking behavior.

---

## Running the Classical System

### Train

```bash
python train_lfw.py
```

### View trained identities

```bash
python view_trained_lfw_celebrities.py
```

### Run recognizer

```bash
python recognize_lfw.py
```

---

# 🚀 Part 2 — Deep Learning Web App (ArcFace)

A production‑style web application powered by **InsightFace ArcFace embeddings**.

Live Demo:
[https://huggingface.co/spaces/Jolaoflagos/face-recognition](https://huggingface.co/spaces/Jolaoflagos/face-recognition)

---

## Features

* Upload image recognition
* Video recognition
* Live webcam detection
* Add new person instantly (no retraining)
* Multi‑face recognition
* Session‑isolated database

---

## Technology Stack

### Backend

* Flask
* InsightFace (ArcFace buffalo_l model)
* OpenCV
* NumPy & SciPy
* Gunicorn

### Frontend

* HTML5 Canvas
* JavaScript
* CSS

### Deployment

* Docker
* Hugging Face Spaces

---

## Recognition Method (Deep Learning)

Instead of classification, ArcFace uses **embedding similarity**:

1. Extract 512‑D face embedding
2. Compare using cosine similarity
3. Match if similarity ≥ threshold

No retraining required when adding people.

---

# 📊 Classical vs Deep Learning

| Feature           | OpenCV LBPH        | ArcFace              |
| ----------------- | ------------------ | -------------------- |
| Speed             | Very fast          | Moderate             |
| Accuracy          | Medium             | Very high            |
| Training required | Yes                | No                   |
| Add new person    | Retrain            | Instant              |
| Offline capable   | Yes                | Yes                  |
| Hardware          | CPU                | CPU/GPU              |
| Stability         | Temporal smoothing | Embedding similarity |

---

# 🧪 Use Cases

* Missing person search
* Attendance systems
* Smart camera tagging
* Security verification
* Research comparison of CV vs Deep Learning

---

# 📁 Project Structure

```
classical/
  train_lfw.py
  recognize_lfw.py
  view_trained_lfw_celebrities.py
  trainer.yml
  labels.pickle

webapp/
  app.py
  templates/
  static/
  embeddings.pkl
  Dockerfile
```

---

# 📝 Resume Highlights

* Built real‑time face recognition system using OpenCV and LBPH
* Implemented temporal identity stabilization and face tracking
* Trained model on LFW dataset with named identities
* Developed production web app using ArcFace embeddings
* Compared classical CV vs deep learning recognition methods

---

# 👤 Author

Anjolaoluwa Dominion Lasekan
GitHub: [https://github.com/Anjolash](https://github.com/Anjolash)

---

# License

MIT License
