"""
Flask Backend for ArcFace Face Recognition System
SERVER-SIDE SESSION STORAGE VERSION - FIXED

- Base embeddings loaded from embeddings.pkl (NEVER modified)
- Each user session stored server-side (not in cookies)
- Handles large datasets (1000+ embeddings)
- User additions/deletions only affect their session
- Thread-safe model loading
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import pickle
import base64
import os
import uuid
import threading
from pathlib import Path
from insightface.app import FaceAnalysis
from scipy.spatial.distance import cosine
import copy
from datetime import datetime, timedelta

app = Flask(__name__)
CORS(app)

# Serve static files
app.config['UPLOAD_FOLDER'] = 'static'
os.makedirs('static/screenshots', exist_ok=True)

import threading

# ... other imports ...

# Load ArcFace model with thread safety
face_app = None
face_app_lock = threading.Lock()

def get_face_app():
    global face_app

    # Double-check locking pattern
    if face_app is not None:
        return face_app

    with face_app_lock:
        # Check again inside the lock
        if face_app is not None:
            return face_app

        print("Loading ArcFace model...")
        face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        face_app.prepare(ctx_id=-1, det_size=(640, 640))
        print("✓ Model loaded")
        return face_app

# BASE EMBEDDINGS (PERMANENT - NEVER MODIFIED)
BASE_EMBEDDINGS = []
BASE_LABELS = []
BASE_PEOPLE = []

EMBEDDINGS_FILE = "embeddings.pkl"

# SERVER-SIDE SESSION STORAGE (instead of cookies)
USER_SESSIONS = {}
SESSION_TIMEOUT = timedelta(hours=2)

def load_base_embeddings():
    """Load base embeddings from file (permanent dataset)"""
    global BASE_EMBEDDINGS, BASE_LABELS, BASE_PEOPLE

    if os.path.exists(EMBEDDINGS_FILE):
        try:
            with open(EMBEDDINGS_FILE, "rb") as f:
                data = pickle.load(f)

            if isinstance(data, tuple):
                BASE_EMBEDDINGS, BASE_LABELS = data
            else:
                BASE_EMBEDDINGS = data.get('embeddings', [])
                BASE_LABELS = data.get('labels', [])

            BASE_EMBEDDINGS = [np.array(emb) if not isinstance(emb, np.ndarray) else emb
                              for emb in BASE_EMBEDDINGS]

            BASE_PEOPLE = sorted(list(set(BASE_LABELS)))
            print(f"✓ Loaded {len(BASE_EMBEDDINGS)} BASE embeddings for {len(BASE_PEOPLE)} people")
        except Exception as e:
            print(f"Error loading base embeddings: {e}")
            BASE_EMBEDDINGS = []
            BASE_LABELS = []
            BASE_PEOPLE = []
    else:
        print("⚠️  embeddings.pkl not found!")
        BASE_EMBEDDINGS = []
        BASE_LABELS = []
        BASE_PEOPLE = []

load_base_embeddings()

def cleanup_old_sessions():
    """Remove expired sessions"""
    now = datetime.now()
    to_remove = [sid for sid, data in USER_SESSIONS.items()
                 if now - data['last_access'] > SESSION_TIMEOUT]

    for session_id in to_remove:
        del USER_SESSIONS[session_id]

def get_session_id():
    """Get or create session ID"""
    session_id = request.cookies.get('session_id')

    if not session_id or session_id not in USER_SESSIONS:
        session_id = str(uuid.uuid4())
        USER_SESSIONS[session_id] = {
            'embeddings': copy.deepcopy(BASE_EMBEDDINGS),
            'labels': BASE_LABELS.copy(),
            'unique_people': BASE_PEOPLE.copy(),
            'base_count': len(BASE_PEOPLE),
            'last_access': datetime.now()
        }
    else:
        USER_SESSIONS[session_id]['last_access'] = datetime.now()

    if len(USER_SESSIONS) > 100:
        cleanup_old_sessions()

    return session_id

def get_session_data(session_id):
    """Get session-specific embeddings and labels"""
    if session_id not in USER_SESSIONS:
        get_session_id()

    data = USER_SESSIONS[session_id]
    return data['embeddings'], data['labels'], data['unique_people']

def update_session_data(session_id, embeddings, labels, unique_people):
    """Update session data"""
    if session_id in USER_SESSIONS:
        USER_SESSIONS[session_id]['embeddings'] = embeddings
        USER_SESSIONS[session_id]['labels'] = labels
        USER_SESSIONS[session_id]['unique_people'] = unique_people
        USER_SESSIONS[session_id]['last_access'] = datetime.now()

def find_match(face_embedding, embeddings, labels, threshold=0.6):
    """Find best match for face embedding"""
    if len(embeddings) == 0:
        return "Unknown", 0.0

    if not isinstance(face_embedding, np.ndarray):
        face_embedding = np.array(face_embedding)

    distances = []
    for emb in embeddings:
        if not isinstance(emb, np.ndarray):
            emb = np.array(emb)
        try:
            dist = cosine(face_embedding, emb)
            distances.append(dist)
        except:
            distances.append(1.0)

    if not distances:
        return "Unknown", 0.0

    min_distance = min(distances)
    best_idx = distances.index(min_distance)
    similarity = 1 - min_distance

    if similarity >= threshold:
        return labels[best_idx], similarity
    else:
        return "Unknown", similarity

def process_image(image, embeddings, labels):
    """Process image and detect faces"""
    try:
        faces = get_face_app().get(image)  # ✅ Fixed: now calls get_face_app()
    except Exception as e:
        print(f"Error detecting faces: {e}")
        return []

    results = []
    for face in faces:
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox
        embedding = face.embedding
        name, confidence = find_match(embedding, embeddings, labels)

        results.append({
            'bbox': [int(x1), int(y1), int(x2), int(y2)],
            'name': name,
            'confidence': float(confidence)
        })

    return results

@app.route('/')
def index():
    """Main page"""
    session_id = get_session_id()
    response = app.make_response(render_template('index.html'))
    response.set_cookie('session_id', session_id, max_age=7200)
    return response

@app.route('/api/people', methods=['GET'])
def get_people():
    """Get list of people in current session"""
    session_id = get_session_id()
    embeddings, labels, unique_people = get_session_data(session_id)

    base_count = USER_SESSIONS[session_id]['base_count']
    session_added = len(unique_people) - base_count

    response = jsonify({
        'people': unique_people,
        'total_embeddings': len(embeddings),
        'base_count': base_count,
        'session_added': session_added
    })
    response.set_cookie('session_id', session_id, max_age=7200)
    return response

@app.route('/api/recognize_image', methods=['POST'])
def recognize_image():
    """Recognize faces in uploaded image"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    session_id = get_session_id()
    file = request.files['image']
    embeddings, labels, _ = get_session_data(session_id)

    try:
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'error': 'Invalid image'}), 400

        results = process_image(img, embeddings, labels)

        for result in results:
            x1, y1, x2, y2 = result['bbox']
            name = result['name']
            conf = result['confidence']

            color = (0, 255, 0) if conf >= 0.7 else (0, 255, 255) if conf >= 0.5 else (0, 0, 255)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = f"{name} ({conf*100:.1f}%)"
            cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        _, buffer = cv2.imencode('.jpg', img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            'detections': results,
            'annotated_image': f'data:image/jpeg;base64,{img_base64}'
        })
    except Exception as e:
        print(f"Error in recognize_image: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/recognize_video', methods=['POST'])
def recognize_video():
    """Recognize faces in uploaded video"""
    if 'video' not in request.files:
        return jsonify({'error': 'No video provided'}), 400

    session_id = get_session_id()
    file = request.files['video']
    embeddings, labels, _ = get_session_data(session_id)

    temp_path = 'temp_video.mp4'
    file.save(temp_path)

    try:
        cap = cv2.VideoCapture(temp_path)
        all_detections = {}
        frame_count = 0
        process_every_n = 10

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % process_every_n == 0:
                results = process_image(frame, embeddings, labels)

                for result in results:
                    name = result['name']
                    confidence = result['confidence']

                    if name != "Unknown" and confidence >= 0.5:
                        if name not in all_detections:
                            all_detections[name] = {
                                'count': 0,
                                'max_confidence': 0,
                                'screenshot': None
                            }

                        all_detections[name]['count'] += 1

                        if confidence > all_detections[name]['max_confidence']:
                            all_detections[name]['max_confidence'] = confidence

                            x1, y1, x2, y2 = result['bbox']
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                            label = f"{name} ({confidence*100:.1f}%)"
                            cv2.putText(frame, label, (x1, y1-10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                            screenshot_name = f"{name.replace(' ', '_')}_{frame_count}.jpg"
                            screenshot_path = os.path.join('static/screenshots', screenshot_name)
                            cv2.imwrite(screenshot_path, frame)
                            all_detections[name]['screenshot'] = f'/static/screenshots/{screenshot_name}'

            frame_count += 1

        cap.release()

        summary = [
            {
                'name': name,
                'appearances': data['count'],
                'confidence': data['max_confidence'],
                'screenshot': data['screenshot']
            }
            for name, data in all_detections.items()
        ]
        summary.sort(key=lambda x: x['appearances'], reverse=True)

        return jsonify({
            'total_frames': frame_count,
            'processed_frames': frame_count // process_every_n,
            'detections': summary
        })
    except Exception as e:
        print(f"Error in recognize_video: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.route('/api/webcam_frame', methods=['POST'])
def process_webcam_frame():
    """Process single webcam frame"""
    try:
        session_id = get_session_id()
        data = request.get_json()
        embeddings, labels, _ = get_session_data(session_id)

        if 'image' not in data:
            return jsonify({'error': 'No image data'}), 400

        img_data = data['image'].split(',')[1]
        img_bytes = base64.b64decode(img_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        results = process_image(img, embeddings, labels)

        return jsonify({'detections': results})
    except Exception as e:
        print(f"Error in webcam_frame: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/add_person', methods=['POST'])
def add_person():
    """Add person to SESSION ONLY"""
    try:
        session_id = get_session_id()
        data = request.get_json()

        name = data.get('name', '').strip()
        if not name:
            return jsonify({'error': 'Name required'}), 400

        embeddings, labels, unique_people = get_session_data(session_id)

        if name in labels:
            return jsonify({'error': f'{name} already exists'}), 400

        images_data = data.get('images', [data.get('image')]) if 'images' in data else [data.get('image')]

        new_embeddings = []
        for img_data in images_data:
            try:
                if ',' in img_data:
                    img_data = img_data.split(',')[1]
                img_bytes = base64.b64decode(img_data)
                nparr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if img is None:
                    continue

                faces = get_face_app().get(img)  # ✅ Fixed

                if len(faces) > 0:
                    face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
                    embedding = face.embedding
                    if not isinstance(embedding, np.ndarray):
                        embedding = np.array(embedding)
                    new_embeddings.append(embedding)
            except:
                continue

        if len(new_embeddings) == 0:
            return jsonify({'error': 'No face detected'}), 400

        for embedding in new_embeddings:
            embeddings.append(embedding)
            labels.append(name)

        if name not in unique_people:
            unique_people.append(name)
            unique_people.sort()

        update_session_data(session_id, embeddings, labels, unique_people)

        return jsonify({
            'success': True,
            'message': f'Added {name} with {len(new_embeddings)} images!',
            'embeddings_added': len(new_embeddings),
            'is_temporary': True
        })
    except Exception as e:
        print(f"Error in add_person: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/delete_person', methods=['POST'])
def delete_person():
    """Delete person from SESSION ONLY"""
    try:
        session_id = get_session_id()
        data = request.get_json()

        name = data.get('name')
        if not name:
            return jsonify({'error': 'Name required'}), 400

        embeddings, labels, unique_people = get_session_data(session_id)

        if name not in labels:
            return jsonify({'error': f'{name} not found'}), 400

        indices_to_remove = [i for i, label in enumerate(labels) if label == name]

        for i in sorted(indices_to_remove, reverse=True):
            del embeddings[i]
            del labels[i]

        if name in unique_people:
            unique_people.remove(name)

        update_session_data(session_id, embeddings, labels, unique_people)

        return jsonify({
            'success': True,
            'message': f'Removed {name}',
            'total_people': len(unique_people)
        })
    except Exception as e:
        print(f"Error in delete_person: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*70)
    print("ARCFACE FACE RECOGNITION - FIXED VERSION")
    print("="*70)
    print(f"📁 Base embeddings: {len(BASE_EMBEDDINGS)} for {len(BASE_PEOPLE)} people")
    print("\n🌐 Server running at: http://localhost:5000")
    print("="*70 + "\n")

    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, port=port, host='0.0.0.0')