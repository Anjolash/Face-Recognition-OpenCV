"""
Flask Backend for ArcFace Face Recognition System
SERVER-SIDE SESSION STORAGE VERSION

- Base embeddings loaded from embeddings.pkl (NEVER modified)
- Each user session stored server-side (not in cookies)
- Handles large datasets (1000+ embeddings)
- User additions/deletions only affect their session
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import pickle
import base64
import os
import uuid
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

# Load ArcFace model
pface_app = None

def get_face_app():
    global face_app
    if face_app is None:
        print("Loading ArcFace model...")
        from insightface.app import FaceAnalysis
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
# Each session_id maps to user's data
USER_SESSIONS = {}
SESSION_TIMEOUT = timedelta(hours=2)  # Sessions expire after 2 hours

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

            # Ensure embeddings are numpy arrays
            BASE_EMBEDDINGS = [np.array(emb) if not isinstance(emb, np.ndarray) else emb
                              for emb in BASE_EMBEDDINGS]

            BASE_PEOPLE = sorted(list(set(BASE_LABELS)))
            print(f"✓ Loaded {len(BASE_EMBEDDINGS)} BASE embeddings for {len(BASE_PEOPLE)} people")
            if len(BASE_PEOPLE) <= 20:
                print(f"  People: {', '.join(BASE_PEOPLE)}")
            else:
                print(f"  People: {', '.join(BASE_PEOPLE[:20])}... (+{len(BASE_PEOPLE)-20} more)")
        except Exception as e:
            print(f"Error loading base embeddings: {e}")
            BASE_EMBEDDINGS = []
            BASE_LABELS = []
            BASE_PEOPLE = []
    else:
        print("⚠️  embeddings.pkl not found! Create it first by training.")
        BASE_EMBEDDINGS = []
        BASE_LABELS = []
        BASE_PEOPLE = []

load_base_embeddings()

def cleanup_old_sessions():
    """Remove expired sessions"""
    now = datetime.now()
    to_remove = []

    for session_id, data in USER_SESSIONS.items():
        if now - data['last_access'] > SESSION_TIMEOUT:
            to_remove.append(session_id)

    for session_id in to_remove:
        del USER_SESSIONS[session_id]
        print(f"🗑️  Cleaned up expired session: {session_id[:8]}...")

def get_session_id():
    """Get or create session ID from cookie"""
    session_id = request.cookies.get('session_id')

    if not session_id or session_id not in USER_SESSIONS:
        session_id = str(uuid.uuid4())
        # Initialize new session with base data
        USER_SESSIONS[session_id] = {
            'embeddings': copy.deepcopy(BASE_EMBEDDINGS),
            'labels': BASE_LABELS.copy(),
            'unique_people': BASE_PEOPLE.copy(),
            'base_count': len(BASE_PEOPLE),
            'last_access': datetime.now()
        }
        print(f"  New session created: {session_id[:8]}... with {len(BASE_PEOPLE)} base people")
    else:
        # Update last access time
        USER_SESSIONS[session_id]['last_access'] = datetime.now()

    # Cleanup old sessions periodically
    if len(USER_SESSIONS) > 100:
        cleanup_old_sessions()

    return session_id

def get_session_data(session_id):
    """Get session-specific embeddings and labels"""
    if session_id not in USER_SESSIONS:
        get_session_id()  # This will create it

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
        except Exception as e:
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
    try:
        faces = get_face_app().get(image)
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
    response.set_cookie('session_id', session_id, max_age=7200)  # 2 hours
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

            if conf >= 0.7:
                color = (0, 255, 0)
            elif conf >= 0.5:
                color = (0, 255, 255)
            else:
                color = (0, 0, 255)

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

        summary = []
        for name, data in all_detections.items():
            summary.append({
                'name': name,
                'appearances': data['count'],
                'confidence': data['max_confidence'],
                'screenshot': data['screenshot']
            })

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
    """Add person to SESSION ONLY (temporary)"""
    try:
        session_id = get_session_id()
        data = request.get_json()

        if 'name' not in data:
            return jsonify({'error': 'Name required'}), 400

        name = data['name'].strip()

        if not name:
            return jsonify({'error': 'Name cannot be empty'}), 400

        embeddings, labels, unique_people = get_session_data(session_id)

        if name in labels:
            return jsonify({'error': f'{name} already exists in your session'}), 400

        images_data = data.get('images', [data.get('image')]) if 'images' in data else [data.get('image')]

        if not images_data or not images_data[0]:
            return jsonify({'error': 'Image(s) required'}), 400

        print(f"[{session_id[:8]}...] Processing {len(images_data)} images for {name}...")

        new_embeddings = []
        faces_detected = 0

        for idx, img_data in enumerate(images_data):
            try:
                if ',' in img_data:
                    img_data = img_data.split(',')[1]
                img_bytes = base64.b64decode(img_data)
                nparr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if img is None:
                    continue

                faces = get_face_app().get(img)

                if len(faces) > 0:
                    face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
                    embedding = face.embedding

                    if not isinstance(embedding, np.ndarray):
                        embedding = np.array(embedding)

                    new_embeddings.append(embedding)
                    faces_detected += 1

            except Exception as e:
                print(f"  Error processing image {idx + 1}: {e}")
                continue

        if faces_detected == 0:
            return jsonify({'error': 'No face detected in any image'}), 400

        # Add to session
        for embedding in new_embeddings:
            embeddings.append(embedding)
            labels.append(name)

        if name not in unique_people:
            unique_people.append(name)
            unique_people.sort()

        update_session_data(session_id, embeddings, labels, unique_people)

        print(f"✓ [{session_id[:8]}...] Added {name} with {faces_detected} embeddings (temporary)")

        return jsonify({
            'success': True,
            'message': f'Added {name} to your session with {faces_detected} images!',
            'total_people': len(unique_people),
            'embeddings_added': faces_detected,
            'is_temporary': True  # THIS WAS MISSING!
        })

    except Exception as e:
        print(f"Error in add_person: {e}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/api/detect_face', methods=['POST'])
def detect_face():
    """NEW: Detect if face is in frame for smart capture"""
    try:
        data = request.get_json()
        if 'image' not in data:
            return jsonify({'error': 'No image data'}), 400

        img_data = data['image'].split(',')[1]
        img_bytes = base64.b64decode(img_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        faces = get_face_app().get(img)


        return jsonify({
            'face_detected': len(faces) > 0,
            'face_count': len(faces)
        })
    except Exception as e:
        return jsonify({'error': str(e), 'face_detected': False}), 500

@app.route('/api/delete_person', methods=['POST'])
def delete_person():
    """Delete person from SESSION ONLY (temporary)"""
    try:
        session_id = get_session_id()
        data = request.get_json()

        if 'name' not in data:
            return jsonify({'error': 'Name required'}), 400

        name = data['name']
        embeddings, labels, unique_people = get_session_data(session_id)

        if name not in labels:
            return jsonify({'error': f'{name} not found in your session'}), 400

        is_base_person = name in BASE_PEOPLE

        indices_to_remove = [i for i, label in enumerate(labels) if label == name]

        if not indices_to_remove:
            return jsonify({'error': f'No embeddings found for {name}'}), 400

        # Remove from session
        for i in sorted(indices_to_remove, reverse=True):
            del embeddings[i]
            del labels[i]

        if name in unique_people:
            unique_people.remove(name)

        update_session_data(session_id, embeddings, labels, unique_people)

        msg = f"Removed {name} from your session"
        if is_base_person:
            msg += " (will reappear on page refresh)"

        print(f"✓ [{session_id[:8]}...] Removed {name} (temporary)")

        return jsonify({
            'success': True,
            'message': msg,
            'total_people': len(unique_people),
            'is_base_person': is_base_person
        })

    except Exception as e:
        print(f"Error in delete_person: {e}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/api/reset', methods=['POST'])
def reset_session():
    """Reset session to base embeddings"""
    session_id = get_session_id()

    # Reinitialize session with base data
    USER_SESSIONS[session_id] = {
        'embeddings': copy.deepcopy(BASE_EMBEDDINGS),
        'labels': BASE_LABELS.copy(),
        'unique_people': BASE_PEOPLE.copy(),
        'base_count': len(BASE_PEOPLE),
        'last_access': datetime.now()
    }

    print(f"🔄 [{session_id[:8]}...] Reset to base data ({len(BASE_PEOPLE)} people)")

    return jsonify({
        'success': True,
        'message': f'Session reset! Back to {len(BASE_PEOPLE)} base people.',
        'total_people': len(BASE_PEOPLE)
    })

if __name__ == '__main__':
    print("\n" + "="*70)
    print("ARCFACE FACE RECOGNITION - SERVER-SIDE SESSION STORAGE")
    print("="*70)
    print(f"📁 Base embeddings: {len(BASE_EMBEDDINGS)} embeddings for {len(BASE_PEOPLE)} people")

    if len(BASE_PEOPLE) > 0:
        if len(BASE_PEOPLE) <= 20:
            print(f"   People: {', '.join(BASE_PEOPLE)}")
        else:
            print(f"   People: {', '.join(BASE_PEOPLE[:20])}... (+{len(BASE_PEOPLE)-20} more)")
    else:
        print("   ⚠️  No base embeddings found!")

    print("\n🔐 SESSION FEATURES:")
    print("   ✓ Server-side session storage (handles large datasets)")
    print("   ✓ Each user gets their own copy of base data")
    print("   ✓ User additions are temporary (session-only)")
    print("   ✓ User deletions are temporary (session-only)")
    print("   ✓ Base embeddings NEVER modified")
    print("   ✓ Multiple users can use simultaneously")
    print("   ✓ Sessions expire after 2 hours of inactivity")

    print("\n💡 USER EXPERIENCE:")
    print("   - Starts with all base people")
    print("   - Can add new people (temporary)")
    print("   - Can delete anyone (temporary)")
    print("   - Refresh page → back to base people")
    print("   - Click Reset → back to base people")

    print("\n🌐 Server running at: http://localhost:5000")
    print("="*70 + "\n")

    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, port=port, host='0.0.0.0')