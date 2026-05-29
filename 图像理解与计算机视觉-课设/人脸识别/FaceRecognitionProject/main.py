import cv2
import numpy as np
import os
import pickle
import base64
import torch
import time
from flask import Flask, render_template, Response, request, jsonify
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

app = Flask(__name__)

FACE_DB_DIR = "./face_database"
FEATURE_DB_PATH = "./face_features.pkl"
SIMILARITY_THRESHOLD = 0.65

if not os.path.exists(FACE_DB_DIR):
    os.makedirs(FACE_DB_DIR)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

database_features = {}
camera = None
recognition_log = []
last_seen_dict = {}

def load_database():
    global database_features
    if os.path.exists(FEATURE_DB_PATH):
        with open(FEATURE_DB_PATH, 'rb') as f:
            database_features = pickle.load(f)
        print(f"[*] Loaded {len(database_features)} faces from database")

def save_database():
    with open(FEATURE_DB_PATH, 'wb') as f:
        pickle.dump(database_features, f)

load_database()

def compute_cosine_similarity(feat1, feat2):
    dot_product = np.dot(feat1, feat2)
    norm1 = np.linalg.norm(feat1)
    norm2 = np.linalg.norm(feat2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)

def process_image(image_bgr, is_video=False):
    global recognition_log, last_seen_dict
    
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    
    recognized_names_in_frame = []
    boxes, probs = mtcnn.detect(pil_img)
    
    if boxes is not None:
        faces = mtcnn.extract(pil_img, boxes, save_path=None)
        if faces is not None:
            faces = faces.to(device)
            
            with torch.no_grad():
                embeddings = resnet(faces).cpu().numpy()
            
            for i in range(len(boxes)):
                box = boxes[i]
                embedding = embeddings[i]
                x1, y1, x2, y2 = [int(b) for b in box]
                
                best_name = "Unknown"
                highest_sim = -1.0
                
                for name, db_embedding in database_features.items():
                    sim = compute_cosine_similarity(embedding, db_embedding)
                    if sim > highest_sim:
                        highest_sim = sim
                        best_name = name
                
                if highest_sim >= SIMILARITY_THRESHOLD:
                    text = f"{best_name} ({highest_sim:.2f})"
                    color = (0, 255, 0)
                else:
                    text = "Unknown"
                    color = (0, 0, 255)
                
                recognized_names_in_frame.append(best_name)
                
                if is_video:
                    now = time.time()
                    if best_name not in last_seen_dict or (now - last_seen_dict[best_name] > 3):
                        recognition_log.insert(0, {"name": best_name, "time": time.strftime("%H:%M:%S")})
                        last_seen_dict[best_name] = now
                        if len(recognition_log) > 20:
                            recognition_log.pop()
                
                cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color, 2)
                cv2.putText(image_bgr, text, (x1, max(y1 - 10, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    return image_bgr, recognized_names_in_frame

@app.route('/')
def index():
    return render_template('index.html')

def generate_frames():
    global camera
    while camera and camera.isOpened():
        success, frame = camera.read()
        if not success:
            break
        frame = cv2.flip(frame, 1)
        processed_frame, _ = process_image(frame, is_video=True)
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/camera/start', methods=['POST'])
def start_camera():
    global camera, recognition_log, last_seen_dict
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0)
        recognition_log = []
        last_seen_dict = {}
    return jsonify({"success": True})

@app.route('/api/camera/stop', methods=['POST'])
def stop_camera():
    global camera
    if camera is not None:
        camera.release()
        camera = None
    return jsonify({"success": True})

@app.route('/api/camera/logs', methods=['GET'])
def get_camera_logs():
    return jsonify({"logs": recognition_log})

@app.route('/api/recognize_static', methods=['POST'])
def recognize_static():
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"})
    file = request.files['file']
    np_img = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    
    if img is None:
        return jsonify({"success": False, "error": "Failed to read image"})
    
    processed_img, recognized_names = process_image(img, is_video=False)
    _, buffer = cv2.imencode('.jpg', processed_img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return jsonify({"success": True, "image": img_base64, "faces": recognized_names})

@app.route('/api/db/list', methods=['GET'])
def db_list():
    return jsonify({"names": list(database_features.keys())})

@app.route('/api/db/add', methods=['POST'])
def db_add():
    name = request.form.get('name')
    file = request.files.get('file')
    
    if not name or not file:
        return jsonify({"success": False, "error": "Missing parameters"})
    if name in database_features:
        return jsonify({"success": False, "error": "Name already exists"})
    
    np_img = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    
    if img is None:
        return jsonify({"success": False, "error": "Failed to decode image"})
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    
    face_tensor = mtcnn(pil_img)
    if face_tensor is None:
        return jsonify({"success": False, "error": "No face detected"})
    
    if face_tensor.dim() == 4:
        face_tensor = face_tensor[0]
    
    face_tensor = face_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = resnet(face_tensor).cpu().numpy()[0]
    
    database_features[name] = embedding
    save_database()
    cv2.imwrite(os.path.join(FACE_DB_DIR, f"{name}.jpg"), img)
    return jsonify({"success": True})

@app.route('/api/db/delete', methods=['POST'])
def db_delete():
    name = request.get_json().get('name')
    if name in database_features:
        del database_features[name]
        save_database()
        file_path = os.path.join(FACE_DB_DIR, f"{name}.jpg")
        if os.path.exists(file_path):
            os.remove(file_path)
    return jsonify({"success": True})

if __name__ == "__main__":
    print("\n" + "="*50)
    print("Face Recognition System Started")
    print("Access: http://127.0.0.1:5000")
    print("="*50 + "\n")
    app.run(host='127.0.0.1', port=5000, debug=False)
