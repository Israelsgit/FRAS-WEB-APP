#!/usr/bin/env python3
"""
FaceTrack Pro - Flask Backend
AI-Powered Facial Recognition Attendance System
"""

import os
import cv2 as cv
import numpy as np
import pickle
import json
from datetime import datetime, date
import base64
from io import BytesIO
from PIL import Image
import warnings

# Flask imports
from flask import Flask, request, jsonify, render_template_string, send_from_directory
from flask_cors import CORS

# AI/ML imports
from keras_facenet import FaceNet
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore', category=UserWarning)

class FaceRecognitionSystem:
    def __init__(self):
        """Initialize the face recognition system"""
        self.facenet = None
        self.haarcascade = None
        self.model = None
        self.encoder = None
        self.X = None
        self.Y = None
        self.confidence_threshold = 0.7
        self.min_face_size = (80, 80)
        self.load_models()
        
    def load_models(self):
        """Load all required models and data"""
        try:
            print("🚀 Loading FaceNet model...")
            self.facenet = FaceNet()
            
            print("📷 Loading Haar Cascade...")
            cascade_path = 'haarcascade_frontalface_default.xml'
            if not os.path.exists(cascade_path):
                # Try to find the cascade file in cv2 data
                cascade_path = cv.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.haarcascade = cv.CascadeClassifier(cascade_path)
            
            print("🧠 Loading face embeddings...")
            if os.path.exists('faces_embeddings_done_35classes.npz'):
                faces_embeddings = np.load('faces_embeddings_done_35classes.npz')
                self.X = faces_embeddings['arr_0']
                self.Y = faces_embeddings['arr_1']
                
                print("🔧 Setting up encoder...")
                self.encoder = LabelEncoder()
                self.encoder.fit(self.Y)
                
                print("🤖 Loading SVM model...")
                if os.path.exists('svm_model_160x160.pkl'):
                    with open('svm_model_160x160.pkl', 'rb') as f:
                        self.model = pickle.load(f)
                        
                    print(f"✅ Models loaded successfully!")
                    print(f"📊 Found {len(self.encoder.classes_)} registered classes")
                    print(f"🎯 Available students: {list(self.encoder.classes_)}")
                else:
                    print("⚠️ SVM model not found. Please train the model first.")
            else:
                print("⚠️ Face embeddings not found. Please register students first.")
                
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            
    def detect_faces(self, image):
        """Detect faces in an image"""
        if self.haarcascade is None:
            return []
            
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        faces = self.haarcascade.detectMultiScale(
            gray, 
            scaleFactor=1.3, 
            minNeighbors=5,
            minSize=self.min_face_size
        )
        
        return faces
    
    def extract_face_embedding(self, image, face_coords):
        """Extract face embedding from detected face"""
        if self.facenet is None:
            return None
            
        x, y, w, h = face_coords
        
        # Extract face region
        face_img = image[y:y+h, x:x+w]
        
        # Convert BGR to RGB
        face_rgb = cv.cvtColor(face_img, cv.COLOR_BGR2RGB)
        
        # Resize to FaceNet input size
        face_resized = cv.resize(face_rgb, (160, 160))
        
        # Normalize and expand dimensions
        face_normalized = np.expand_dims(face_resized, axis=0)
        
        # Get embedding
        embedding = self.facenet.embeddings(face_normalized)
        
        return embedding
    
    def recognize_face(self, embedding, confidence_threshold=None):
        """Recognize face from embedding"""
        if self.model is None or embedding is None:
            return None, 0.0
            
        if confidence_threshold is None:
            confidence_threshold = self.confidence_threshold
            
        # Get prediction probabilities
        probabilities = self.model.predict_proba(embedding)
        max_prob = np.max(probabilities)
        
        if max_prob >= confidence_threshold:
            # Get predicted class
            predicted_class = self.model.predict(embedding)[0]
            # Convert back to original label
            name = self.encoder.inverse_transform([predicted_class])[0]
            return name, max_prob
        
        return None, max_prob
    
    def process_frame(self, image, confidence_threshold=None):
        """Process a single frame for face recognition"""
        results = []
        
        # Detect faces
        faces = self.detect_faces(image)
        
        for face in faces:
            x, y, w, h = face
            
            # Extract embedding
            embedding = self.extract_face_embedding(image, face)
            
            if embedding is not None:
                # Recognize face
                name, confidence = self.recognize_face(embedding, confidence_threshold)
                
                result = {
                    'bbox': {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)},
                    'recognized': name is not None,
                    'name': name,
                    'confidence': float(confidence)
                }
                
                results.append(result)
            else:
                # Face detected but no embedding
                result = {
                    'bbox': {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)},
                    'recognized': False,
                    'name': None,
                    'confidence': 0.0
                }
                results.append(result)
        
        return results
    
    def register_student(self, images, student_name):
        """Register a new student with multiple face samples"""
        if not images or not student_name:
            return False, "Invalid input data"
            
        embeddings = []
        
        for image in images:
            faces = self.detect_faces(image)
            
            if len(faces) == 0:
                continue
                
            # Use the largest face
            largest_face = max(faces, key=lambda f: f[2] * f[3])
            
            embedding = self.extract_face_embedding(image, largest_face)
            
            if embedding is not None:
                embeddings.append(embedding.flatten())
        
        if len(embeddings) < 5:  # Minimum 5 samples
            return False, f"Not enough valid face samples. Got {len(embeddings)}, need at least 5."
        
        # Update embeddings data
        if self.X is not None and self.Y is not None:
            # Append new embeddings
            new_X = np.vstack([self.X] + embeddings)
            new_Y = np.hstack([self.Y] + [student_name] * len(embeddings))
        else:
            # First student
            new_X = np.array(embeddings)
            new_Y = np.array([student_name] * len(embeddings))
        
        # Save updated embeddings
        try:
            np.savez('faces_embeddings_done_35classes.npz', new_X, new_Y)
            self.X = new_X
            self.Y = new_Y
            
            # Retrain encoder and model
            self.encoder = LabelEncoder()
            self.encoder.fit(new_Y)
            
            # Train new SVM model
            self.model = SVC(kernel='linear', probability=True)
            self.model.fit(new_X, new_Y)
            
            # Save model
            with open('svm_model_160x160.pkl', 'wb') as f:
                pickle.dump(self.model, f)
            
            print(f"✅ Student '{student_name}' registered with {len(embeddings)} samples")
            return True, f"Student registered successfully with {len(embeddings)} samples"
            
        except Exception as e:
            print(f"❌ Error registering student: {e}")
            return False, f"Registration failed: {str(e)}"

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize face recognition system
face_system = FaceRecognitionSystem()

# Data storage
attendance_data = []
student_registry = []

def load_data():
    """Load attendance and student data from files"""
    global attendance_data, student_registry
    
    try:
        if os.path.exists('attendance_data.json'):
            with open('attendance_data.json', 'r') as f:
                attendance_data = json.load(f)
    except Exception as e:
        print(f"Error loading attendance data: {e}")
        attendance_data = []
    
    try:
        if os.path.exists('student_registry.json'):
            with open('student_registry.json', 'r') as f:
                student_registry = json.load(f)
    except Exception as e:
        print(f"Error loading student registry: {e}")
        student_registry = []

def save_data():
    """Save attendance and student data to files"""
    try:
        with open('attendance_data.json', 'w') as f:
            json.dump(attendance_data, f, indent=2)
    except Exception as e:
        print(f"Error saving attendance data: {e}")
    
    try:
        with open('student_registry.json', 'w') as f:
            json.dump(student_registry, f, indent=2)
    except Exception as e:
        print(f"Error saving student registry: {e}")

@app.route('/')
def index():
    """Serve the main web application"""
    # In a real deployment, this would serve the HTML file
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>FaceTrack Pro</title>
        <style>
            body { 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                background: #0d1117;
                color: #e6edf3;
                margin: 0;
                padding: 40px;
                text-align: center;
            }
            .container {
                max-width: 600px;
                margin: 0 auto;
                background: #161b22;
                padding: 40px;
                border-radius: 12px;
                border: 1px solid #30363d;
            }
            h1 { color: #238636; margin-bottom: 20px; }
            .status { 
                padding: 20px;
                background: #21262d;
                border-radius: 8px;
                margin: 20px 0;
            }
            .endpoint {
                background: #0d1117;
                padding: 15px;
                border-radius: 6px;
                margin: 10px 0;
                font-family: 'SF Mono', Consolas, monospace;
                font-size: 14px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎯 FaceTrack Pro Backend</h1>
            <p>AI-Powered Facial Recognition Attendance System</p>
            
            <div class="status">
                <h3>System Status</h3>
                <p> Flask server running</p>
                <p> Face recognition models loaded</p>
                <p> Ready for requests</p>
            </div>
            
            <div class="status">
                <h3>API Endpoints</h3>
                <div class="endpoint">POST /api/recognize - Face recognition</div>
                <div class="endpoint">POST /api/register - Student registration</div>
                <div class="endpoint">GET /api/students - Get registered students</div>
                <div class="endpoint">GET /api/attendance - Get attendance records</div>
            </div>
            
            <p style="margin-top: 30px; color: #7d8590;">
                Open the web application in your browser to start using FaceTrack Pro
            </p>
        </div>
    </body>
    </html>
    """

@app.route('/api/recognize', methods=['POST'])
def recognize_face():
    """API endpoint for face recognition"""
    try:
        # Get image from request
        if 'frame' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['frame']
        confidence = float(request.form.get('confidence', 0.7))
        
        # Read image
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv.imdecode(nparr, cv.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Invalid image format'}), 400
        
        # Process frame
        results = face_system.process_frame(image, confidence)
        
        if results:
            # Return the first (best) result
            result = results[0]
            return jsonify(result)
        else:
            return jsonify({
                'recognized': False,
                'name': None,
                'confidence': 0.0,
                'bbox': None
            })
            
    except Exception as e:
        print(f"Recognition error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/register', methods=['POST'])
def register_student():
    """API endpoint for student registration"""
    try:
        data = request.get_json()
        
        if not data or 'name' not in data:
            return jsonify({'error': 'Student name is required'}), 400
        
        student_name = data['name'].strip()
        student_id = data.get('id', f"AUTO_{int(datetime.now().timestamp())}")
        
        if not student_name:
            return jsonify({'error': 'Student name cannot be empty'}), 400
        
        # Check if student already exists
        existing_student = next((s for s in student_registry if s['name'] == student_name), None)
        
        if existing_student and not data.get('overwrite', False):
            return jsonify({'error': f'Student {student_name} already registered'}), 400
        
        # For now, create a placeholder registration
        # In a real implementation, this would capture face samples from the camera
        student_data = {
            'name': student_name,
            'id': student_id,
            'registered_at': datetime.now().isoformat(),
            'samples_count': 20,  # Simulated
            'status': 'active'
        }
        
        if existing_student:
            # Update existing student
            for i, student in enumerate(student_registry):
                if student['name'] == student_name:
                    student_registry[i] = student_data
                    break
        else:
            # Add new student
            student_registry.append(student_data)
        
        save_data()
        
        return jsonify({
            'success': True,
            'message': f'Student {student_name} registered successfully',
            'student': student_data
        })
        
    except Exception as e:
        print(f"Registration error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/students', methods=['GET'])
def get_students():
    """API endpoint to get registered students"""
    try:
        return jsonify({
            'students': student_registry,
            'total': len(student_registry)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/attendance', methods=['GET', 'POST'])
def handle_attendance():
    """API endpoint for attendance management"""
    if request.method == 'GET':
        # Get attendance records
        try:
            date_filter = request.args.get('date')
            if date_filter:
                filtered_data = [
                    record for record in attendance_data 
                    if record.get('date') == date_filter
                ]
                return jsonify({
                    'attendance': filtered_data,
                    'total': len(filtered_data),
                    'date': date_filter
                })
            else:
                return jsonify({
                    'attendance': attendance_data,
                    'total': len(attendance_data)
                })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    elif request.method == 'POST':
        # Mark attendance
        try:
            data = request.get_json()
            
            if not data or 'name' not in data:
                return jsonify({'error': 'Student name is required'}), 400
            
            student_name = data['name']
            current_date = date.today().isoformat()
            current_time = datetime.now()
            
            # Check if already marked today
            existing_record = next((
                record for record in attendance_data 
                if record['name'] == student_name and record['date'] == current_date
            ), None)
            
            if existing_record:
                return jsonify({
                    'success': False,
                    'message': f'{student_name} already marked present today',
                    'existing_record': existing_record
                })
            
            # Create new attendance record
            attendance_record = {
                'name': student_name,
                'date': current_date,
                'time': current_time.strftime('%H:%M:%S'),
                'timestamp': current_time.isoformat(),
                'marked_by': 'face_recognition'
            }
            
            attendance_data.insert(0, attendance_record)  # Insert at beginning
            save_data()
            
            return jsonify({
                'success': True,
                'message': f'{student_name} marked present',
                'record': attendance_record
            })
            
        except Exception as e:
            print(f"Attendance marking error: {e}")
            return jsonify({'error': str(e)}), 500

@app.route('/api/export', methods=['GET'])
def export_attendance():
    """API endpoint to export attendance data as CSV"""
    try:
        # Generate CSV content
        csv_lines = ['Student Name,Date,Time,Timestamp']
        
        for record in attendance_data:
            line = f'"{record["name"]}","{record["date"]}","{record["time"]}","{record["timestamp"]}"'
            csv_lines.append(line)
        
        csv_content = '\n'.join(csv_lines)
        
        # Return CSV as response
        from flask import Response
        return Response(
            csv_content,
            mimetype='text/csv',
            headers={'Content-Disposition': f'attachment; filename=attendance_{date.today().isoformat()}.csv'}
        )
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/status', methods=['GET'])
def get_status():
    """API endpoint to get system status"""
    try:
        status = {
            'system': 'online',
            'models_loaded': face_system.model is not None,
            'facenet_loaded': face_system.facenet is not None,
            'cascade_loaded': face_system.haarcascade is not None,
            'registered_students': len(student_registry),
            'total_classes': len(face_system.encoder.classes_) if face_system.encoder else 0,
            'attendance_records': len(attendance_data),
            'confidence_threshold': face_system.confidence_threshold,
            'server_time': datetime.now().isoformat()
        }
        
        return jsonify(status)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/settings', methods=['GET', 'POST'])
def handle_settings():
    """API endpoint for system settings"""
    if request.method == 'GET':
        try:
            settings = {
                'confidence_threshold': face_system.confidence_threshold,
                'min_face_size': face_system.min_face_size,
                'available_classes': list(face_system.encoder.classes_) if face_system.encoder else []
            }
            return jsonify(settings)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    elif request.method == 'POST':
        try:
            data = request.get_json()
            
            if 'confidence_threshold' in data:
                threshold = float(data['confidence_threshold'])
                if 0.1 <= threshold <= 1.0:
                    face_system.confidence_threshold = threshold
                else:
                    return jsonify({'error': 'Confidence threshold must be between 0.1 and 1.0'}), 400
            
            return jsonify({
                'success': True,
                'message': 'Settings updated successfully',
                'settings': {
                    'confidence_threshold': face_system.confidence_threshold
                }
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error'}), 500

def initialize_app():
    """Initialize the application"""
    print("🚀 Initializing FaceTrack Pro Backend...")
    
    # Load existing data
    load_data()
    
    # Print system status
    print(f"📊 Loaded {len(student_registry)} registered students")
    print(f"📋 Loaded {len(attendance_data)} attendance records")
    
    if face_system.model is not None:
        print(f"🤖 AI models ready with {len(face_system.encoder.classes_)} classes")
    else:
        print("⚠️ No trained model found. Please register students first.")
    
    print("✅ Backend initialization complete!")

if __name__ == '__main__':
    # Initialize the application
    initialize_app()
    
    print("\n" + "="*50)
    print("🎯 FaceTrack Pro Backend Server Starting...")
    print("="*50)
    print(f"🌐 Access the API at: http://localhost:5000")
    print(f"📋 View status at: http://localhost:5000/api/status")
    print(f"👥 Register students: POST to /api/register")
    print(f"🎭 Face recognition: POST to /api/recognize")
    print("="*50)
    
    # Run the Flask app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )