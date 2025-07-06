# Enhanced Backend Registration System for FaceTrack Pro
# Updated for new interface with dashboard features and enhanced functionality
# Integrates with existing data files: faces_embeddings_done_35classes.npz, svm_model_160x160.pkl, 
# student_registry.json, attendance_data.json

import cv2 as cv
import numpy as np
import json
import pickle
import os
import csv
from datetime import datetime, date
from flask import Flask, request, jsonify, Response, render_template_string, send_from_directory, make_response
from flask_cors import CORS
from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import base64
import io
from PIL import Image
import threading
import time

class EnhancedFaceRegistrationSystem:
    def __init__(self):
        """Initialize the enhanced face registration system"""
        print("🚀 Initializing Enhanced FaceTrack Pro Registration System...")
        
        # Initialize core components
        self.facenet = FaceNet()
        self.haarcascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
        
        # Load existing data
        self.load_existing_data()
        
        # Registration tracking
        self.active_registrations = {}
        self.registration_lock = threading.Lock()
        
        print("✅ Enhanced registration system initialized successfully!")
    
    def load_existing_data(self):
        """Load existing embeddings, model, and registry data"""
        try:
            # Load face embeddings
            if os.path.exists('faces_embeddings_done_35classes.npz'):
                print("📊 Loading existing face embeddings...")
                self.faces_embeddings = np.load('faces_embeddings_done_35classes.npz')
                self.X = self.faces_embeddings['arr_0']
                self.Y = self.faces_embeddings['arr_1']
                print(f"   Loaded {len(self.X)} face embeddings for {len(set(self.Y))} students")
            else:
                print("📊 No existing embeddings found - starting fresh")
                self.X = np.array([])
                self.Y = np.array([])
            
            # Load trained model
            if os.path.exists('svm_model_160x160.pkl') and len(self.Y) > 0:
                print("🤖 Loading trained SVM model...")
                with open('svm_model_160x160.pkl', 'rb') as f:
                    self.model = pickle.load(f)
                
                # Setup encoder
                self.encoder = LabelEncoder()
                self.encoder.fit(self.Y)
                print(f"   Model loaded with {len(self.encoder.classes_)} classes")
            else:
                print("🤖 No trained model found - will create after first registration")
                self.model = None
                self.encoder = LabelEncoder()
            
            # Load student registry
            if os.path.exists('student_registry.json'):
                print("👥 Loading student registry...")
                with open('student_registry.json', 'r') as f:
                    self.student_registry = json.load(f)
                print(f"   Loaded {len(self.student_registry)} registered students")
            else:
                print("👥 No student registry found - creating new")
                self.student_registry = []
            
            # Load attendance data
            if os.path.exists('attendance_data.json'):
                print("📋 Loading attendance data...")
                with open('attendance_data.json', 'r') as f:
                    self.attendance_data = json.load(f)
                print(f"   Loaded {len(self.attendance_data)} attendance records")
            else:
                print("📋 No attendance data found - creating new")
                self.attendance_data = []
                
        except Exception as e:
            print(f"❌ Error loading existing data: {e}")
            # Initialize empty data structures
            self.X = np.array([])
            self.Y = np.array([])
            self.model = None
            self.encoder = LabelEncoder()
            self.student_registry = []
            self.attendance_data = []
    
    def save_all_data(self):
        """Save all data to respective files"""
        try:
            # Save embeddings
            if len(self.X) > 0:
                np.savez('faces_embeddings_done_35classes.npz', self.X, self.Y)
                print("💾 Face embeddings saved")
            
            # Save model
            if self.model is not None:
                with open('svm_model_160x160.pkl', 'wb') as f:
                    pickle.dump(self.model, f)
                print("💾 SVM model saved")
            
            # Save student registry
            with open('student_registry.json', 'w') as f:
                json.dump(self.student_registry, f, indent=2)
            print("💾 Student registry saved")
            
            # Save attendance data
            with open('attendance_data.json', 'w') as f:
                json.dump(self.attendance_data, f, indent=2)
            print("💾 Attendance data saved")
            
        except Exception as e:
            print(f"❌ Error saving data: {e}")
    
    def start_registration(self, student_name, student_id=None, max_samples=20):
        """Start a new student registration session"""
        with self.registration_lock:
            if student_name in self.active_registrations:
                return {'error': 'Registration already in progress for this student'}
            
            # Create registration session
            session_id = f"{student_name}_{int(time.time())}"
            self.active_registrations[student_name] = {
                'session_id': session_id,
                'student_name': student_name,
                'student_id': student_id or f"AUTO_{int(datetime.now().timestamp())}",
                'captured_samples': [],
                'max_samples': max_samples,
                'start_time': datetime.now().isoformat(),
                'status': 'active'
            }
            
            print(f"📸 Started registration session for {student_name}")
            return {
                'success': True,
                'session_id': session_id,
                'message': f'Registration started for {student_name}',
                'max_samples': max_samples
            }
    
    def process_registration_frame(self, student_name, frame_data, quality_threshold=70):
        """Process a frame for student registration"""
        try:
            with self.registration_lock:
                if student_name not in self.active_registrations:
                    return {'error': 'No active registration session for this student'}
                
                session = self.active_registrations[student_name]
                if len(session['captured_samples']) >= session['max_samples']:
                    return {'error': 'Maximum samples already captured'}
            
            # Decode base64 image
            if isinstance(frame_data, str):
                # Remove data URL prefix if present
                if frame_data.startswith('data:image'):
                    frame_data = frame_data.split(',')[1]
                image_bytes = base64.b64decode(frame_data)
            else:
                image_bytes = frame_data
            
            # Convert to OpenCV format
            image = Image.open(io.BytesIO(image_bytes))
            image_np = np.array(image)
            if len(image_np.shape) == 3 and image_np.shape[2] == 3:
                image_bgr = cv.cvtColor(image_np, cv.COLOR_RGB2BGR)
            else:
                image_bgr = image_np
            
            # Detect faces
            gray = cv.cvtColor(image_bgr, cv.COLOR_BGR2GRAY)
            faces = self.haarcascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) == 0:
                return {
                    'face_detected': False,
                    'message': 'No face detected',
                    'captured_count': len(session['captured_samples'])
                }
            
            # Process the largest face
            largest_face = max(faces, key=lambda f: f[2] * f[3])
            x, y, w, h = largest_face
            
            # Check minimum face size
            if w < 100 or h < 100:
                return {
                    'face_detected': True,
                    'face_too_small': True,
                    'message': 'Face too small - move closer to camera',
                    'captured_count': len(session['captured_samples'])
                }
            
            # Extract face
            face_roi = image_bgr[y:y+h, x:x+w]
            face_rgb = cv.cvtColor(face_roi, cv.COLOR_BGR2RGB)
            face_resized = cv.resize(face_rgb, (160, 160))
            
            # Quality assessment
            quality_score = self.assess_face_quality(face_resized)
            
            if quality_score < quality_threshold:
                return {
                    'face_detected': True,
                    'quality_too_low': True,
                    'quality_score': quality_score,
                    'message': f'Image quality too low ({quality_score:.1f}%) - improve lighting and focus',
                    'captured_count': len(session['captured_samples'])
                }
            
            # Generate embedding
            face_expanded = np.expand_dims(face_resized, axis=0)
            embedding = self.facenet.embeddings(face_expanded)[0]
            
            # Add to captured samples
            with self.registration_lock:
                session['captured_samples'].append({
                    'embedding': embedding.tolist(),
                    'quality_score': quality_score,
                    'timestamp': datetime.now().isoformat(),
                    'bbox': [int(x), int(y), int(w), int(h)]
                })
                
                captured_count = len(session['captured_samples'])
                max_samples = session['max_samples']
                progress = (captured_count / max_samples) * 100
                
                print(f"📸 Captured sample {captured_count}/{max_samples} for {student_name} (quality: {quality_score:.1f}%)")
                
                if captured_count >= max_samples:
                    # Registration complete - process samples
                    result = self.complete_registration(student_name)
                    return result
                
                return {
                    'face_detected': True,
                    'sample_captured': True,
                    'quality_score': quality_score,
                    'captured_count': captured_count,
                    'max_samples': max_samples,
                    'progress': progress,
                    'message': f'Sample {captured_count}/{max_samples} captured (quality: {quality_score:.1f}%)',
                    'bbox': [int(x), int(y), int(w), int(h)]
                }
                
        except Exception as e:
            print(f"❌ Error processing registration frame: {e}")
            return {'error': f'Frame processing error: {str(e)}'}
    
    def assess_face_quality(self, face_image):
        """Assess the quality of a face image"""
        try:
            # Convert to grayscale for analysis
            gray = cv.cvtColor(face_image, cv.COLOR_RGB2GRAY)
            
            # Check brightness
            mean_brightness = np.mean(gray)
            brightness_score = 100 - abs(mean_brightness - 128) / 128 * 100
            
            # Check sharpness using Laplacian variance
            laplacian_var = cv.Laplacian(gray, cv.CV_64F).var()
            sharpness_score = min(laplacian_var / 10, 100)  # Normalize to 0-100
            
            # Check contrast using standard deviation
            contrast_score = min(np.std(gray) / 64 * 100, 100)  # Normalize to 0-100
            
            # Overall quality score (weighted average)
            quality_score = (brightness_score * 0.3 + sharpness_score * 0.5 + contrast_score * 0.2)
            
            return max(0, min(100, quality_score))
            
        except Exception as e:
            print(f"❌ Error assessing face quality: {e}")
            return 50  # Default medium quality
    
    def complete_registration(self, student_name):
        """Complete the registration process and update the model"""
        try:
            with self.registration_lock:
                if student_name not in self.active_registrations:
                    return {'error': 'No active registration session'}
                
                session = self.active_registrations[student_name]
                samples = session['captured_samples']
                student_id = session['student_id']
                
                if len(samples) == 0:
                    return {'error': 'No samples captured'}
                
                print(f"🔄 Completing registration for {student_name} with {len(samples)} samples...")
                
                # Extract embeddings
                embeddings = np.array([sample['embedding'] for sample in samples])
                labels = np.array([student_name] * len(samples))
                
                # Update or add to existing data
                if len(self.X) == 0:
                    # First student
                    self.X = embeddings
                    self.Y = labels
                else:
                    # Check if student already exists
                    existing_mask = self.Y == student_name
                    if np.any(existing_mask):
                        # Remove existing samples for this student
                        keep_mask = ~existing_mask
                        self.X = self.X[keep_mask]
                        self.Y = self.Y[keep_mask]
                    
                    # Add new samples
                    self.X = np.vstack([self.X, embeddings])
                    self.Y = np.hstack([self.Y, labels])
                
                # Update encoder
                self.encoder = LabelEncoder()
                self.encoder.fit(self.Y)
                
                # Train new model
                print(f"🤖 Training SVM model with {len(self.X)} samples from {len(set(self.Y))} students...")
                self.model = SVC(kernel='linear', probability=True, random_state=42)
                self.model.fit(self.X, self.Y)
                
                # Update student registry
                existing_student_idx = None
                for i, student in enumerate(self.student_registry):
                    if student['name'] == student_name:
                        existing_student_idx = i
                        break
                
                student_data = {
                    'name': student_name,
                    'id': student_id,
                    'registered_at': session['start_time'],
                    'samples_count': len(samples),
                    'last_updated': datetime.now().isoformat(),
                    'average_quality': np.mean([s['quality_score'] for s in samples]),
                    'status': 'active'
                }
                
                if existing_student_idx is not None:
                    # Update existing student
                    student_data['registered_at'] = self.student_registry[existing_student_idx]['registered_at']
                    self.student_registry[existing_student_idx] = student_data
                    print(f"✅ Updated existing student {student_name}")
                else:
                    # Add new student
                    self.student_registry.append(student_data)
                    print(f"✅ Added new student {student_name}")
                
                # Save all data
                self.save_all_data()
                
                # Clean up registration session
                del self.active_registrations[student_name]
                
                total_students = len(set(self.Y))
                print(f"🎉 Registration completed! Total students: {total_students}")
                
                return {
                    'success': True,
                    'message': f'Registration completed successfully for {student_name}',
                    'student_data': student_data,
                    'total_students': total_students,
                    'total_samples': len(self.X)
                }
                
        except Exception as e:
            print(f"❌ Error completing registration: {e}")
            # Clean up failed registration
            if student_name in self.active_registrations:
                del self.active_registrations[student_name]
            return {'error': f'Registration completion failed: {str(e)}'}
    
    def cancel_registration(self, student_name):
        """Cancel an active registration session"""
        with self.registration_lock:
            if student_name in self.active_registrations:
                del self.active_registrations[student_name]
                print(f"❌ Registration cancelled for {student_name}")
                return {'success': True, 'message': f'Registration cancelled for {student_name}'}
            else:
                return {'error': 'No active registration session to cancel'}
    
    def get_registration_status(self, student_name):
        """Get the status of a registration session"""
        with self.registration_lock:
            if student_name in self.active_registrations:
                session = self.active_registrations[student_name]
                return {
                    'active': True,
                    'captured_count': len(session['captured_samples']),
                    'max_samples': session['max_samples'],
                    'progress': (len(session['captured_samples']) / session['max_samples']) * 100,
                    'start_time': session['start_time']
                }
            else:
                return {'active': False}
    
    def process_frame_for_recognition(self, frame_data, confidence_threshold=0.7):
        """Process frame for face recognition"""
        try:
            if self.model is None:
                return {'error': 'No trained model available'}
            
            # Decode frame similar to registration
            if isinstance(frame_data, str):
                if frame_data.startswith('data:image'):
                    frame_data = frame_data.split(',')[1]
                image_bytes = base64.b64decode(frame_data)
            else:
                image_bytes = frame_data
            
            image = Image.open(io.BytesIO(image_bytes))
            image_np = np.array(image)
            if len(image_np.shape) == 3 and image_np.shape[2] == 3:
                image_bgr = cv.cvtColor(image_np, cv.COLOR_RGB2BGR)
            else:
                image_bgr = image_np
            
            # Detect faces
            gray = cv.cvtColor(image_bgr, cv.COLOR_BGR2GRAY)
            faces = self.haarcascade.detectMultiScale(gray, 1.3, 5)
            
            results = []
            for (x, y, w, h) in faces:
                if w < 80 or h < 80:
                    continue
                
                # Extract and process face
                face_roi = image_bgr[y:y+h, x:x+w]
                face_rgb = cv.cvtColor(face_roi, cv.COLOR_BGR2RGB)
                face_resized = cv.resize(face_rgb, (160, 160))
                face_expanded = np.expand_dims(face_resized, axis=0)
                
                # Get prediction
                embedding = self.facenet.embeddings(face_expanded)
                prediction = self.model.predict(embedding)
                confidence_scores = self.model.predict_proba(embedding)
                max_confidence = np.max(confidence_scores)
                
                if max_confidence >= confidence_threshold:
                    name = self.encoder.inverse_transform(prediction)[0]
                    results.append({
                        'recognized': True,
                        'name': name,
                        'confidence': float(max_confidence),
                        'bbox': [int(x), int(y), int(w), int(h)]
                    })
                else:
                    results.append({
                        'recognized': False,
                        'confidence': float(max_confidence),
                        'bbox': [int(x), int(y), int(w), int(h)]
                    })
            
            return {'faces': results}
            
        except Exception as e:
            print(f"❌ Error in face recognition: {e}")
            return {'error': f'Recognition error: {str(e)}'}

    def simple_register_student(self, name, student_id=None):
        """Simple registration for backward compatibility"""
        try:
            print(f"📸 Simple registration for {name}")
            
            # Create a basic student record
            student_data = {
                'name': name,
                'id': student_id or f"AUTO_{int(datetime.now().timestamp())}",
                'registered_at': datetime.now().isoformat(),
                'samples_count': 0,
                'last_updated': datetime.now().isoformat(),
                'average_quality': 0,
                'status': 'active'
            }
            
            # Check if student already exists
            existing_student_idx = None
            for i, student in enumerate(self.student_registry):
                if student['name'] == name:
                    existing_student_idx = i
                    break
            
            if existing_student_idx is not None:
                # Update existing student
                self.student_registry[existing_student_idx] = student_data
                print(f"✅ Updated existing student {name}")
            else:
                # Add new student
                self.student_registry.append(student_data)
                print(f"✅ Added new student {name}")
            
            # Save data
            self.save_all_data()
            
            return {
                'success': True,
                'message': f'Student {name} registered successfully',
                'student_data': student_data
            }
            
        except Exception as e:
            print(f"❌ Error in simple registration: {e}")
            return {'error': f'Registration failed: {str(e)}'}

# Flask Application with Enhanced Registration
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend-backend communication
face_system = EnhancedFaceRegistrationSystem()

@app.route('/')
def serve_frontend():
    """Serve the frontend index.html file from the project root."""
    return send_from_directory('.', 'index.html')

@app.route('/api/registration/start', methods=['POST'])
def start_registration():
    """Start a new student registration session"""
    try:
        data = request.get_json()
        if not data or 'name' not in data:
            return jsonify({'error': 'Student name is required'}), 400
        
        student_name = data['name'].strip()
        student_id = data.get('id', None)
        max_samples = int(data.get('max_samples', 20))
        
        if not student_name:
            return jsonify({'error': 'Student name cannot be empty'}), 400
        
        if max_samples < 5 or max_samples > 50:
            return jsonify({'error': 'Max samples must be between 5 and 50'}), 400
        
        result = face_system.start_registration(student_name, student_id, max_samples)
        
        if 'error' in result:
            return jsonify(result), 400
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/registration/process', methods=['POST'])
def process_registration_frame():
    """Process a frame for student registration"""
    try:
        data = request.get_json()
        if not data or 'student_name' not in data or 'frame' not in data:
            return jsonify({'error': 'Student name and frame data required'}), 400
        
        student_name = data['student_name']
        frame_data = data['frame']
        quality_threshold = float(data.get('quality_threshold', 70))
        
        result = face_system.process_registration_frame(student_name, frame_data, quality_threshold)
        
        if 'error' in result:
            return jsonify(result), 400
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/registration/cancel', methods=['POST'])
def cancel_registration():
    """Cancel an active registration session"""
    try:
        data = request.get_json()
        if not data or 'student_name' not in data:
            return jsonify({'error': 'Student name is required'}), 400
        
        student_name = data['student_name']
        result = face_system.cancel_registration(student_name)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/registration/status/<student_name>', methods=['GET'])
def get_registration_status(student_name):
    """Get registration status for a student"""
    try:
        result = face_system.get_registration_status(student_name)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Enhanced API for backward compatibility
@app.route('/api/register', methods=['POST'])
def simple_register():
    """Simple registration endpoint for backward compatibility"""
    try:
        data = request.get_json()
        if not data or 'name' not in data:
            return jsonify({'error': 'Student name is required'}), 400
        
        name = data['name'].strip()
        student_id = data.get('id', None)
        
        if not name:
            return jsonify({'error': 'Student name cannot be empty'}), 400
        
        result = face_system.simple_register_student(name, student_id)
        
        if 'error' in result:
            return jsonify(result), 400
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/students', methods=['GET'])
def get_students():
    """Get all registered students"""
    try:
        return jsonify({
            'students': face_system.student_registry,
            'total': len(face_system.student_registry)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/recognize', methods=['POST'])
def recognize_face():
    """Process frame for face recognition"""
    try:
        data = request.get_json()
        if not data or 'frame' not in data:
            return jsonify({'error': 'Frame data required'}), 400
        
        frame_data = data['frame']
        confidence_threshold = float(data.get('confidence', 0.7))
        
        result = face_system.process_frame_for_recognition(frame_data, confidence_threshold)
        
        if 'error' in result:
            return jsonify(result), 400
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/attendance', methods=['GET', 'POST'])
def handle_attendance():
    """Handle attendance operations"""
    if request.method == 'GET':
        # Get attendance records
        try:
            date_filter = request.args.get('date')
            if date_filter:
                filtered_data = [
                    record for record in face_system.attendance_data 
                    if record.get('date') == date_filter
                ]
                return jsonify({
                    'attendance': filtered_data,
                    'total': len(filtered_data),
                    'date': date_filter
                })
            else:
                return jsonify({
                    'attendance': face_system.attendance_data,
                    'total': len(face_system.attendance_data)
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
            marked_by = data.get('marked_by', 'face_recognition')
            current_date = date.today().isoformat()
            current_time = datetime.now()
            
            # Check if already marked today
            existing_record = next((
                record for record in face_system.attendance_data 
                if record['name'] == student_name and record['date'] == current_date
            ), None)
            
            if existing_record:
                return jsonify({
                    'success': True,
                    'already_present': True,
                    'message': f'{student_name} already marked present today',
                    'existing_record': existing_record
                })
            
            # Create new attendance record
            attendance_record = {
                'name': student_name,
                'date': current_date,
                'time': current_time.strftime('%H:%M:%S'),
                'timestamp': current_time.isoformat(),
                'marked_by': marked_by
            }
            
            face_system.attendance_data.insert(0, attendance_record)
            face_system.save_all_data()
            
            return jsonify({
                'success': True,
                'already_present': False,
                'message': f'{student_name} marked present',
                'record': attendance_record
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500

@app.route('/api/export', methods=['GET'])
def export_attendance():
    """Export attendance data as CSV"""
    try:
        # Create CSV content
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow(['Name', 'Date', 'Time', 'Method', 'Timestamp'])
        
        # Write data
        for record in face_system.attendance_data:
            writer.writerow([
                record.get('name', ''),
                record.get('date', ''),
                record.get('time', ''),
                record.get('marked_by', 'face_recognition'),
                record.get('timestamp', '')
            ])
        
        # Prepare response
        output.seek(0)
        response = make_response(output.getvalue())
        response.headers['Content-Type'] = 'text/csv'
        response.headers['Content-Disposition'] = f'attachment; filename=attendance_{date.today().isoformat()}.csv'
        
        return response
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/system/status', methods=['GET'])
def get_system_status():
    """Get enhanced system status"""
    try:
        today = date.today().isoformat()
        today_attendance = [r for r in face_system.attendance_data if r.get('date') == today]
        
        return jsonify({
            'system': 'online',
            'models_loaded': face_system.model is not None,
            'facenet_loaded': face_system.facenet is not None,
            'cascade_loaded': face_system.haarcascade is not None,
            'registered_students': len(face_system.student_registry),
            'total_samples': len(face_system.X) if len(face_system.X) > 0 else 0,
            'active_registrations': len(face_system.active_registrations),
            'total_records': len(face_system.attendance_data),
            'today_attendance': len(today_attendance),
            'attendance_rate': round((len(today_attendance) / len(face_system.student_registry) * 100) if len(face_system.student_registry) > 0 else 0, 1),
            'version': '2.0_enhanced',
            'last_updated': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def get_statistics():
    """Get detailed statistics for dashboard"""
    try:
        today = date.today().isoformat()
        today_attendance = [r for r in face_system.attendance_data if r.get('date') == today]
        
        # Weekly stats
        from datetime import timedelta
        week_start = datetime.now() - timedelta(days=7)
        week_attendance = [
            r for r in face_system.attendance_data 
            if datetime.fromisoformat(r.get('timestamp', r.get('date', ''))) >= week_start
        ]
        
        # Monthly stats
        month_start = datetime.now() - timedelta(days=30)
        month_attendance = [
            r for r in face_system.attendance_data 
            if datetime.fromisoformat(r.get('timestamp', r.get('date', ''))) >= month_start
        ]
        
        # Student-wise stats
        student_stats = []
        for student in face_system.student_registry:
            student_records = [r for r in face_system.attendance_data if r['name'] == student['name']]
            student_today = [r for r in today_attendance if r['name'] == student['name']]
            
            student_stats.append({
                'name': student['name'],
                'id': student['id'],
                'total_records': len(student_records),
                'present_today': len(student_today) > 0,
                'last_attendance': student_records[0]['date'] if student_records else None
            })
        
        return jsonify({
            'today': {
                'present': len(today_attendance),
                'total_students': len(face_system.student_registry),
                'rate': round((len(today_attendance) / len(face_system.student_registry) * 100) if len(face_system.student_registry) > 0 else 0, 1)
            },
            'week': {
                'total_records': len(week_attendance),
                'unique_students': len(set(r['name'] for r in week_attendance))
            },
            'month': {
                'total_records': len(month_attendance),
                'unique_students': len(set(r['name'] for r in month_attendance))
            },
            'students': student_stats,
            'total_records': len(face_system.attendance_data)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/attendance/today/absent', methods=['GET'])
def get_today_absent():
    """Get list of students absent today"""
    try:
        today = date.today().isoformat()
        today_attendance = [r for r in face_system.attendance_data if r.get('date') == today]
        present_students = set(r['name'] for r in today_attendance)
        
        absent_students = [
            student for student in face_system.student_registry 
            if student['name'] not in present_students
        ]
        
        return jsonify({
            'absent_students': absent_students,
            'count': len(absent_students),
            'date': today
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/attendance/clear/today', methods=['DELETE'])
def clear_today_attendance():
    """Clear today's attendance records"""
    try:
        today = date.today().isoformat()
        original_count = len(face_system.attendance_data)
        
        # Filter out today's records
        face_system.attendance_data = [
            record for record in face_system.attendance_data 
            if record.get('date') != today
        ]
        
        cleared_count = original_count - len(face_system.attendance_data)
        face_system.save_all_data()
        
        return jsonify({
            'success': True,
            'cleared_count': cleared_count,
            'remaining_count': len(face_system.attendance_data),
            'date': today
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/students/<student_name>', methods=['DELETE'])
def delete_student(student_name):
    """Delete a student and their records"""
    try:
        # Remove from student registry
        original_count = len(face_system.student_registry)
        face_system.student_registry = [
            student for student in face_system.student_registry 
            if student['name'] != student_name
        ]
        
        if len(face_system.student_registry) == original_count:
            return jsonify({'error': 'Student not found'}), 404
        
        # Remove from embeddings and retrain model if needed
        if len(face_system.Y) > 0:
            mask = face_system.Y != student_name
            face_system.X = face_system.X[mask] if len(face_system.X) > 0 else np.array([])
            face_system.Y = face_system.Y[mask]
            
            # Retrain model if we still have data
            if len(face_system.Y) > 0:
                face_system.encoder = LabelEncoder()
                face_system.encoder.fit(face_system.Y)
                face_system.model = SVC(kernel='linear', probability=True, random_state=42)
                face_system.model.fit(face_system.X, face_system.Y)
            else:
                face_system.model = None
        
        face_system.save_all_data()
        
        return jsonify({
            'success': True,
            'message': f'Student {student_name} deleted successfully',
            'remaining_students': len(face_system.student_registry)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/backup', methods=['POST'])
def create_backup():
    """Create a backup of all data"""
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_data = {
            'timestamp': timestamp,
            'students': face_system.student_registry,
            'attendance': face_system.attendance_data,
            'system_info': {
                'total_students': len(face_system.student_registry),
                'total_records': len(face_system.attendance_data),
                'total_samples': len(face_system.X) if len(face_system.X) > 0 else 0
            }
        }
        
        backup_filename = f'facetrack_backup_{timestamp}.json'
        with open(backup_filename, 'w') as f:
            json.dump(backup_data, f, indent=2)
        
        return jsonify({
            'success': True,
            'backup_file': backup_filename,
            'timestamp': timestamp,
            'records_backed_up': len(face_system.attendance_data)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'system': {
                'backend': 'online',
                'face_detection': face_system.haarcascade is not None,
                'face_recognition': face_system.facenet is not None,
                'model_trained': face_system.model is not None,
                'data_files': {
                    'embeddings': os.path.exists('faces_embeddings_done_35classes.npz'),
                    'model': os.path.exists('svm_model_160x160.pkl'),
                    'registry': os.path.exists('student_registry.json'),
                    'attendance': os.path.exists('attendance_data.json')
                }
            }
        })
    except Exception as e:
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(400)
def bad_request(error):
    return jsonify({'error': 'Bad request'}), 400

# Additional utility endpoints for dashboard
@app.route('/api/dashboard/summary', methods=['GET'])
def get_dashboard_summary():
    """Get summary data for dashboard"""
    try:
        today = date.today().isoformat()
        today_attendance = [r for r in face_system.attendance_data if r.get('date') == today]
        
        # Recent activity (last 10 records)
        recent_activity = sorted(
            face_system.attendance_data[:10], 
            key=lambda x: x.get('timestamp', x.get('date', '')), 
            reverse=True
        )
        
        # Top students by attendance
        student_attendance_count = {}
        for record in face_system.attendance_data:
            name = record['name']
            student_attendance_count[name] = student_attendance_count.get(name, 0) + 1
        
        top_students = sorted(
            student_attendance_count.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]
        
        return jsonify({
            'today': {
                'present': len(today_attendance),
                'total': len(face_system.student_registry),
                'rate': round((len(today_attendance) / len(face_system.student_registry) * 100) if len(face_system.student_registry) > 0 else 0, 1)
            },
            'recent_activity': recent_activity,
            'top_students': [{'name': name, 'count': count} for name, count in top_students],
            'system_status': {
                'model_loaded': face_system.model is not None,
                'active_registrations': len(face_system.active_registrations),
                'total_records': len(face_system.attendance_data)
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting Enhanced FaceTrack Pro Backend...")
    print("📊 System Status:")
    print(f"   Students: {len(face_system.student_registry)}")
    print(f"   Face Samples: {len(face_system.X) if len(face_system.X) > 0 else 0}")
    print(f"   Attendance Records: {len(face_system.attendance_data)}")
    print(f"   Model Status: {'✅ Loaded' if face_system.model else '❌ Not trained'}")
    print("\n🌐 Server starting on http://localhost:5000")
    print("📱 Enhanced Features Available:")
    print("   • Advanced Registration System")
    print("   • Real-time Quality Assessment")
    print("   • Comprehensive Dashboard APIs")
    print("   • Data Export & Backup")
    print("   • System Health Monitoring")
    print("   • Enhanced Attendance Management")
    
    # Create necessary directories
    os.makedirs('backups', exist_ok=True)
    
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)