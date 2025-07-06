#!/usr/bin/env python3
"""
FaceTrack Pro - Configuration Settings
Centralized configuration for the facial recognition attendance system
"""

import os
from datetime import timedelta

class Config:
    """Base configuration class"""
    
    # Flask Configuration
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'facetrack-pro-secret-key-2024'
    DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    # Server Configuration
    HOST = os.environ.get('FACETRACK_HOST', '0.0.0.0')
    PORT = int(os.environ.get('FACETRACK_PORT', 5000))
    THREADED = True
    
    # Face Recognition Configuration
    CONFIDENCE_THRESHOLD = float(os.environ.get('FACETRACK_CONFIDENCE_THRESHOLD', 0.7))
    MIN_FACE_SIZE = (
        int(os.environ.get('FACETRACK_MIN_FACE_WIDTH', 80)),
        int(os.environ.get('FACETRACK_MIN_FACE_HEIGHT', 80))
    )
    MAX_FACE_SIZE = (
        int(os.environ.get('FACETRACK_MAX_FACE_WIDTH', 400)),
        int(os.environ.get('FACETRACK_MAX_FACE_HEIGHT', 400))
    )
    
    # Model Paths
    FACENET_MODEL_PATH = os.environ.get('FACENET_MODEL_PATH', 'facenet_model')
    HAARCASCADE_PATH = os.environ.get('HAARCASCADE_PATH', 'haarcascade_frontalface_default.xml')
    SVM_MODEL_PATH = os.environ.get('SVM_MODEL_PATH', 'svm_model_160x160.pkl')
    EMBEDDINGS_PATH = os.environ.get('EMBEDDINGS_PATH', 'faces_embeddings_done_35classes.npz')
    
    # Data Storage
    STUDENT_REGISTRY_FILE = os.environ.get('STUDENT_REGISTRY_FILE', 'student_registry.json')
    ATTENDANCE_DATA_FILE = os.environ.get('ATTENDANCE_DATA_FILE', 'attendance_data.json')
    
    # Registration Settings
    MIN_REGISTRATION_SAMPLES = int(os.environ.get('MIN_REGISTRATION_SAMPLES', 5))
    MAX_REGISTRATION_SAMPLES = int(os.environ.get('MAX_REGISTRATION_SAMPLES', 20))
    REGISTRATION_TIMEOUT = int(os.environ.get('REGISTRATION_TIMEOUT', 60))  # seconds
    
    # Recognition Settings
    RECOGNITION_INTERVAL = float(os.environ.get('RECOGNITION_INTERVAL', 2.0))  # seconds
    MAX_DAILY_ATTENDANCE = int(os.environ.get('MAX_DAILY_ATTENDANCE', 1))  # entries per day
    
    # Image Processing
    IMAGE_QUALITY = float(os.environ.get('IMAGE_QUALITY', 0.8))  # JPEG quality for processing
    MAX_IMAGE_SIZE = int(os.environ.get('MAX_IMAGE_SIZE', 1024))  # pixels
    
    # Security Settings
    CORS_ORIGINS = os.environ.get('CORS_ORIGINS', '*').split(',')
    MAX_CONTENT_LENGTH = int(os.environ.get('MAX_CONTENT_LENGTH', 16 * 1024 * 1024))  # 16MB
    
    # Logging Configuration
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FILE = os.environ.get('LOG_FILE', 'facetrack.log')
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Performance Settings
    TENSORFLOW_THREADS = int(os.environ.get('TENSORFLOW_THREADS', 4))
    OPENCV_THREADS = int(os.environ.get('OPENCV_THREADS', 4))
    
    # Backup Settings
    AUTO_BACKUP = os.environ.get('AUTO_BACKUP', 'True').lower() == 'true'
    BACKUP_INTERVAL = int(os.environ.get('BACKUP_INTERVAL', 24))  # hours
    BACKUP_RETENTION = int(os.environ.get('BACKUP_RETENTION', 7))  # days
    
class DevelopmentConfig(Config):
    """Development environment configuration"""
    DEBUG = True
    LOG_LEVEL = 'DEBUG'
    CONFIDENCE_THRESHOLD = 0.6  # Lower threshold for testing
    MIN_REGISTRATION_SAMPLES = 3  # Faster registration for testing

class ProductionConfig(Config):
    """Production environment configuration"""
    DEBUG = False
    LOG_LEVEL = 'WARNING'
    CONFIDENCE_THRESHOLD = 0.75  # Higher threshold for production
    MIN_REGISTRATION_SAMPLES = 10  # More samples for better accuracy
    
    # Security enhancements for production
    SECRET_KEY = os.environ.get('SECRET_KEY') or os.urandom(32).hex()
    CORS_ORIGINS = os.environ.get('CORS_ORIGINS', 'http://localhost:3000').split(',')

class TestingConfig(Config):
    """Testing environment configuration"""
    TESTING = True
    DEBUG = True
    CONFIDENCE_THRESHOLD = 0.5  # Very low threshold for testing
    MIN_REGISTRATION_SAMPLES = 1  # Minimal samples for testing
    STUDENT_REGISTRY_FILE = 'test_student_registry.json'
    ATTENDANCE_DATA_FILE = 'test_attendance_data.json'

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}

def get_config():
    """Get configuration based on environment"""
    env = os.environ.get('FLASK_ENV', 'development')
    return config.get(env, config['default'])

# Camera Configuration
class CameraConfig:
    """Camera-specific configuration"""
    
    # Default camera settings
    DEFAULT_CAMERA_INDEX = int(os.environ.get('CAMERA_INDEX', 0))
    DEFAULT_WIDTH = int(os.environ.get('CAMERA_WIDTH', 1280))
    DEFAULT_HEIGHT = int(os.environ.get('CAMERA_HEIGHT', 720))
    DEFAULT_FPS = int(os.environ.get('CAMERA_FPS', 30))
    
    # Video codec preferences
    PREFERRED_CODECS = ['MJPG', 'YUYV', 'H264']
    
    # Auto-adjustment settings
    AUTO_EXPOSURE = os.environ.get('AUTO_EXPOSURE', 'True').lower() == 'true'
    AUTO_WHITE_BALANCE = os.environ.get('AUTO_WHITE_BALANCE', 'True').lower() == 'true'
    AUTO_FOCUS = os.environ.get('AUTO_FOCUS', 'True').lower() == 'true'

# AI Model Configuration
class AIConfig:
    """AI model-specific configuration"""
    
    # TensorFlow settings
    TF_GPU_MEMORY_GROWTH = os.environ.get('TF_GPU_MEMORY_GROWTH', 'True').lower() == 'true'
    TF_XLA_FLAGS = os.environ.get('TF_XLA_FLAGS', '--tf_xla_enable_xla_devices')
    
    # FaceNet settings
    FACENET_INPUT_SIZE = (160, 160)
    FACENET_EMBEDDING_SIZE = 512
    
    # SVM settings
    SVM_KERNEL = os.environ.get('SVM_KERNEL', 'linear')
    SVM_PROBABILITY = True
    SVM_C = float(os.environ.get('SVM_C', 1.0))
    
    # Face detection settings
    HAAR_SCALE_FACTOR = float(os.environ.get('HAAR_SCALE_FACTOR', 1.3))
    HAAR_MIN_NEIGHBORS = int(os.environ.get('HAAR_MIN_NEIGHBORS', 5))
    
    # Data augmentation for registration
    AUGMENT_DATA = os.environ.get('AUGMENT_DATA', 'True').lower() == 'true'
    AUGMENTATION_FACTOR = int(os.environ.get('AUGMENTATION_FACTOR', 2))

# UI Configuration
class UIConfig:
    """User interface configuration"""
    
    # Theme settings
    THEME = os.environ.get('UI_THEME', 'dark')
    PRIMARY_COLOR = os.environ.get('UI_PRIMARY_COLOR', '#238636')
    
    # Display settings
    SHOW_CONFIDENCE = os.environ.get('SHOW_CONFIDENCE', 'True').lower() == 'true'
    SHOW_BOUNDING_BOXES = os.environ.get('SHOW_BOUNDING_BOXES', 'True').lower() == 'true'
    ANIMATION_SPEED = int(os.environ.get('ANIMATION_SPEED', 300))  # milliseconds
    
    # Notification settings
    NOTIFICATION_DURATION = int(os.environ.get('NOTIFICATION_DURATION', 3000))  # milliseconds
    SOUND_ENABLED = os.environ.get('SOUND_ENABLED', 'False').lower() == 'true'

# Database Configuration (for future use)
class DatabaseConfig:
    """Database configuration for advanced deployments"""
    
    # SQLite (default)
    SQLITE_DB_PATH = os.environ.get('SQLITE_DB_PATH', 'facetrack.db')
    
    # PostgreSQL (optional)
    POSTGRES_HOST = os.environ.get('POSTGRES_HOST', 'localhost')
    POSTGRES_PORT = int(os.environ.get('POSTGRES_PORT', 5432))
    POSTGRES_DB = os.environ.get('POSTGRES_DB', 'facetrack')
    POSTGRES_USER = os.environ.get('POSTGRES_USER', 'facetrack')
    POSTGRES_PASSWORD = os.environ.get('POSTGRES_PASSWORD', '')
    
    # MySQL (optional)
    MYSQL_HOST = os.environ.get('MYSQL_HOST', 'localhost')
    MYSQL_PORT = int(os.environ.get('MYSQL_PORT', 3306))
    MYSQL_DB = os.environ.get('MYSQL_DB', 'facetrack')
    MYSQL_USER = os.environ.get('MYSQL_USER', 'facetrack')
    MYSQL_PASSWORD = os.environ.get('MYSQL_PASSWORD', '')

# Export all configurations
__all__ = [
    'Config',
    'DevelopmentConfig', 
    'ProductionConfig',
    'TestingConfig',
    'CameraConfig',
    'AIConfig',
    'UIConfig',
    'DatabaseConfig',
    'get_config'
]

# Usage example:
if __name__ == '__main__':
    # Print current configuration
    current_config = get_config()
    print(f"Current environment: {os.environ.get('FLASK_ENV', 'development')}")
    print(f"Debug mode: {current_config.DEBUG}")
    print(f"Confidence threshold: {current_config.CONFIDENCE_THRESHOLD}")
    print(f"Host: {current_config.HOST}:{current_config.PORT}")
    print(f"Min face size: {current_config.MIN_FACE_SIZE}")
    print(f"Registration samples: {current_config.MIN_REGISTRATION_SAMPLES}-{current_config.MAX_REGISTRATION_SAMPLES}")
