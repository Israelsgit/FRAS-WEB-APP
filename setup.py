#!/usr/bin/env python3
"""
FaceTrack Pro Setup Script
Automated setup for the facial recognition attendance system
"""

import os
import sys
import subprocess
import shutil
import urllib.request
import zipfile
from pathlib import Path

def print_header(text):
    """Print a formatted header"""
    print("\n" + "="*60)
    print(f"FaceTrack Pro Setup: {text}")
    print("="*60)

def print_step(step, description):
    """Print a setup step"""
    print(f"\n{step}. {description}")
    print("-" * 40)

def run_command(command, description=""):
    """Run a shell command and handle errors"""
    print(f"Running: {command}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: {e}")
        if e.stderr:
            print(f"Error details: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print("ERROR: Python 3.7 or higher is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    
    print(f"SUCCESS: Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def create_virtual_environment():
    """Create and activate virtual environment"""
    print("Creating virtual environment...")
    
    if os.path.exists("venv"):
        print("Virtual environment already exists")
        return True
    
    if run_command("python -m venv venv"):
        print("SUCCESS: Virtual environment created successfully")
        return True
    else:
        print("ERROR: Failed to create virtual environment")
        return False

def install_dependencies():
    """Install required Python packages"""
    print("Installing dependencies...")
    
    # Determine pip command based on OS
    if os.name == 'nt':  # Windows
        pip_cmd = "venv\\Scripts\\pip"
    else:  # Unix/Linux/macOS
        pip_cmd = "venv/bin/pip"
    
    # Upgrade pip first
    if not run_command(f"{pip_cmd} install --upgrade pip"):
        print("ERROR: Failed to upgrade pip")
        return False
    
    # Install requirements
    if not run_command(f"{pip_cmd} install -r requirements.txt"):
        print("ERROR: Failed to install requirements")
        return False
    
    print("SUCCESS: Dependencies installed successfully")
    return True

def download_cascade_file():
    """Download Haar cascade file if not present"""
    cascade_file = "haarcascade_frontalface_default.xml"
    
    if os.path.exists(cascade_file):
        print(f"SUCCESS: {cascade_file} already exists")
        return True
    
    print(f"Downloading {cascade_file}...")
    url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
    
    try:
        urllib.request.urlretrieve(url, cascade_file)
        print(f"SUCCESS: Downloaded {cascade_file}")
        return True
    except Exception as e:
        print(f"ERROR: Failed to download {cascade_file}: {e}")
        return False

def create_data_files():
    """Create necessary data files if they don't exist"""
    data_files = [
        ("student_registry.json", "[]"),
        ("attendance_data.json", "[]")
    ]
    
    for filename, default_content in data_files:
        if not os.path.exists(filename):
            try:
                with open(filename, 'w') as f:
                    f.write(default_content)
                print(f"SUCCESS: Created {filename}")
            except Exception as e:
                print(f"ERROR: Failed to create {filename}: {e}")
                return False
        else:
            print(f"SUCCESS: {filename} already exists")
    
    return True

def copy_model_files():
    """Copy existing model files if they exist in the original project"""
    model_files = [
        "faces_embeddings_done_35classes.npz",
        "svm_model_160x160.pkl"
    ]
    
    for filename in model_files:
        if os.path.exists(filename):
            print(f"SUCCESS: {filename} already exists")
        else:
            print(f"WARNING: {filename} not found - you'll need to register students first")
    
    return True

def create_startup_scripts():
    """Create startup scripts for different platforms"""
    
    # Windows batch file
    windows_script = """@echo off
echo Starting FaceTrack Pro Backend...
cd /d "%~dp0"
call venv\\Scripts\\activate
python app.py
pause
"""
    
    # Unix shell script
    unix_script = """#!/bin/bash
echo "Starting FaceTrack Pro Backend..."
cd "$(dirname "$0")"
source venv/bin/activate
python app.py
"""
    
    try:
        # Create Windows script
        with open("start_backend.bat", 'w') as f:
            f.write(windows_script)
        print("SUCCESS: Created start_backend.bat")
        
        # Create Unix script
        with open("start_backend.sh", 'w') as f:
            f.write(unix_script)
        
        # Make Unix script executable
        if os.name != 'nt':
            os.chmod("start_backend.sh", 0o755)
        
        print("SUCCESS: Created start_backend.sh")
        return True
        
    except Exception as e:
        print(f"ERROR: Failed to create startup scripts: {e}")
        return False

def create_html_launcher():
    """Create a simple HTML file to launch the web app"""
    html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FaceTrack Pro Launcher</title>
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
        .btn {
            display: inline-block;
            padding: 12px 24px;
            background: #238636;
            color: white;
            text-decoration: none;
            border-radius: 6px;
            margin: 10px;
            font-weight: 500;
            transition: background 0.2s;
        }
        .btn:hover {
            background: #2ea043;
        }
        .status {
            padding: 20px;
            background: #21262d;
            border-radius: 8px;
            margin: 20px 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>FaceTrack Pro</h1>
        <p>AI-Powered Facial Recognition Attendance System</p>
        
        <div class="status">
            <h3>Quick Start</h3>
            <p>1. Make sure the backend server is running</p>
            <p>2. Click the button below to open the web application</p>
        </div>
        
        <a href="index.html" class="btn">Launch Web App</a>
        <a href="http://localhost:5000" class="btn">Backend Status</a>
        
        <div class="status">
            <h3>Instructions</h3>
            <p>• Run start_backend.bat (Windows) or ./start_backend.sh (Unix/Linux)</p>
            <p>• Open this HTML file in your browser</p>
            <p>• Register students and start tracking attendance</p>
        </div>
    </div>
</body>
</html>'''
    
    try:
        with open("launcher.html", 'w') as f:
            f.write(html_content)
        print("SUCCESS: Created launcher.html")
        return True
    except Exception as e:
        print(f"ERROR: Failed to create launcher.html: {e}")
        return False

def main():
    """Main setup function"""
    print_header("FaceTrack Pro Setup")
    print("Welcome to the FaceTrack Pro setup wizard!")
    print("This will set up the facial recognition attendance system.")
    
    # Step 1: Check Python version
    print_step(1, "Checking Python version")
    if not check_python_version():
        sys.exit(1)
    
    # Step 2: Create virtual environment
    print_step(2, "Creating virtual environment")
    if not create_virtual_environment():
        sys.exit(1)
    
    # Step 3: Install dependencies
    print_step(3, "Installing dependencies")
    if not install_dependencies():
        sys.exit(1)
    
    # Step 4: Download required files
    print_step(4, "Downloading required files")
    if not download_cascade_file():
        sys.exit(1)
    
    # Step 5: Create data files
    print_step(5, "Creating data files")
    if not create_data_files():
        sys.exit(1)
    
    # Step 6: Check model files
    print_step(6, "Checking model files")
    copy_model_files()
    
    # Step 7: Create startup scripts
    print_step(7, "Creating startup scripts")
    if not create_startup_scripts():
        sys.exit(1)
    
    # Step 8: Create HTML launcher
    print_step(8, "Creating HTML launcher")
    if not create_html_launcher():
        sys.exit(1)
    
    # Final instructions
    print_header("Setup Complete!")
    print("SUCCESS: FaceTrack Pro has been set up successfully!")
    print("\nTo start the system:")
    
    if os.name == 'nt':  # Windows
        print("   • Run: start_backend.bat")
    else:  # Unix/Linux/macOS
        print("   • Run: ./start_backend.sh")
    
    print("   • Open launcher.html in your browser")
    print("   • Or open index.html directly")
    
    print("\nNext steps:")
    print("   1. Start the backend server")
    print("   2. Open the web application")
    print("   3. Register students using the camera")
    print("   4. Begin tracking attendance")
    
    print("\nTips:")
    print("   • Ensure good lighting for better recognition")
    print("   • Register students with multiple face angles")
    print("   • Adjust confidence threshold as needed")
    
    print("\nTroubleshooting:")
    print("   • Check requirements.txt for missing dependencies")
    print("   • Ensure camera permissions are granted")
    print("   • Verify model files are present for recognition")

if __name__ == "__main__":
    main()