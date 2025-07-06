#!/bin/bash
echo "Starting FaceTrack Pro Backend..."
cd "$(dirname "$0")"
source venv/Scripts/activate
python app.py
