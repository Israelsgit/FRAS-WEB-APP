@echo off
echo Starting FaceTrack Pro Backend...
cd /d "%~dp0"
call venv\Scripts\activate
python app.py
pause
