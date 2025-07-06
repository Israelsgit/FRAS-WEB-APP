# FaceTrack Pro

AI-Powered Facial Recognition Attendance System

## 🚀 Features

- Real-time face recognition for attendance
- Student registration and management
- Adjustable recognition confidence
- Attendance records and CSV export
- Modern, responsive dashboard UI

## 🛠️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPO.git
cd YOUR-REPO
```

### 2. Backend Setup (Flask)

- Ensure you have Python 3.8+
- (Recommended) Create a virtual environment:
  ```bash
  python -m venv venv
  source venv/bin/activate  # On Windows: venv\Scripts\activate
  ```
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```
- Start the backend server:
  ```bash
  python app.py
  # or
  flask run --host=127.0.0.1 --port=5000
  ```

### 3. Frontend Setup (Static HTML/JS)

- Serve the frontend using a simple HTTP server:
  ```bash
  python -m http.server 3000
  ```
- Open your browser and go to: [http://127.0.0.1:3000](http://127.0.0.1:3000)

> **Note:** The frontend communicates with the backend at `http://127.0.0.1:5000` by default. Change the `BACKEND_URL` variable in `index.html` if your backend runs elsewhere.

## 🤝 Contributing

1. Fork this repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Make your changes
4. Commit and push (`git commit -am 'Add new feature' && git push origin feature-branch`)
5. Open a Pull Request

## 📄 License

MIT License
