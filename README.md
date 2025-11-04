# FACE-DETECTION-SYSTEM-USING-DEEPFACE-AND-OPENCV
A full-stack AI-powered Face Detection System built with Python, DeepFace, and OpenCV. Supports real-time video analysis, user registration, and attendance management via a Flask web interface.
# 👁️ Face Detection System using DeepFace and OpenCV

A real-time **Face Detection and Recognition System** built with **Python, DeepFace, and OpenCV**.  
This project captures live video, detects faces, and identifies individuals using pre-trained deep learning models — with options to register new users and manage attendance records through a web interface.

---

## 🚀 Features

- 🎥 Real-time face detection using **OpenCV**  
- 🧠 Face recognition powered by **DeepFace** (supports multiple models like VGG-Face, Facenet, OpenFace, DeepID, etc.)  
- 🗂️ User registration and dataset management  
- 📅 Attendance tracking with date and time  
- 🔐 Admin panel (via Flask) for managing users and viewing records  
- 🌐 Web-based interface with camera integration  
- 🧩 Flexible backend — can use local webcam or IP camera stream  

---

## 🧰 Tech Stack

| Component | Technology |
|------------|-------------|
| **Frontend** | HTML, CSS, Bootstrap |
| **Backend** | Python (Flask Framework) |
| **AI/ML** | DeepFace, OpenCV |
| **Database** | SQLite / MySQL (configurable) |
| **Tools** | NumPy, Pandas, datetime, Flask-Admin |

---

## 🗂️ Folder Structure

project/
│
├── app.py # Main Flask application
├── static/ # CSS, JS, and images
├── templates/ # HTML templates
├── face_data/ # Stored face images
├── models/ # Model files if any
├── database/ # Attendance and user data
└── README.md


Create and activate virtual environment

python -m venv venv
venv\Scripts\activate   # On Windows
source venv/bin/activate   # On Linux/Mac


Install dependencies

pip install -r requirements.txt

Run the application
python app.py


Access in browser
http://127.0.0.1:5000

Sample Demo

<img width="1920" height="1080" alt="Screenshot 2025-10-30 215949" src="https://github.com/user-attachments/assets/de536999-0bd8-4d22-ac76-5c1fc2bb297e" />

<img width="1069" height="742" alt="Screenshot 2025-09-18 231044" src="https://github.com/user-attachments/assets/920e0be7-6816-467e-b641-17a3d4b8b737" />

🧪 Supported DeepFace Models
VGG-Face

Facenet

OpenFace

DeepFace

DeepID

You can choose your model in the configuration file for accuracy/performance balance.


🧑‍💻 Future Improvements

Integration with cloud database

Enhanced UI/UX with React

Face mask detection

Emotion analysis and live analytics dashboard


📄 License

This project is licensed under the MIT License .


👤 Author

Hardik Kumar Bihari
🎓 B.Tech in Computer Science and Engineering
📍 Trident Academy of Technology, Bhubaneswar
📧 hardikkumarbihari@gmail.com



