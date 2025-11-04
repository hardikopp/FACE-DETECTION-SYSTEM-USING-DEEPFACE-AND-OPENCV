import os
import time
import threading
import json
import urllib.request
from flask import (
    Flask, render_template, request, redirect, url_for, session,
    send_from_directory, jsonify, flash, Response
)
import cv2
import numpy as np

from train_embeddings import train_with_progress
from recognize import load_embeddings, recognize_multiple_faces, recognize_face_single

app = Flask(__name__)
app.secret_key = "replace_with_random_secret_key"

# Folders
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("embeddings", exist_ok=True)

PROGRESS_FILE = "embeddings/progress.json"

# Admin
ADMIN_USER = "admin"
ADMIN_PASS = "1234"

# Default IP camera (prefilled in UI)
DEFAULT_CAMERA_URL = "http://10.240.66.42:8080/shot.jpg"


def write_progress(processed, total):
    os.makedirs(os.path.dirname(PROGRESS_FILE), exist_ok=True)
    with open(PROGRESS_FILE, "w") as f:
        json.dump({"processed": processed, "total": total}, f)


def read_progress():
    if not os.path.exists(PROGRESS_FILE):
        return {"processed": 0, "total": 0}
    with open(PROGRESS_FILE, "r") as f:
        return json.load(f)


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        if username == ADMIN_USER and password == ADMIN_PASS:
            session["admin"] = True
            flash("Logged in as admin", "success")
            return redirect(url_for("index"))
        flash("Invalid credentials", "danger")
        return redirect(url_for("login"))
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.pop("admin", None)
    return redirect(url_for("index"))


@app.route("/")
def index():
    return render_template("index.html", default_camera=DEFAULT_CAMERA_URL)


# TRAIN / RETRAIN
def retrain_background():
    def progress_cb(processed, total):
        write_progress(processed, total)

    write_progress(0, 0)
    count = train_with_progress(progress_callback=progress_cb)
    write_progress(count, count)
    return count


@app.route("/train", methods=["POST"])
def train():
    # Save the uploaded training image into dataset/<name>/
    name = request.form.get("name", "").strip()
    file = request.files.get("file")
    if not name or not file:
        flash("Please provide a name and an image for training.", "warning")
        return redirect(url_for("index"))

    person_folder = os.path.join("dataset", name)
    os.makedirs(person_folder, exist_ok=True)
    save_path = os.path.join(person_folder, file.filename)
    file.save(save_path)
    flash(
        f"Saved training image to {person_folder}. Now retrain the model (admin).", "success")
    return redirect(url_for("index"))


@app.route("/start_retrain", methods=["POST"])
def start_retrain():
    if not session.get("admin"):
        flash("Admin login required to retrain.", "danger")
        return redirect(url_for("login"))

    t = threading.Thread(target=retrain_background, daemon=True)
    t.start()
    return jsonify({"status": "started"})


@app.route("/progress")
def progress():
    return jsonify(read_progress())


# UPLOAD recognition (single static image)
@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("file")
    if not file:
        flash("No file uploaded.", "warning")
        return redirect(url_for("index"))

    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)
    frame = cv2.imread(path)

    data = load_embeddings()
    if not data.get("embeddings"):
        flash("No embeddings found. Retrain first.", "warning")
        return redirect(url_for("index"))

    # Recognize multiple faces and overlay boxes
    frame_out, results = recognize_multiple_faces(
        frame, data, target_name="Hardik")
    out_path = os.path.join("static", "output.jpg")
    cv2.imwrite(out_path, frame_out)

    if results:
        detected_names = ", ".join([r[0] for r in results])
    else:
        detected_names = "No faces detected"

    return render_template("result.html", image_url=url_for("static", filename="output.jpg"), result=detected_names)


# STREAM (one-shot capture redirected to live view OR snapshot)
@app.route("/stream", methods=["POST"])
def stream():
    ip_url = request.form.get("ip_url", "").strip()
    if not ip_url:
        flash("IP URL required", "warning")
        return redirect(url_for("index"))

    # Provide a live viewer page that will connect to /live_feed
    return render_template("live.html", ip_url=ip_url)


# LIVE_FEED streaming (multipart frame streaming)
@app.route("/live_feed")
def live_feed():
    url = request.args.get("ip_url")
    if not url:
        return "Camera URL required", 400

    def generate_frames():
        data = load_embeddings()
        while True:
            try:
                # Read image from IP webcam shot.jpg (works for shot.jpg, use /video for MJPEG if available)
                resp = urllib.request.urlopen(url, timeout=5)
                img_data = np.array(bytearray(resp.read()), dtype=np.uint8)
                frame = cv2.imdecode(img_data, cv2.IMREAD_COLOR)

                frame_out, _ = recognize_multiple_faces(
                    frame, data, target_name="Hardik")
                _, buffer = cv2.imencode(".jpg", frame_out)
                frame_bytes = buffer.tobytes()
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
            except Exception as e:
                # on error wait and retry (prevents server crash on connection errors)
                time.sleep(1)
                continue

    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


# static files route (Flask usually serves static, but keep explicit)
@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory("static", filename)


if __name__ == "__main__":
    import webbrowser
    import threading

    def open_browser():
        webbrowser.open_new("http://127.0.0.1:5000/")  # auto open homepage

    threading.Timer(1.5, open_browser).start()
    app.run(debug=True, use_reloader=False)
