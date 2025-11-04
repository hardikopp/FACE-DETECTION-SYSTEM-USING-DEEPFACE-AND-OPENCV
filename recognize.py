import os
import pickle
import numpy as np
from deepface import DeepFace
from scipy.spatial.distance import cosine
import cv2

EMB_PATH = os.path.join("embeddings", "embeddings.pkl")


def load_embeddings():
    if not os.path.exists(EMB_PATH):
        # initial empty structure
        return {"names": [], "embeddings": []}
    with open(EMB_PATH, "rb") as f:
        data = pickle.load(f)
    return data


def save_embeddings(data):
    os.makedirs(os.path.dirname(EMB_PATH), exist_ok=True)
    with open(EMB_PATH, "wb") as f:
        pickle.dump(data, f)


def get_embedding_from_face(face_roi):
    """
    face_roi: BGR image (numpy array) cropped to face
    returns: 1D embedding array or None
    """
    try:
        rep = DeepFace.represent(
            face_roi, model_name="Facenet", enforce_detection=False)
        if rep and isinstance(rep, list) and "embedding" in rep[0]:
            return np.array(rep[0]["embedding"])
        # older versions: rep[0]['embedding'] or rep[0]
        if rep and isinstance(rep, list) and isinstance(rep[0], (list, tuple, np.ndarray)):
            return np.array(rep[0])
    except Exception as e:
        print("Embedding error:", e)
    return None


def recognize_face_single(img, data, threshold=0.35):
    """
    Recognize a single whole-frame face (uses whole image embedding)
    returns (name or None, score)
    """
    emb = get_embedding_from_face(img)
    if emb is None or not data.get("embeddings"):
        return None, 1.0

    best_score = 1.0
    best_name = None
    for name, ref_emb in zip(data["names"], data["embeddings"]):
        score = cosine(emb, ref_emb)
        if score < best_score:
            best_score = score
            best_name = name

    if best_score < threshold:
        return best_name, best_score
    return None, best_score


def recognize_multiple_faces(frame, data, target_name="Hardik", threshold=0.35):
    """
    Detect faces (Haar cascade), compute embedding per face, compare to all stored embeddings,
    draw boxes on frame, and return (annotated_frame, results_list)
    results_list -> list of tuples (label, (x,y,w,h))
    """
    if frame is None:
        return frame, []

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    results = []
    for (x, y, w, h) in faces:
        x1, y1, x2, y2 = x, y, x + w, y + h
        face_roi = frame[y1:y2, x1:x2]
        # Ensure face ROI is not empty
        if face_roi.size == 0:
            continue

        emb = get_embedding_from_face(face_roi)
        if emb is None:
            label = "NoEmb"
            color = (0, 0, 255)
            results.append((label, (x, y, w, h)))
            continue

        best_score = 1.0
        best_name = None
        for name, ref_emb in zip(data.get("names", []), data.get("embeddings", [])):
            try:
                score = cosine(emb, ref_emb)
            except Exception:
                # fallback to np.linalg
                score = np.linalg.norm(emb - ref_emb)
            if score < best_score:
                best_score = score
                best_name = name

        if best_score < threshold:
            if best_name and best_name.lower() == target_name.lower():
                label = f"Hardik ({best_score:.3f})"
                color = (0, 255, 0)  # green
                thickness = 3
            else:
                label = f"{best_name} ({best_score:.3f})"
                color = (255, 215, 0)  # gold
                thickness = 2
        else:
            label = f"Unknown ({best_score:.3f})"
            color = (0, 0, 255)
            thickness = 2

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(frame, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        results.append((label, (x, y, w, h)))

    return frame, results
