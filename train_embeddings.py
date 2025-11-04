import os
import pickle
import time
import numpy as np
from deepface import DeepFace
from recognize import save_embeddings

DATASET_DIR = r"face_recog_flask_advanced 2\dataset"
EMB_FILE = os.path.join("embeddings", "embeddings.pkl")


def _face_embedding_from_image(img_path):
    try:
        # read image as BGR
        import cv2
        img = cv2.imread(img_path)
        rep = DeepFace.represent(
            img, model_name="Facenet", enforce_detection=False)
        if rep and isinstance(rep, list):
            emb = rep[0].get("embedding") if isinstance(
                rep[0], dict) else rep[0]
            return np.array(emb)
    except Exception as e:
        print(f"Embedding error for {img_path}: {e}")
    return None


def train_with_progress(progress_callback=None):
    """
    Walks through dataset/<PersonName>/*, compute embeddings and save embeddings.pkl
    progress_callback(processed_count, total_count) optional.
    Returns number of embeddings saved.
    """
    people = []
    # collect all image paths
    entries = []
    for person in sorted(os.listdir(DATASET_DIR) if os.path.exists(DATASET_DIR) else []):
        person_folder = os.path.join(DATASET_DIR, person)
        if not os.path.isdir(person_folder):
            continue
        images = [os.path.join(person_folder, f) for f in os.listdir(person_folder)
                  if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if images:
            people.append(person)
            for img in images:
                entries.append((person, img))

    total = len(entries)
    processed = 0
    names = []
    embeddings = []

    for person, img_path in entries:
        emb = _face_embedding_from_image(img_path)
        if emb is not None:
            names.append(person)
            embeddings.append(emb)
        processed += 1
        if progress_callback:
            try:
                progress_callback(processed, total)
            except Exception:
                pass
        # small delay to avoid hammering (if needed)
        time.sleep(0.01)

    data = {"names": names, "embeddings": embeddings}
    save_embeddings(data)
    # return count of embeddings
    return len(embeddings)


# For direct run (optional)
if __name__ == "__main__":
    print("Training embeddings from dataset/ ...")
    n = train_with_progress()
    print(f"Saved {n} embeddings to {EMB_FILE}")
