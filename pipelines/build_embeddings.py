import os
import sys
import json
import sqlite3
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch

# ---------------- Project Root ----------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

# ---------------- Imports ----------------
from models.embedding_model import load_vit, get_vit_transform
from utils.color_features import extract_color_histogram

# ---------------- Paths ----------------
DATASET_ROOT = os.path.join(PROJECT_ROOT, "data", "dataset_v1")
IMAGES_ROOT = os.path.join(DATASET_ROOT, "raw_images")

ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")
DB_PATH = os.path.join(PROJECT_ROOT, "data", "eyewear.db")

EMBEDDINGS_PATH = os.path.join(ARTIFACTS_DIR, "vit_embeddings.npy")
COLOR_FEATURES_PATH = os.path.join(ARTIFACTS_DIR, "color_features.npy")
ID_MAP_PATH = os.path.join(ARTIFACTS_DIR, "id_mapping.json")


def build_embeddings():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = load_vit(device)
    transform = get_vit_transform()

    # ---------------- Load catalog ----------------
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT image_id, image_path FROM catalog_items")
    rows = cursor.fetchall()
    conn.close()

    embeddings = []
    color_features = []
    id_map = {}

    for image_id, rel_path in tqdm(rows, desc="Building embeddings"):
        img_path = os.path.join(IMAGES_ROOT, rel_path)

        image = Image.open(img_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)

        # -------- ViT embedding --------
        with torch.no_grad():
            emb = model(image_tensor).cpu().numpy()[0]

        # L2 normalize (CRITICAL for cosine similarity)
        emb = emb / np.linalg.norm(emb)

        # -------- Color features --------
        color_feat = extract_color_histogram(img_path)

        embeddings.append(emb)
        color_features.append(color_feat)
        id_map[str(image_id)] = rel_path

    # ---------------- Save artifacts ----------------
    np.save(EMBEDDINGS_PATH, np.array(embeddings).astype("float32"))
    np.save(COLOR_FEATURES_PATH, np.array(color_features).astype("float32"))

    with open(ID_MAP_PATH, "w") as f:
        json.dump(id_map, f, indent=2)

    print("ViT embeddings saved:", EMBEDDINGS_PATH)
    print("Color features saved:", COLOR_FEATURES_PATH)
    print("Embedding shape:", np.array(embeddings).shape)


if __name__ == "__main__":
    build_embeddings()
