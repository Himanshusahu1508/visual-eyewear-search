import os
import json
import sqlite3
import numpy as np
import faiss
from PIL import Image
import torch

from models.embedding_model import load_vit, get_vit_transform
from models.vit_shape_classifier import ViTShapeClassifier

from utils.smart_crop import smart_crop_eyewear
from utils.text_intent_parser import parse_text_intent
from utils.color_features import (
    extract_color_histogram,
    compute_color_similarity,
)

from feedback.feedback_store import FeedbackStore


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

DB_PATH = os.path.join(DATA_DIR, "eyewear.db")

EMBEDDINGS_PATH = os.path.join(ARTIFACTS_DIR, "vit_embeddings.npy")
COLOR_FEATURES_PATH = os.path.join(ARTIFACTS_DIR, "color_features.npy")
FAISS_INDEX_PATH = os.path.join(ARTIFACTS_DIR, "faiss.index")
ID_MAP_PATH = os.path.join(ARTIFACTS_DIR, "id_mapping.json")
SHAPE_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "shape_classifier.pth")


STYLE_WEIGHT = 0.7
COLOR_WEIGHT = 0.3
FEEDBACK_WEIGHT = 0.1


class VisualSearchEngine:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.embedding_model = load_vit(self.device)
        self.transform = get_vit_transform()

        checkpoint = torch.load(SHAPE_MODEL_PATH, map_location=self.device)
        self.shape_to_idx = checkpoint["shape_to_idx"]
        self.idx_to_shape = {v: k for k, v in self.shape_to_idx.items()}

        self.shape_model = ViTShapeClassifier(
            num_classes=len(self.shape_to_idx)
        ).to(self.device)
        self.shape_model.load_state_dict(checkpoint["model_state"])
        self.shape_model.eval()

        self.index = faiss.read_index(FAISS_INDEX_PATH)
        self.color_histograms = np.load(COLOR_FEATURES_PATH)

        with open(ID_MAP_PATH, "r") as f:
            self.id_map = json.load(f)

        self.index_to_image_id = {
            idx: int(image_id)
            for idx, image_id in enumerate(self.id_map.keys())
        }

        self.feedback_store = FeedbackStore()

    def _embed_image(self, image_path: str) -> np.ndarray:
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding = self.embedding_model(tensor).cpu().numpy()[0]

        return embedding / np.linalg.norm(embedding)

    def _predict_shape(self, image_path: str) -> str:
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.shape_model(tensor)
            pred_idx = logits.argmax(dim=1).item()

        return self.idx_to_shape[pred_idx]

    def search(
        self,
        query_image_path: str,
        top_k: int = 5,
        brand: str | None = None,
        material: str | None = None,
        price_min: int | None = None,
        price_max: int | None = None,
        text_query: str | None = None,
    ):
        intent = parse_text_intent(text_query) if text_query else {}

        query_image_path = smart_crop_eyewear(query_image_path)

        query_embedding = self._embed_image(query_image_path)
        query_shape = self._predict_shape(query_image_path)
        query_color_hist = extract_color_histogram(query_image_path)

        scores, indices = self.index.search(
            query_embedding.reshape(1, -1),
            top_k * 10,
        )

        same_shape_results = []
        other_shape_results = []

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        for style_sim, faiss_idx in zip(scores[0], indices[0]):
            image_id = self.index_to_image_id[faiss_idx]

            cursor.execute(
                """
                SELECT image_path, brand, shape, material, price
                FROM catalog_items
                WHERE image_id = ?
                """,
                (image_id,),
            )

            row = cursor.fetchone()
            if row is None:
                continue

            image_path, brand_db, shape_db, material_db, price = row

            if brand and brand_db.lower() != brand.lower():
                continue
            if material and material_db.lower() != material.lower():
                continue
            if price_min is not None and price < price_min:
                continue
            if price_max is not None and price > price_max:
                continue

            color_sim = compute_color_similarity(
                query_color_hist,
                self.color_histograms[faiss_idx],
            )

            feedback_boost = self.feedback_store.get_boost(
                query_shape,
                image_id,
            )

            if shape_db == query_shape:
                score = (
                    STYLE_WEIGHT * float(style_sim)
                    + COLOR_WEIGHT * float(color_sim)
                    + FEEDBACK_WEIGHT * float(feedback_boost)
                )
            else:
                score = (
                    0.3 * float(style_sim)
                    + 0.1 * float(color_sim)
                )

            if intent.get("shape") == shape_db:
                score += 0.1
            if intent.get("material") == material_db:
                score += 0.05
            if intent.get("price_max") and price > intent["price_max"]:
                score -= 0.1

            result = {
                "image_id": image_id,
                "image_path": image_path,
                "brand": brand_db,
                "shape": shape_db,
                "material": material_db,
                "price": price,
                "similarity": float(score),
            }

            if shape_db == query_shape:
                same_shape_results.append(result)
            else:
                other_shape_results.append(result)

        conn.close()

        same_shape_results.sort(key=lambda x: x["similarity"], reverse=True)
        other_shape_results.sort(key=lambda x: x["similarity"], reverse=True)

        return (same_shape_results + other_shape_results)[:top_k]
