import os
import numpy as np
import faiss

ARTIFACTS_DIR = "artifacts"
EMBEDDINGS_PATH = os.path.join(ARTIFACTS_DIR, "vit_embeddings.npy")
FAISS_INDEX_PATH = os.path.join(ARTIFACTS_DIR, "faiss.index")


def build_faiss_index():
    embeddings = np.load(EMBEDDINGS_PATH)

    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, FAISS_INDEX_PATH)

    print(f"FAISS index built with {index.ntotal} vectors")


if __name__ == "__main__":
    build_faiss_index()
