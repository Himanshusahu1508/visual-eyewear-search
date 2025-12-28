# Visual Similarity Search for Eyewear

This repository contains a complete, end-to-end visual similarity search system for eyewear products.  
The system is designed as a production-style machine learning application that combines computer vision models, vector search, structured metadata filtering, APIs, and a user interface.

The goal is to allow a user to upload an eyewear image and retrieve visually similar products based on **style, color, and shape**, with strong emphasis on shape correctness.

---

## Problem Statement

Given an input image of eyewear, retrieve visually similar eyewear products from a catalog while accounting for:

- Overall visual style
- Frame color
- Frame shape

Additionally, the system must:

- Store user-uploaded images in a structured database
- Support filters such as brand, material, and price
- Provide a backend API and frontend UI
- Allow future extensibility using feedback and text intent
- Be modular, explainable, and production-ready

---

## High-Level Pipeline Flow

1. User uploads an image via UI
2. Uploaded image is stored in a structured directory
3. Smart cropping is applied to isolate eyewear from face images
4. Visual features are extracted using a Vision Transformer
5. Color features are extracted using histogram-based methods
6. Shape is predicted using a trained classifier
7. FAISS retrieves visually similar candidates
8. Hard filters are applied (brand, material, price)
9. Results are re-ranked with a weighted scoring function
10. Results are returned via API and displayed in the UI
11. User feedback is recorded for future ranking improvements

---

## System Architecture

### Architecture Diagram

User
 │
 │  (Upload eyewear image + optional text / filters)
 ▼
Streamlit UI
 │
 │  HTTP request 
 ▼
FastAPI Backend
 │
 ├── Image Storage
 │     └─ Save uploaded image in structured folder
 │
 ├── Smart Crop Module
 │     └─ Crop face / eyewear region if detected
 │
 ├── Feature Extraction
 │     ├─ ViT Embedding Model (style features)
 │     ├─ Color Histogram Extractor (color features)
 │     └─ ViT Shape Classifier (frame shape)
 │
 ├── Vector Search
 │     └─ FAISS Index (style similarity search)
 │
 ├── Metadata Store
 │     └─ SQLite Database (brand, price, material, shape)
 │
 ├── Ranking Engine
 │     ├─ Style similarity score
 │     ├─ Color similarity score
 │     ├─ Shape match priority
 │     ├─ Text intent boost (optional)
 │     └─ User feedback boost
 │
 ▼
Ranked Results
 │
 │  JSON response
 ▼
Streamlit UI
 │
 │  Display images + metadata
 ▼
User Feedback
 │
 ▼
Feedback Store


---

### Component Breakdown

#### Feature Extraction
- Vision Transformer (ViT) for visual style embeddings
- HSV color histogram for frame color similarity
- ViT-based shape classifier for frame shape detection

#### Vector Search
- FAISS index for fast nearest-neighbor retrieval
- Over-fetching strategy to support filtering and re-ranking

#### Metadata Storage
- SQLite database for product metadata
- Tables include catalog items, user uploads, and feedback

#### Backend
- FastAPI for search, feedback, and health endpoints
- Clean separation of concerns using dependency injection

#### Frontend
- Streamlit UI for image upload and result visualization
- Filters and text input supported via UI controls

---

## Similarity Scoring Logic

The final similarity score is computed as a weighted combination of multiple signals.
Final Score =
α × Style Similarity

β × Color Similarity

γ × Shape Similarity

δ × Feedback Boost


Default weights:

- Style (α): 0.45
- Color (β): 0.25
- Shape (γ): 0.25
- Feedback (δ): 0.05

Shape similarity is treated as a high-priority signal to ensure geometrically correct results.

---

## Data Sources

- Manually curated metadata CSV file
- Image dataset organized by shape categories

---

## Model Details

### Style Embeddings
- Pretrained Vision Transformer
- Used only for feature extraction
- No fine-tuning to avoid overfitting on small datasets

### Shape Classifier
- Vision Transformer fine-tuned for shape classification
- Handles class imbalance using weighted loss
- Light data augmentation applied
- Saved as a reusable artifact

---

## Smart Cropping

The smart cropping module detects faces in uploaded images and crops around the eyewear region.

This improves robustness for real-world images where backgrounds are cluttered.

If no face is detected, the original image is used.

---

## Multi-Modal Search

The system supports optional text input along with the image.

Text intent can be used to:

- Apply filters (material, price range)
- Boost specific attributes (shape or material)
- Improve relevance without hardcoded rules

This design allows easy future integration with large language models.

---

## Database Design

SQLite is used as the primary metadata store.

Reasons for choosing SQLite:

- Lightweight and embedded
- No external service dependency
- Ideal for read-heavy workloads
- Easy to version and distribute

Tables include:

- catalog_items
- user_uploads
- feedback

---

## API Design

The backend is implemented using FastAPI.

### Endpoints

- POST /search  
  Accepts an image and optional parameters, returns ranked results

- POST /feedback  
  Records user feedback for a given result

- GET /health  
  Basic health check

Swagger UI is available for interactive testing.

---

## User Interface

<img width="1908" height="1056" alt="image" src="https://github.com/user-attachments/assets/11078c92-7fa8-4db9-a96d-6c52a77a3b80" />


The Streamlit UI provides:

- Image upload functionality
- Optional text input
- Filters for brand, material, and price
- Ranked result display
- Feedback buttons


---

## Project Structure

<img width="359" height="459" alt="Screenshot 2025-12-27 155005" src="https://github.com/user-attachments/assets/d18064ee-c5ee-4e80-a398-32e8690a96b4" />


---

## Execution Steps

### Install dependencies
pip install -r requirements.txt


### Build embeddings and FAISS index

python -m pipelines.build_embeddings
python -m pipelines.build_faiss_index


### Train shape classifier
python -m pipelines.train_shape_classifier


### Start backend
uvicorn api.main:app --reload


### Launch UI
streamlit run ui/app.py


---

## Design Rationale

- Vision Transformer provides better global context than CNNs
- Explicit shape classification reduces semantic ambiguity
- Hybrid scoring balances perception and structure
- Feedback loop enables learning without retraining
- Modular design allows independent improvement of components

---

## Future Improvements

- Learning-to-rank using feedback signals
- Shape-specific FAISS indices
- Better dataset balancing and augmentation
- CLIP-based multi-modal embeddings
- Automated evaluation framework

---

## Conclusion

This project delivers a complete visual similarity search system covering data processing, model training, vector search, ranking logic, APIs, and user interaction.

The focus is on clarity, correctness, and extensibility rather than shortcuts.

The architecture is suitable for real-world deployment and future production hardening.






