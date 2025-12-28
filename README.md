# Visual Similarity Search for Eyewear

This project implements an end-to-end visual similarity search system for eyewear products.
The system allows users to upload an image of eyewear and retrieve visually similar products based on multiple attributes such as style, color, and shape.

The solution is designed as a production-style machine learning system rather than a standalone model experiment.

---

## Problem Statement

Given an input image of eyewear, retrieve visually similar eyewear products from a catalog while accounting for:

- Overall visual style
- Frame color
- Frame shape

The system must also:

- Store uploaded images in a structured manner
- Support filtering and ranking
- Provide an API and user interface
- Allow future extensibility through feedback and text intent

---

## High-Level Pipeline Flow

1. User uploads an image (optionally with text input)
2. Image is optionally cropped to focus on the eyewear region
3. Visual features are extracted using a Vision Transformer
4. Color features are extracted using histogram-based methods
5. Shape is predicted using a trained classifier
6. Nearest neighbors are retrieved using FAISS
7. Results are re-ranked using a weighted scoring strategy
8. Results are returned via API and displayed in the UI
9. User feedback is recorded for future ranking improvements

---

## System Design Overview

The system is composed of independent but connected components.

### Feature Extraction

- Vision Transformer (ViT) for style embeddings
- Color histogram extraction in HSV space
- ViT-based shape classification model

### Storage

- FAISS index for fast vector similarity search
- SQLite database for metadata and structured queries
- File system storage for uploaded images

### Serving

- FastAPI backend for search and feedback endpoints
- Streamlit frontend for user interaction

---

## Similarity Scoring Logic

The final ranking score is computed as a weighted combination of multiple similarity signals.

FinalScore =  
α × Style similarity (ViT embeddings)  
β × Color similarity (histogram comparison)  
γ × Shape similarity (exact match)  
δ × Feedback boost  

Default weights:

- Style (α): 0.45
- Color (β): 0.25
- Shape (γ): 0.25
- Feedback (δ): 0.05

Shape similarity is treated as a high-priority signal to ensure geometrically correct results.

---

## Data Sources

- Product image Excel file provided by the company
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
- Trained with light data augmentation
- Stored as a reusable artifact

---

## Smart Cropping Feature

The smart cropping module detects faces in uploaded images and crops around the eyewear region.
This improves search quality for real-world photos where the background may be cluttered.

If no face is detected, the original image is used.

---

## Multi-Modal Search Support

The system supports optional text input along with the image.

Text input can be used to:

- Apply filters (material, price range)
- Boost attributes (shape or color)
- Improve relevance without hardcoded rules

The design allows future extension using large language models.

---

## Database Design

SQLite is used as the primary metadata store.

Reasons for choosing SQLite:

- Lightweight and embedded
- No external service dependency
- Well-suited for read-heavy workloads
- Easy to version and distribute

Tables include:

- Catalog items
- User uploads
- Feedback records

---

## API Design

The backend is implemented using FastAPI.

Endpoints:

- POST /search
- POST /feedback
- GET /health

Swagger UI is available for interactive testing.

---

## User Interface

The Streamlit UI provides:

- Image upload
- Optional text input
- Ranked result display
- Feedback buttons

The UI communicates only through the API layer.

---

## Project Structure

visual-eyewear-search/
│
├── api/ FastAPI backend
├── artifacts/ Trained models and generated files
├── data/ Dataset, metadata, SQLite database
├── feedback/ Feedback handling logic
├── models/ ViT embedding and shape models
├── pipelines/ Training and preprocessing scripts
├── search/ Core search and ranking logic
├── ui/ Streamlit application
├── uploads/ Stored user uploads
├── utils/ Utility modules
├── README.md
└── requirements.txt


---

## Execution Steps

Install dependencies: pip install -r requirements.txt

Build embeddings and FAISS index:
-python -m pipelines.build_embeddings
-python -m pipelines.build_faiss_index


Train shape classifier:
python -m pipelines.train_shape_classifier

-Start backend:
uvicorn api.main:app --reload

-Launch UI:
streamlit run ui/app.py


---

## Design Rationale

- ViT provides better global context than CNNs
- Explicit shape classification reduces ambiguity
- Hybrid scoring balances perception and structure
- Feedback loop enables improvement without retraining
- Modular design allows independent upgrades

---

## Future Improvements

- Learning-to-rank using feedback data
- Shape-specific FAISS indices
- Improved dataset balancing
- CLIP-based multi-modal embeddings
- Automated evaluation metrics

---

## Conclusion

This project delivers a complete visual similarity search system covering data ingestion, model training, vector search, ranking logic, APIs, and UI.
The architecture is extensible and suitable for production hardening.





