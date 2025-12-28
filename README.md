# Visual Similarity Search for Eyewear

This project implements an end-to-end visual similarity search system for eyewear products.
The system allows users to upload an image of eyewear and retrieve visually similar products based on multiple attributes such as style, color, and shape.

The solution is designed as a production-style machine learning system rather than a standalone model experiment.


Problem Statement

Given an input image of eyewear, retrieve visually similar eyewear products from a catalog while accounting for:

Overall visual style

Frame color

Frame shape

The system must also:

Store uploaded images in a structured manner

Support filtering and ranking

Provide an API and user interface

Allow future extensibility through feedback and text intent

High-Level Pipeline Flow

User uploads an image (optionally with text input)

Image is optionally cropped to focus on the eyewear region

Visual features are extracted using a Vision Transformer

Color features are extracted using histogram-based methods

Shape is predicted using a trained classifier

Nearest neighbors are retrieved using FAISS

Results are re-ranked using a weighted scoring strategy

Results are returned via API and displayed in the UI

User feedback is recorded for future ranking improvements

System Design Overview

The system is composed of independent but connected components.

Feature Extraction

Vision Transformer (ViT) for style embeddings

Color histogram extraction in HSV space

ViT-based shape classification model

Storage

FAISS index for fast vector similarity search

SQLite database for metadata and structured queries

File system storage for uploaded images

Serving

FastAPI backend for search and feedback endpoints

Streamlit frontend for user interaction

Similarity Scoring Logic

The final ranking score is computed as a weighted combination of multiple similarity signals.

FinalScore =

α × Style similarity (ViT embeddings)

β × Color similarity (histogram comparison)

γ × Shape similarity (exact match)

δ × Feedback boost

Default weights used in the project:

Style (α): 0.45

Color (β): 0.25

Shape (γ): 0.25

Feedback (δ): 0.05

Shape similarity is treated as a high-priority signal to ensure geometrically correct results.

Data Sources

Product image Excel file provided by the company

Manually curated metadata CSV file

Image dataset organized by shape categories

Model Details
Style Embeddings

Pretrained Vision Transformer

Used only for feature extraction

No fine-tuning performed to avoid overfitting on small data

Shape Classifier

Vision Transformer fine-tuned for shape classification

Handles class imbalance using weighted loss

Trained with light data augmentation

Saved as a reusable artifact

Smart Cropping Feature

The smart cropping module detects faces in uploaded images and crops around the eyewear region.
This improves search quality for real-world photos where the background may be cluttered.

If no face is detected, the original image is used.

Multi-Modal Search Support

The system supports optional text input along with the image.

Text input can be used to:

Apply filters (material, price range)

Boost certain attributes (shape or color)

Improve relevance without hardcoding rules

This design allows future extension using large language models if required.

Database Design

SQLite is used as the primary metadata store.

Reasons for choosing SQLite:

Lightweight and embedded

No external service dependency

Well-suited for read-heavy workloads

Easy to version and distribute

Tables include:

Catalog items

User uploads

Feedback records

API Design

The backend is implemented using FastAPI.

Key endpoints:

POST /search
Accepts an image and optional parameters, returns ranked results

POST /feedback
Records user feedback for a given result

GET /health
Basic health check

Interactive API documentation is available via Swagger UI.

User Interface

The Streamlit UI provides:

Image upload functionality

Optional text input

Display of ranked results

Feedback buttons for user interaction

The UI communicates exclusively through the API layer.


visual-eyewear-search/
│
├── api/                FastAPI backend
├── artifacts/          Trained models and generated files
├── data/               Dataset, metadata, SQLite database
├── feedback/           Feedback handling logic
├── models/             ViT embedding and shape models
├── pipelines/          Training and preprocessing scripts
├── search/             Core search and ranking logic
├── ui/                 Streamlit application
├── uploads/            Stored user uploads
├── utils/              Utility modules
├── README.md
└── requirements.txt

Execution Steps

Install dependencies

pip install -r requirements.txt


Build embeddings and FAISS index

python -m pipelines.build_embeddings
python -m pipelines.build_faiss_index


Train the shape classifier

python -m pipelines.train_shape_classifier


Start the backend

uvicorn api.main:app --reload


Launch the UI

streamlit run ui/app.py


Design Rationale

Vision Transformer provides better global context than CNNs

Explicit shape classification reduces semantic ambiguity

Hybrid scoring balances perception and structure

Feedback loop enables learning without retraining

Modular design allows independent improvement of components

Future Improvements
Persist feedback data and incorporate learning-to-rank

Shape-specific FAISS indices

Better dataset balancing and augmentation

CLIP-based multi-modal embeddings

Automated evaluation framework

Conclusion

This project delivers a complete visual similarity search system, covering data processing, model training, vector search, ranking logic, APIs, and user interaction.
The focus was on clarity, extensibility, and correctness rather than shortcuts or over-optimization.

The architecture is suitable for real-world extension and production hardening.
