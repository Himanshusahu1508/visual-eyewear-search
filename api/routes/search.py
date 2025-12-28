import os
import uuid
from fastapi import APIRouter, UploadFile, File, Form, Depends
from api.dependencies import get_search_engine
from search.search_engine import VisualSearchEngine

UPLOAD_DIR = "data/user_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

router = APIRouter()


@router.post("/search")
async def search(
    image: UploadFile = File(...),
    top_k: int = Form(5),
    brand: str = Form(None),
    material: str = Form(None),
    price_min: int = Form(None),
    price_max: int = Form(None),
    text_query: str = Form(None),
    engine: VisualSearchEngine = Depends(get_search_engine),
):
    # Save uploaded image (functional requirement)
    ext = image.filename.split(".")[-1]
    filename = f"{uuid.uuid4()}.{ext}"
    image_path = os.path.join(UPLOAD_DIR, filename)

    with open(image_path, "wb") as f:
        f.write(await image.read())

    # Perform search
    results = engine.search(
        query_image_path=image_path,
        top_k=top_k,
        brand=brand,
        material=material,
        price_min=price_min,
        price_max=price_max,
        text_query=text_query,
    )

    return {
        "query_image": filename,
        "results": results,
    }
