from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes.search import router as search_router
from api.routes.feedback import router as feedback_router

app = FastAPI(
    title="Visual Eyewear Search API",
    description="ViT-based visual similarity search for eyewear",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(search_router)
app.include_router(feedback_router)


@app.get("/")
def health_check():
    return {"status": "ok"}

from fastapi.staticfiles import StaticFiles

app.mount(
    "/static",
    StaticFiles(directory="data/dataset_v1/raw_images"),
    name="static"
)
