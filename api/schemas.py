from typing import List
from pydantic import BaseModel


class SearchResponseItem(BaseModel):
    image_id: int
    image_path: str
    brand: str
    shape: str
    material: str
    price: int
    similarity: float


class SearchResponse(BaseModel):
    query_image: str
    results: List[SearchResponseItem]


class FeedbackRequest(BaseModel):
    image_id: int
    shape: str
