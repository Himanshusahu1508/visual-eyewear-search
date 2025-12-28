from fastapi import APIRouter, Depends
from api.schemas import FeedbackRequest
from api.dependencies import get_search_engine

router = APIRouter()


@router.post("/feedback")
def submit_feedback(
    data: FeedbackRequest,
    engine=Depends(get_search_engine)
):
    engine.feedback_store.add_feedback(
        shape=data.shape,
        image_id=data.image_id
    )

    return {"status": "feedback recorded"}
