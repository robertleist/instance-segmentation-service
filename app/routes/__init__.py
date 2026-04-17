from fastapi import APIRouter


router = APIRouter()
session_router = APIRouter(prefix="/annotation_session", tags=["annotation_session"])


@router.get("/health")
async def health_check():
    return {"status": "ok"}
