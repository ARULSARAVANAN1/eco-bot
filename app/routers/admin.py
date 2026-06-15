from fastapi import APIRouter
from pydantic import BaseModel

from app.core.vector_store import init_vector_store, is_ready, get_status

router = APIRouter(tags=["Admin"])


class EmbedResponse(BaseModel):
    message: str
    status: str
    chunks: int


@router.post("/embed", response_model=EmbedResponse, summary="Trigger embedding flow")
async def embed() -> EmbedResponse:
    """
    Triggers the PDF parsing and embedding flow.
    """
    if is_ready():
        status = get_status()
        return EmbedResponse(
            message="Vector store is already initialised.",
            status="already_ready",
            chunks=status["chunks"]
        )

    await init_vector_store()  # ← await directly, no background task

    status = get_status()
    return EmbedResponse(
        message="Embedding complete",
        status="ready",
        chunks=status["chunks"]
    )


@router.get("/status", summary="Check embedding status")
async def status():
    return get_status()
