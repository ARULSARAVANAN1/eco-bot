from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.core.config import settings
from app.core.vector_store import init_vector_store
from app.routers import chat


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the vector DB once on startup, clean up on shutdown."""
    await init_vector_store()
    yield


app = FastAPI(
    title="Eco-Bot API",
    description="E-Waste Awareness Chatbot powered by Groq + ChromaDB",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(chat.router, prefix="/api/v1")
