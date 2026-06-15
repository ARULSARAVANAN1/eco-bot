from fastapi import FastAPI
from app.routers import chat, admin

app = FastAPI(
    title="Eco-Bot API",
    description="E-Waste Awareness Chatbot powered by Groq + ChromaDB",
    version="1.0.0",
)

@app.get("/health")
async def health():
    return {"status": "ok"}

app.include_router(chat.router, prefix="/api/v1")
app.include_router(admin.router, prefix="/api/v1/admin")