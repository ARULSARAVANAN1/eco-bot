from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.services.chat_service import answer_question

router = APIRouter(tags=["Chat"])


# --------------------------------------------------------------------------- #
# Request / Response schemas                                                    #
# --------------------------------------------------------------------------- #

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, example="What is e-waste?")


class ChatResponse(BaseModel):
    question: str
    answer: str


class InitResponse(BaseModel):
    bot: str
    message: str
    hint: str


# --------------------------------------------------------------------------- #
# Endpoint                                                                      #
# --------------------------------------------------------------------------- #

@router.get("/init", response_model=InitResponse, summary="Init / Welcome message")
async def init() -> InitResponse:
    return InitResponse(
        bot="Eco-Bot 🌿",
        message="👋 Hey there! I'm Eco-Bot — your go-to guide for e-waste & sustainable practices. Ask me anything!",
        hint="POST /api/v1/chat with { question: '...' } to get started.",
    )


@router.post("/chat", response_model=ChatResponse, summary="Ask Eco-Bot a question")
async def chat(payload: ChatRequest) -> ChatResponse:
    """
    Send a question to Eco-Bot.

    - Greetings get a friendly canned response instantly.  
    - All other questions go through the RAG pipeline:  
      ChromaDB retrieval → Groq (llama-3.3-70b-versatile) → answer.
    """
    answer = answer_question(payload.question)
    return ChatResponse(question=payload.question, answer=answer)
