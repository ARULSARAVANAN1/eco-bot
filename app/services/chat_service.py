import random

from groq import Groq

from app.core.config import settings
from app.core.vector_store import get_collection, get_embed_model

# --------------------------------------------------------------------------- #
# Constants                                                                     #
# --------------------------------------------------------------------------- #

GREETING_INPUTS = {
    "hi", "hello", "hey", "hiya", "howdy", "greetings",
    "good morning", "good afternoon", "good evening",
    "what's up", "whats up", "sup", "how are you",
    "how r you", "how do you do", "nice to meet you",
    "pleased to meet you", "hi there", "hello there",
}

GREETING_RESPONSES = [
    "Hello! 👋 Welcome to Eco-Bot! I'm here to help you learn about e-waste "
    "and sustainable electronics disposal. What would you like to know?",
    "Hi there! 🌿 Great to see you! I'm Eco-Bot, your guide to understanding "
    "e-waste and how to manage it responsibly. Ask me anything!",
    "Hey! 😊 Welcome! I'm Eco-Bot — ready to help you explore e-waste awareness "
    "and sustainable practices. How can I assist you today?",
    "Greetings! 🌍 I'm Eco-Bot, here to help you understand the impact of "
    "electronic waste and how to handle it sustainably. What's on your mind?",
]

SYSTEM_PROMPT = (
    "You are Eco-Bot, a sharp and friendly AI guide specialising in e-waste awareness "
    "and sustainable electronics disposal. Answer in a concise, helpful, and engaging way. "
    "Keep answers under 60 words unless the user asks for more detail. "
    "Base your answer primarily on the provided context."
)

# --------------------------------------------------------------------------- #
# Groq client (lazy singleton)                                                  #
# --------------------------------------------------------------------------- #

_groq_client: Groq | None = None


def _get_groq_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=settings.groq_api_key)
    return _groq_client


def _retrieve(query: str) -> list[str]:
    """Return top-k relevant text chunks from ChromaDB."""
    embed_model = get_embed_model()
    collection = get_collection()

    q_embed = embed_model.encode([query]).tolist()
    results = collection.query(query_embeddings=q_embed, n_results=settings.retrieve_top_k)
    return results["documents"][0]


def _call_groq(prompt: str, system_prompt: str = "") -> str:
    client = _get_groq_client()
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=settings.groq_model,
        messages=messages,
        max_tokens=settings.groq_max_tokens,
        temperature=settings.groq_temperature,
    )
    return response.choices[0].message.content.strip()


# --------------------------------------------------------------------------- #
# Public service function                                                        #
# --------------------------------------------------------------------------- #

def answer_question(question: str) -> str:
    """
    Core RAG pipeline:
      1. Return a greeting if the message is a greeting.
      2. Retrieve relevant chunks from ChromaDB.
      3. Build a context-aware prompt and call Groq.
    """

    context_chunks = _retrieve(question)
    context_str = "\n---\n".join(context_chunks)

    prompt = (
        f"Context from e-waste knowledge base:\n{context_str}\n\n"
        f"User question: {question}\n\n"
        "Answer based on the context above:"
    )

    try:
        answer = _call_groq(prompt, system_prompt=SYSTEM_PROMPT)
        return answer or "I'm sorry, I couldn't generate a response. Please try rephrasing."
    except Exception as exc:
        return f"⚠️ Error calling Groq: {exc}"
