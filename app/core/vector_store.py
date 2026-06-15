import os
import chromadb
from chromadb.utils import embedding_functions
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings

_collection = None
_is_ready: bool = False
_status: str = "not_started"   # not_started | loading | ready | error
_error: str = ""


def is_ready() -> bool:
    return _is_ready


def get_status() -> dict:
    return {
        "ready": _is_ready,
        "status": _status,
        "error": _error if _error else None,
        "chunks": _collection.count() if _collection else 0,
    }


def get_collection():
    if _collection is None:
        raise RuntimeError("Vector store not initialised. Trigger POST /api/v1/admin/embed first.")
    return _collection


async def init_vector_store() -> None:
    global _collection, _is_ready, _status, _error

    try:
        _status = "loading"
        print("⏳ Loading PDFs...")

        documents = PyPDFDirectoryLoader(settings.pdf_dir).load()
        if not documents:
            raise RuntimeError(f"No PDFs found in '{settings.pdf_dir}'.")
        print(f"✅ Loaded {len(documents)} page(s)")

        chunks = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        ).split_documents(documents)
        print(f"✅ Split into {len(chunks)} chunks")

        embed_fn = embedding_functions.DefaultEmbeddingFunction()

        client = chromadb.Client()
        try:
            client.delete_collection(settings.chroma_collection)
        except Exception:
            pass

        _collection = client.create_collection(
            settings.chroma_collection,
            embedding_function=embed_fn
        )

        texts = [c.page_content for c in chunks]
        ids = [str(i) for i in range(len(texts))]
        _collection.add(documents=texts, ids=ids)

        _is_ready = True
        _status = "ready"
        print(f"✅ Vector DB ready — {_collection.count()} chunks indexed")

    except Exception as e:
        _status = "error"
        _error = str(e)
        print(f"❌ Embedding failed: {e}")