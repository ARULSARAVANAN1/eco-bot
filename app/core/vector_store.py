import chromadb
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings

# Singletons shared across requests
_collection: chromadb.Collection | None = None
_embed_model: SentenceTransformer | None = None

BATCH_SIZE = 64


async def init_vector_store() -> None:
    """Called once at startup to build the ChromaDB collection."""
    global _collection, _embed_model

    print("⏳ Loading embedding model...")
    _embed_model = SentenceTransformer(settings.embed_model_name)
    print("✅ Embedding model ready")

    print("⏳ Loading PDFs...")
    documents = PyPDFDirectoryLoader(settings.pdf_dir).load()
    if not documents:
        raise RuntimeError(
            f"No PDFs found in '{settings.pdf_dir}'. "
            "Update PDF_DIR in .env or drop PDFs into that folder."
        )
    print(f"✅ Loaded {len(documents)} page(s)")

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    ).split_documents(documents)
    print(f"✅ Split into {len(chunks)} chunks")

    client = chromadb.Client()
    try:
        client.delete_collection(settings.chroma_collection)
    except Exception:
        pass

    _collection = client.create_collection(settings.chroma_collection)

    texts = [c.page_content for c in chunks]
    ids = [str(i) for i in range(len(texts))]
    embeddings: list[list[float]] = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = _embed_model.encode(texts[i : i + BATCH_SIZE], show_progress_bar=False).tolist()
        embeddings.extend(batch)

    _collection.add(documents=texts, embeddings=embeddings, ids=ids)
    print(f"✅ Vector DB ready — {_collection.count()} chunks indexed")


def get_collection() -> chromadb.Collection:
    if _collection is None:
        raise RuntimeError("Vector store not initialised. Did startup run?")
    return _collection


def get_embed_model() -> SentenceTransformer:
    if _embed_model is None:
        raise RuntimeError("Embedding model not loaded. Did startup run?")
    return _embed_model
