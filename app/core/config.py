from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    groq_api_key: str
    groq_model: str = "llama-3.3-70b-versatile"
    pdf_dir: str = "pdfs"                             # folder that holds your PDF files
    chroma_collection: str = "ewaste_docs"
    embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 400
    chunk_overlap: int = 60
    retrieve_top_k: int = 3
    groq_max_tokens: int = 200
    groq_temperature: float = 0.7


settings = Settings()
