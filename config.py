import os
from pydantic_settings import BaseSettings
from typing import Literal


class Settings(BaseSettings):
    # HuggingFace
    hf_token: str = ""

    # LLM
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    llm_provider: Literal["openai", "anthropic"] = "openai"
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-large"
    embedding_dim: int = 3072

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "multimodal_rag"

    # Redis / Celery
    redis_url: str = "redis://localhost:6379/0"
    celery_broker: str = "redis://localhost:6379/0"
    celery_backend: str = "redis://localhost:6379/1"

    # Ingestion
    chunk_size: int = 512
    chunk_overlap: int = 64
    max_image_size: int = 1024  # px, longest side
    image_embed_model: str = "openai/clip-vit-large-patch14"

    # Retrieval
    top_k_dense: int = 10
    top_k_sparse: int = 10
    top_k_rerank: int = 5
    rerank_model: str = "rerank-english-v3.0"  # Cohere
    cohere_api_key: str = ""

    # NLI hallucination guard
    nli_model: str = "cross-encoder/nli-deberta-v3-base"
    nli_threshold: float = 0.5

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    max_upload_mb: int = 50

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()

# Apply HF token to env so transformers/huggingface_hub picks it up automatically
if settings.hf_token:
    os.environ["HF_TOKEN"] = settings.hf_token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = settings.hf_token  # legacy name