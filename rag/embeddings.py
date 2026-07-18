# rag/embeddings.py
"""
Embedding model configuration for ChromaDB.
Uses sentence-transformers (runs fully locally — no API key required).
"""

from langchain_community.embeddings import HuggingFaceEmbeddings

_MODEL_NAME = "all-MiniLM-L6-v2"   # Fast, small, ChromaDB-compatible

def get_embedding_function() -> HuggingFaceEmbeddings:
    """Return a cached HuggingFace embedding model instance."""
    return HuggingFaceEmbeddings(
        model_name=_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": 128,   # 4× larger batches → much faster on CPU
        },
    )

