"""
embedding/embedder.py

Manages HuggingFace embeddings and per-video persistent Chroma vector stores.

Key design decisions:
- Singleton embedder to avoid reloading the model on each request.
- Each video gets its own Chroma collection stored at:
    yt_chroma_db/<video_id>/chroma/
- Caching check: if the collection already has documents, skip re-embedding.
- Metadata stored per chunk: {start, end, video_id, title}
"""

import logging
import os
import uuid

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_BASE = os.path.join(BASE_DIR, "yt_chroma_db")

# ── Singleton embedder (loaded once at import time) ──────────────────────────
_embedder: HuggingFaceEmbeddings | None = None


def get_embedder() -> HuggingFaceEmbeddings:
    """Return shared HuggingFace embedder (lazy-loaded singleton)."""
    global _embedder
    if _embedder is None:
        logger.info("[embedder] Loading sentence-transformer model...")
        _embedder = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        logger.info("[embedder] Model loaded.")
    return _embedder


# ── Per-video vector store ────────────────────────────────────────────────────
def _chroma_persist_dir(video_id: str) -> str:
    path = os.path.join(CHROMA_BASE, video_id, "chroma")
    os.makedirs(path, exist_ok=True)
    return path


def get_or_create_store(video_id: str, chunks: list[dict]) -> Chroma:
    """
    Return a Chroma vector store for the given video.

    If the store already exists on disk with data, it's loaded from disk
    (no re-embedding). If it's empty or missing, the chunks are embedded
    and persisted.

    Args:
        video_id: YouTube video ID
        chunks: list of { text, start, end, video_id, title }

    Returns:
        Chroma vector store ready for similarity search
    """
    persist_dir = _chroma_persist_dir(video_id)
    embedder = get_embedder()
    collection_name = f"video_{video_id}"

    # ── Try loading existing store ─────────────────────────────────────────────
    existing = Chroma(
        collection_name=collection_name,
        embedding_function=embedder,
        persist_directory=persist_dir,
    )
    count = existing._collection.count()
    if count > 0:
        logger.info(f"[embedder] Loaded existing store for {video_id} ({count} chunks)")
        return existing

    # ── Build Documents with metadata ────────────────────────────────────────
    logger.info(f"[embedder] Embedding {len(chunks)} chunks for {video_id}...")
    documents = [
        Document(
            page_content=chunk["text"],
            metadata={
                "start": chunk["start"],
                "end": chunk["end"],
                "video_id": chunk["video_id"],
                "title": chunk.get("title", video_id),
                # Unique ID to avoid Chroma deduplication collisions
                "chunk_id": str(uuid.uuid4()),
            },
        )
        for chunk in chunks
    ]

    # ── Embed and persist ─────────────────────────────────────────────────────
    store = Chroma.from_documents(
        documents=documents,
        embedding=embedder,
        collection_name=collection_name,
        persist_directory=persist_dir,
    )
    store.persist()
    logger.info(f"[embedder] Embedded and persisted {len(documents)} chunks for {video_id}")
    return store


def load_existing_store(video_id: str) -> Chroma | None:
    """
    Load an already-persisted Chroma store for a video.
    Returns None if the store does not exist or is empty.
    """
    persist_dir = _chroma_persist_dir(video_id)
    if not os.path.exists(persist_dir):
        return None
    embedder = get_embedder()
    store = Chroma(
        collection_name=f"video_{video_id}",
        embedding_function=embedder,
        persist_directory=persist_dir,
    )
    if store._collection.count() == 0:
        return None
    return store
