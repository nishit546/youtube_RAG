"""
retrieval/retriever.py

Retrieves semantically relevant chunks from one or more video stores.

Features:
- Top-k semantic similarity search (k=5 default)
- Simple keyword-based re-ranking on top of semantic results
- Cross-video retrieval for comparison queries
- Returns structured source objects with timestamps
"""

import logging
from langchain_community.vectorstores import Chroma

logger = logging.getLogger(__name__)


def _format_timestamp(seconds: float) -> str:
    """Convert seconds to human-readable MM:SS or H:MM:SS format."""
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def _rerank(docs: list, query: str) -> list:
    """
    Simple keyword re-ranker: boost docs whose text contains query keywords.
    Docs are returned sorted by (keyword_hits DESC, original_order ASC).
    This complements semantic search to prefer exact matches.
    """
    query_words = set(query.lower().split())
    scored = []
    for rank, doc in enumerate(docs):
        text_words = set(doc.page_content.lower().split())
        overlap = len(query_words & text_words)
        scored.append((doc, overlap, rank))
    # Sort by overlap desc, then original rank asc
    scored.sort(key=lambda x: (-x[1], x[2]))
    return [item[0] for item in scored]


def retrieve_chunks(
    store: Chroma,
    query: str,
    k: int = 5,
) -> list[dict]:
    """
    Retrieve and re-rank top-k relevant chunks from a single video store.

    Returns:
        List of source dicts:
        {
            "text": str,
            "start": float,
            "end": float,
            "start_fmt": "M:SS",
            "end_fmt": "M:SS",
            "video_id": str,
            "title": str,
            "url": str   (clickable YouTube timestamp link)
        }
    """
    docs = store.similarity_search(query, k=k)
    docs = _rerank(docs, query)

    sources = []
    for doc in docs:
        meta = doc.metadata
        start = meta.get("start", 0.0)
        end = meta.get("end", 0.0)
        vid = meta.get("video_id", "")
        sources.append({
            "text": doc.page_content,
            "start": start,
            "end": end,
            "start_fmt": _format_timestamp(start),
            "end_fmt": _format_timestamp(end),
            "video_id": vid,
            "title": meta.get("title", vid),
            "url": f"https://www.youtube.com/watch?v={vid}&t={int(start)}",
        })
    return sources


def retrieve_chunks_multi(
    stores: dict[str, Chroma],
    query: str,
    k_per_video: int = 3,
) -> list[dict]:
    """
    Retrieve chunks from multiple video stores for cross-video comparison.

    Args:
        stores: { video_id: Chroma store }
        query: user query
        k_per_video: chunks to retrieve per video

    Returns:
        Combined list of source dicts (same format as retrieve_chunks)
    """
    all_sources = []
    for vid, store in stores.items():
        try:
            chunks = retrieve_chunks(store, query, k=k_per_video)
            all_sources.extend(chunks)
        except Exception as e:
            logger.warning(f"[retriever] Failed retrieving from {vid}: {e}")
    return all_sources


def build_context_string(sources: list[dict]) -> str:
    """
    Format sources into a context block for the LLM prompt.
    Each chunk is annotated with its timestamp for citation.
    """
    if not sources:
        return "No relevant context found."
    parts = []
    for i, src in enumerate(sources, 1):
        vid_label = f"[{src['title']}]" if src.get("title") else f"[{src['video_id']}]"
        parts.append(
            f"[Chunk {i} | {vid_label} | {src['start_fmt']} – {src['end_fmt']}]\n"
            f"{src['text']}"
        )
    return "\n\n".join(parts)
