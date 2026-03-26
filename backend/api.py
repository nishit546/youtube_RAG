"""
api.py — Route handlers

Endpoints:
  POST /ingest        → Load and embed a video (cached)
  POST /ask           → Answer a question (with mode + history)
  POST /compare       → Cross-video comparison query
  GET  /history/{id}  → Get chat history for a video
  DELETE /history/{id}→ Clear chat history for a video
  GET  /videos        → List all ingested videos
"""

import logging
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ingestion.transcript import load_transcript
from embedding.embedder import get_or_create_store, load_existing_store
from retrieval.retriever import retrieve_chunks, retrieve_chunks_multi, build_context_string
from generation.generator import generate_answer
from memory import session as mem

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Request / Response Models ─────────

class IngestRequest(BaseModel):
    video_id: str
    force_refresh: bool = False


class IngestResponse(BaseModel):
    video_id: str
    title: str
    chunk_count: int
    cached: bool
    source: str  


class AskRequest(BaseModel):
    video_id: str
    question: str
    mode: str = Field(
        default="ask",
        description="One of: ask | summary | keypoints | deep | quiz | navigate"
    )
    k: int = Field(default=5, ge=1, le=10, description="Number of chunks to retrieve")


class SourceChunk(BaseModel):
    text: str
    start: float
    end: float
    start_fmt: str
    end_fmt: str
    video_id: str
    title: str
    url: str


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    mode: str


class CompareRequest(BaseModel):
    video_ids: list[str] = Field(min_length=2)
    question: str
    k_per_video: int = Field(default=3, ge=1, le=5)


class HistoryTurn(BaseModel):
    question: str
    answer: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest):
    """
    Load, chunk, and embed a YouTube video.
    Returns immediately if the video is already cached (embedding skipped).
    """
    try:
        data = load_transcript(req.video_id, force_refresh=req.force_refresh)
    except RuntimeError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception(f"[ingest] Failed for {req.video_id}: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to load video: {e}")

    # Build / load embedding store
    try:
        get_or_create_store(req.video_id, data["chunks"])
    except Exception as e:
        logger.exception(f"[ingest] Embedding failed for {req.video_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")

    return IngestResponse(
        video_id=data["video_id"],
        title=data["title"],
        chunk_count=len(data["chunks"]),
        cached=data["cached"],
        source=data["source"],
    )


@router.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    """
    Answer a question about a video using RAG + chat memory.

    Modes: ask | summary | keypoints | deep | quiz | navigate
    """
    # Load the embedding store
    store = load_existing_store(req.video_id)
    if store is None:
        raise HTTPException(
            status_code=404,
            detail="Video not ingested. Call POST /ingest first."
        )

    # For summary/keypoints/quiz we grab more context (top chunks overall)
    k = req.k
    if req.mode in ("summary", "keypoints", "quiz"):
        k = 10  # retrieve broadly for full-video modes

    # Retrieve relevant chunks
    sources = retrieve_chunks(store, req.question, k=k)

    # Build context string for LLM
    context = build_context_string(sources)

    # Fetch conversation history window
    history_turns = mem.get_window(req.video_id, n=4)
    history_str = mem.format_history_for_prompt(history_turns)

    # Generate answer
    try:
        answer = generate_answer(
            context=context,
            question=req.question,
            mode=req.mode,
            history=history_str,
        )
    except Exception as e:
        logger.exception(f"[ask] Generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

    # Store turn in session memory (skip for stateless modes)
    if req.mode in ("ask", "deep", "navigate"):
        mem.add_turn(req.video_id, req.question, answer)

    return AskResponse(
        answer=answer,
        sources=[SourceChunk(**s) for s in sources],
        mode=req.mode,
    )


@router.post("/compare", response_model=AskResponse)
async def compare(req: CompareRequest):
    """
    Answer a cross-video comparison question.
    Retrieves context from all specified video IDs and generates a comparative answer.
    """
    stores = {}
    for vid in req.video_ids:
        store = load_existing_store(vid)
        if store is None:
            raise HTTPException(
                status_code=404,
                detail=f"Video '{vid}' not ingested. Call POST /ingest first."
            )
        stores[vid] = store

    sources = retrieve_chunks_multi(stores, req.question, k_per_video=req.k_per_video)
    context = build_context_string(sources)

    # Use a combined session key for multi-video history
    session_key = "_vs_".join(sorted(req.video_ids))
    history_turns = mem.get_window(session_key, n=4)
    history_str = mem.format_history_for_prompt(history_turns)

    try:
        answer = generate_answer(
            context=context,
            question=req.question,
            mode="compare",
            history=history_str,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

    mem.add_turn(session_key, req.question, answer)

    return AskResponse(
        answer=answer,
        sources=[SourceChunk(**s) for s in sources],
        mode="compare",
    )


@router.get("/history/{video_id}", response_model=list[HistoryTurn])
def get_history(video_id: str):
    """Return the full conversation history for a video session."""
    return mem.get_all(video_id)


@router.delete("/history/{video_id}")
def clear_history(video_id: str):
    """Clear the conversation history for a video session."""
    mem.clear(video_id)
    return {"status": "cleared", "video_id": video_id}


@router.get("/videos")
def list_videos():
    """
    List all videos that have been ingested (have a persisted embedding store).
    """
    import os, json
    base = os.path.join(os.path.dirname(__file__), "yt_chroma_db")
    if not os.path.exists(base):
        return {"videos": []}
    videos = []
    for entry in os.scandir(base):
        if entry.is_dir():
            meta_path = os.path.join(entry.path, "meta.json")
            chroma_path = os.path.join(entry.path, "chroma")
            if os.path.exists(chroma_path):
                title = entry.name
                if os.path.exists(meta_path):
                    with open(meta_path, "r", encoding="utf-8") as f:
                        title = json.load(f).get("title", entry.name)
                videos.append({"video_id": entry.name, "title": title})
    return {"videos": videos}
