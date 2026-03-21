"""
ingestion/transcript.py

Fetches YouTube transcripts with timestamps preserved.
Falls back to Whisper STT if no transcript is available.

Each chunk is returned as:
  { "text": str, "start": float, "end": float, "video_id": str, "title": str }
"""

import json
import os
import logging
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound
try:
    # v1.x raises TranscriptsDisabled as a different exception name
    from youtube_transcript_api import TranscriptsDisabled
except ImportError:
    TranscriptsDisabled = Exception  # fallback

logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(BASE_DIR, "yt_chroma_db")


def _cache_path(video_id: str) -> str:
    folder = os.path.join(CACHE_DIR, video_id)
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, "transcript.json")


def _title_cache_path(video_id: str) -> str:
    folder = os.path.join(CACHE_DIR, video_id)
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, "meta.json")


# ── Title fetching ────────────────────────────────────────────────────────────
def _get_video_title(video_id: str) -> str:
    """Attempt to retrieve the video title; returns a fallback on failure."""
    meta_path = _title_cache_path(video_id)
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f).get("title", video_id)

    title = video_id  # default fallback
    try:
        # yt-dlp is the most reliable title fetcher; it may not be installed.
        import yt_dlp  # type: ignore
        ydl_opts = {"quiet": True, "skip_download": True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(
                f"https://www.youtube.com/watch?v={video_id}", download=False
            )
            title = info.get("title", video_id)
    except Exception:
        pass  # title stays as video_id fallback

    # Cache result
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"title": title, "video_id": video_id}, f)

    return title


# ── Semantic chunking ─────────────────────────────────────────────────────────
def _semantic_chunk(
    entries: list[dict],
    target_words: int = 80,
    overlap_entries: int = 1,
) -> list[dict]:
    """
    Combine raw transcript entries (each ~1–5 words) into semantic chunks
    of ~target_words words while preserving start/end timestamps.

    Args:
        entries: list of {text, start, duration} from YouTubeTranscriptApi
        target_words: approx word count per chunk
        overlap_entries: number of entries to re-include from previous chunk
    """
    chunks = []
    current_texts = []
    current_start = None
    current_end = None
    word_count = 0
    prev_tail = []  # for overlap

    for entry in entries:
        text = entry["text"].strip()
        start = entry["start"]
        end = start + entry.get("duration", 0)
        words = len(text.split())

        if current_start is None:
            current_start = start

        current_texts.append(text)
        current_end = end
        word_count += words

        if word_count >= target_words:
            chunk_text = " ".join(current_texts)
            chunks.append({
                "text": chunk_text,
                "start": current_start,
                "end": current_end,
            })
            # carry overlap into next chunk
            prev_tail = current_texts[-overlap_entries:] if overlap_entries else []
            current_texts = list(prev_tail)
            word_count = sum(len(t.split()) for t in current_texts)
            current_start = start  # rough: start of last overlap entry
            current_end = end

    # flush remaining
    if current_texts:
        chunks.append({
            "text": " ".join(current_texts),
            "start": current_start or 0.0,
            "end": current_end or 0.0,
        })

    return chunks


# ── Public API ────────────────────────────────────────────────────────────────
def load_transcript(video_id: str, force_refresh: bool = False) -> dict:
    """
    Load and chunk a YouTube video transcript.

    Returns:
        {
            "video_id": str,
            "title": str,
            "chunks": [{"text", "start", "end", "video_id", "title"}],
            "cached": bool,       # True if loaded from disk cache
            "source": "youtube" | "whisper"
        }

    Raises:
        RuntimeError if transcript is unavailable and Whisper fallback fails.
    """
    cache_path = _cache_path(video_id)

    # ── Return cached transcript if available and not forced ──────────────────
    if not force_refresh and os.path.exists(cache_path):
        logger.info(f"[transcript] Loading from cache: {video_id}")
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["cached"] = True
        return data

    # ── Attempt YouTube transcript ────────────────────────────────────────────
    raw_entries = None
    source = "youtube"
    try:
        # youtube-transcript-api v1.x: use list_transcripts + fetch()
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        # Try to find any available transcript (manual first, then auto-generated)
        try:
            transcript = transcript_list.find_transcript(["en", "en-US", "en-GB"])
        except NoTranscriptFound:
            # Fall back to the first available transcript in any language
            transcript = next(iter(transcript_list))
        fetched = transcript.fetch()
        # Convert to plain dicts for compatibility with the rest of the pipeline
        raw_entries = [
            {
                "text": getattr(snippet, "text", snippet.get("text", "") if isinstance(snippet, dict) else ""),
                "start": getattr(snippet, "start", snippet.get("start", 0.0) if isinstance(snippet, dict) else 0.0),
                "duration": getattr(snippet, "duration", snippet.get("duration", 0.0) if isinstance(snippet, dict) else 0.0),
            }
            for snippet in fetched
        ]
        logger.info(f"[transcript] Fetched YouTube transcript for {video_id} ({len(raw_entries)} entries)")
    except NoTranscriptFound:
        logger.warning(f"[transcript] No YT transcript for {video_id}, trying Whisper")
    except Exception as e:
        logger.warning(f"[transcript] YT transcript error for {video_id}: {e}")

    # ── Whisper fallback ──────────────────────────────────────────────────────
    if raw_entries is None:
        from ingestion.whisper_fallback import transcribe_with_whisper
        raw_entries = transcribe_with_whisper(video_id)
        source = "whisper"

    # ── Fetch title ───────────────────────────────────────────────────────────
    title = _get_video_title(video_id)

    # ── Chunk with timestamps ─────────────────────────────────────────────────
    raw_chunks = _semantic_chunk(raw_entries)
    chunks = [
        {
            "text": c["text"],
            "start": c["start"],
            "end": c["end"],
            "video_id": video_id,
            "title": title,
        }
        for c in raw_chunks
    ]

    result = {
        "video_id": video_id,
        "title": title,
        "chunks": chunks,
        "cached": False,
        "source": source,
    }

    # ── Persist to cache ──────────────────────────────────────────────────────
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(result, f)

    return result
