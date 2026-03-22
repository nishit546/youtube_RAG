"""
memory/session.py

In-memory chat history store, keyed per video session.
Each session holds a list of { "question": str, "answer": str } turns.

This is intentionally simple (in-process dict) for a single-user desktop
extension. For multi-user production, replace with Redis or a DB-backed store.
"""

from collections import defaultdict
from typing import Optional

# ── Session store (singleton) ──
_sessions: dict[str, list[dict]] = defaultdict(list)


def add_turn(video_id: str, question: str, answer: str) -> None:
    """Append a question/answer pair to the session."""
    _sessions[video_id].append({"question": question, "answer": answer})


def get_window(video_id: str, n: int = 4) -> list[dict]:
    """Return the last n turns for the given video session."""
    return _sessions[video_id][-n:]


def get_all(video_id: str) -> list[dict]:
    """Return the full conversation history for a video."""
    return list(_sessions[video_id])


def clear(video_id: str) -> None:
    """Clear the conversation history for a video."""
    _sessions[video_id] = []


def format_history_for_prompt(history: list[dict]) -> str:
    """
    Format chat history into a readable string for injection into prompts.

    Example output:
        User: What is this video about?
        Assistant: This video explains.
        User: Tell me more about X.
        Assistant: X refers to.
    """
    if not history:
        return ""
    lines = []
    for turn in history:
        lines.append(f"User: {turn['question']}")
        lines.append(f"Assistant: {turn['answer']}")
    return "\n".join(lines)
