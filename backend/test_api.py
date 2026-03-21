"""
test_api.py — Basic integration tests for YT RAG Assistant API

Run with:
    cd backend
    pip install pytest httpx
    pytest test_api.py -v

NOTE: These tests require the server to be running:
    uvicorn main:app --host 0.0.0.0 --port 8000

A public video with a known transcript is used: dQw4w9WgXcQ
(Rick Astley - Never Gonna Give You Up, always has a transcript)
"""

import pytest
import httpx

BASE = "http://localhost:8000"
TEST_VIDEO = "dQw4w9WgXcQ"
TEST_VIDEO_2 = "aircAruvnKk"  # 3Blue1Brown — Neural Networks

client = httpx.Client(base_url=BASE, timeout=120.0)


# ── /health ───────────────────────────────────────────────────────────────────
def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


# ── /ingest ───────────────────────────────────────────────────────────────────
def test_ingest_video():
    res = client.post("/ingest", json={"video_id": TEST_VIDEO})
    assert res.status_code == 200, res.text
    data = res.json()
    assert data["video_id"] == TEST_VIDEO
    assert isinstance(data["chunk_count"], int) and data["chunk_count"] > 0
    assert data["source"] in ("youtube", "whisper")
    print(f"\n  ✓ Ingested: {data['title']} ({data['chunk_count']} chunks, cached={data['cached']})")


def test_ingest_cached():
    """Second ingest of the same video should return cached=True."""
    res = client.post("/ingest", json={"video_id": TEST_VIDEO})
    assert res.status_code == 200
    data = res.json()
    assert data["cached"] is True
    print(f"\n  ✓ From cache: {data['title']}")


def test_ingest_invalid_video():
    """Invalid video ID should return 400."""
    res = client.post("/ingest", json={"video_id": "invalid_video_id_xyz"})
    assert res.status_code in (400, 422)


# ── /ask ─────────────────────────────────────────────────────────────────────
def test_ask_default():
    res = client.post("/ask", json={
        "video_id": TEST_VIDEO,
        "question": "What is the main message of the song?",
        "mode": "ask",
    })
    assert res.status_code == 200, res.text
    data = res.json()
    assert data["answer"] and len(data["answer"]) > 10
    assert isinstance(data["sources"], list)
    print(f"\n  ✓ Answer: {data['answer'][:100]}…")
    print(f"  ✓ Sources: {len(data['sources'])} chunks")


def test_ask_returns_sources_with_timestamps():
    res = client.post("/ask", json={
        "video_id": TEST_VIDEO,
        "question": "What does the singer promise?",
        "mode": "ask",
    })
    assert res.status_code == 200
    data = res.json()
    if data["sources"]:
        src = data["sources"][0]
        assert "start" in src and "end" in src
        assert "start_fmt" in src and "url" in src
        print(f"\n  ✓ First source: {src['start_fmt']} – {src['end_fmt']}")


def test_ask_summary_mode():
    res = client.post("/ask", json={
        "video_id": TEST_VIDEO,
        "question": "Summarize this video",
        "mode": "summary",
    })
    assert res.status_code == 200
    data = res.json()
    assert data["mode"] == "summary"
    assert "Summary" in data["answer"] or len(data["answer"]) > 20
    print(f"\n  ✓ Summary: {data['answer'][:120]}…")


def test_ask_keypoints_mode():
    res = client.post("/ask", json={
        "video_id": TEST_VIDEO,
        "question": "Key points",
        "mode": "keypoints",
    })
    assert res.status_code == 200
    data = res.json()
    assert data["mode"] == "keypoints"
    print(f"\n  ✓ Key points answer length: {len(data['answer'])}")


def test_ask_quiz_mode():
    res = client.post("/ask", json={
        "video_id": TEST_VIDEO,
        "question": "Generate a quiz",
        "mode": "quiz",
    })
    assert res.status_code == 200
    data = res.json()
    assert data["mode"] == "quiz"
    print(f"\n  ✓ Quiz: {data['answer'][:120]}…")


def test_ask_navigate_mode():
    res = client.post("/ask", json={
        "video_id": TEST_VIDEO,
        "question": "Where does the singer talk about promises?",
        "mode": "navigate",
    })
    assert res.status_code == 200
    data = res.json()
    assert data["mode"] == "navigate"
    print(f"\n  ✓ Navigate: {data['answer'][:120]}…")


def test_ask_unloaded_video():
    """Asking about a video that hasn't been ingested should return 404."""
    res = client.post("/ask", json={
        "video_id": "notingested99",
        "question": "What is this about?",
        "mode": "ask",
    })
    assert res.status_code == 404


# ── /history ─────────────────────────────────────────────────────────────────
def test_get_history():
    res = client.get(f"/history/{TEST_VIDEO}")
    assert res.status_code == 200
    data = res.json()
    assert isinstance(data, list)
    print(f"\n  ✓ History: {len(data)} turns")


def test_clear_history():
    res = client.delete(f"/history/{TEST_VIDEO}")
    assert res.status_code == 200
    assert res.json()["status"] == "cleared"
    # Confirm it's empty
    res2 = client.get(f"/history/{TEST_VIDEO}")
    assert res2.json() == []


# ── /videos ──────────────────────────────────────────────────────────────────
def test_list_videos():
    res = client.get("/videos")
    assert res.status_code == 200
    data = res.json()
    assert "videos" in data
    print(f"\n  ✓ Videos: {[v['video_id'] for v in data['videos']]}")


# ── /compare ─────────────────────────────────────────────────────────────────
def test_compare_two_videos():
    # Ingest second video first
    client.post("/ingest", json={"video_id": TEST_VIDEO_2})
    res = client.post("/compare", json={
        "video_ids": [TEST_VIDEO, TEST_VIDEO_2],
        "question": "Compare the topics of these two videos.",
        "k_per_video": 2,
    })
    assert res.status_code == 200, res.text
    data = res.json()
    assert data["mode"] == "compare"
    assert len(data["sources"]) > 0
    print(f"\n  ✓ Compare answer: {data['answer'][:120]}…")
