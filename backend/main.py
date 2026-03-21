"""
main.py — FastAPI application entry point

Starts the YT RAG Assistant backend.
Run with:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

from dotenv import load_dotenv
load_dotenv()  # Load .env before importing modules that use env vars

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api import router

app = FastAPI(
    title="YT RAG Assistant API",
    description="Production-level RAG backend for the YouTube RAG Chrome Extension",
    version="2.0.0",
)

# Allow Chrome extension to make requests (extensions have chrome-extension:// origin)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok", "version": "2.0.0"}
