
<h1>🎬 YouTube RAG Assistant</h1>

<h3>Chrome Extension to Chat with Any YouTube Video Using AI</h3>

<p>
<a href="https://python.org" target="_blank">
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
</a>
<a href="https://fastapi.tiangolo.com" target="_blank">
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI">
</a>
<a href="https://ai.google.dev" target="_blank">
  <img src="https://img.shields.io/badge/Gemini_2.5_Flash-AI-4285F4?style=for-the-badge&logo=google&logoColor=white" alt="Gemini">
</a>
<a href="https://python.langchain.com/" target="_blank">
  <img src="https://img.shields.io/badge/LangChain-RAG-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white" alt="LangChain">
</a>
<a href="https://developer.chrome.com/docs/extensions/" target="_blank">
  <img src="https://img.shields.io/badge/Chrome-Extension_MV3-4285F4?style=for-the-badge&logo=googlechrome&logoColor=white" alt="Chrome Extension">
</a>
<a href="LICENSE" target="_blank">
  <img src="https://img.shields.io/badge/License-MIT-gold?style=for-the-badge" alt="License">
</a>
</p>

<p>
  <em>Enter a YouTube video ID, ask any question, and get AI-powered answers grounded in the video's transcript — all from a sleek Chrome extension.</em>
</p>

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Getting Started](#-getting-started)
- [How It Works](#-how-it-works)
- [API Reference](#-api-reference)
- [Project Structure](#-project-structure)
- [Troubleshooting](#-troubleshooting)
- [License](#-license)

---

## 🔍 Overview

**YouTube RAG Assistant** brings the power of **Retrieval-Augmented Generation (RAG)** to YouTube. Instead of watching an entire video, you can:

1. Provide a **YouTube Video ID**
2. Ask a **question in natural language**
3. Get an **accurate, transcript-grounded answer** powered by **Google Gemini 2.5 Flash**

The system automatically downloads the video transcript, chunks it, embeds it into a vector store, and runs a full RAG pipeline to answer your question — all in real time.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🎥 **Transcript Q&A** | Ask anything about a YouTube video and get answers from its transcript |
| 🤖 **Gemini 2.5 Flash** | Fast, accurate language model for natural-language understanding |
| 🧠 **RAG Pipeline** | Retrieval-Augmented Generation ensures factual, grounded responses |
| 🔍 **Semantic Search** | ChromaDB + HuggingFace embeddings for intelligent context retrieval |
| 💬 **Chat Interface** | Beautiful dark-themed popup with animated chat bubbles |
| ⚡ **Real-Time** | Processes transcripts and answers on-the-fly |

---

## 🏗 Architecture

```
┌──────────────────────────────────────┐
│        Chrome Extension (MV3)        │
│  ┌────────────┐   ┌───────────────┐ │
│  │ popup.html │   │   popup.js    │ │
│  │ (Chat UI)  │   │ (API calls)   │ │
│  └─────┬──────┘   └───────┬───────┘ │
└────────┼───────────────────┼─────────┘
         │  POST /ask        │
         ▼                   ▼
┌──────────────────────────────────────┐
│         FastAPI Backend (api.py)     │
│                                      │
│  1. YouTube Transcript Loader        │
│         │                            │
│         ▼                            │
│  2. RecursiveCharacterTextSplitter   │
│         │                            │
│         ▼                            │
│  3. ChromaDB Vector Store            │
│     (HuggingFace Embeddings)         │
│         │                            │
│         ▼                            │
│  4. LangChain RAG Chain              │
│     (Retriever → Prompt → LLM)      │
│         │                            │
│         ▼                            │
│  5. Google Gemini 2.5 Flash          │
│         │                            │
│         ▼                            │
│  6. Parsed Answer → JSON Response    │
└──────────────────────────────────────┘
```

---

## 🛠 Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **LLM** | Google Gemini 2.5 Flash | Answer generation from transcript context |
| **RAG Framework** | LangChain | Pipeline orchestration — loader, splitter, retriever, chain |
| **Transcript** | `YoutubeLoader` (LangChain) | Automatic YouTube transcript extraction |
| **Vector DB** | ChromaDB | Semantic search over transcript chunks |
| **Embeddings** | HuggingFace `all-MiniLM-L6-v2` | Sentence-level embeddings |
| **Backend** | FastAPI + Uvicorn | Async REST API |
| **Frontend** | Chrome Extension (Manifest V3) | Chat interface popup |
| **Styling** | Vanilla CSS | Dark theme, indigo gradients, smooth animations |

---

## 🚀 Getting Started

### Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.8+ |
| Google Chrome | Latest |
| Google API Key | [Get one here](https://makersuite.google.com/app/apikey) |

### 1️⃣ Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# Install dependencies
pip install fastapi uvicorn python-dotenv langchain langchain-google-genai langchain-community chromadb sentence-transformers

# Create .env file
echo GOOGLE_API_KEY=your_api_key_here > .env
```

### 2️⃣ Start the Server

```bash
uvicorn api:app --reload --port 8000
```

API available at **`http://localhost:8000`**.

### 3️⃣ Load Chrome Extension

1. Open **`chrome://extensions/`**
2. Enable **Developer mode**
3. Click **"Load unpacked"**
4. Select the `frontend/` folder
5. Pin the extension

### 4️⃣ Try It Out!

1. Click the extension icon
2. Enter a **YouTube Video ID** (e.g., `aircAruvnKk`)
3. Type a question: *"What is this video about?"*
4. Click **"Ask"** and get your answer!

> **Tip:** The Video ID is the part after `v=` in a YouTube URL.  
> For `https://www.youtube.com/watch?v=aircAruvnKk`, the ID is `aircAruvnKk`.

---

## 📡 API Reference

### `POST /ask`

Send a video ID and question to get a transcript-grounded answer.

**Request Body:**
```json
{
  "video_id": "aircAruvnKk",
  "question": "What are the main topics discussed?"
}
```

**Response:**
```json
{
  "answer": "The video discusses neural networks, specifically..."
}
```

**Error Response (no transcript available):**
```json
{
  "detail": "Transcript not available for this video"
}
```

---

## ⚙️ How It Works

1. **Transcript Loading** — `YoutubeLoader` fetches the auto-generated or manual transcript
2. **Text Splitting** — `RecursiveCharacterTextSplitter` chunks transcript into 1000-char pieces with 200-char overlap
3. **Embedding & Storage** — Chunks embedded with `all-MiniLM-L6-v2` and stored in ChromaDB
4. **Retrieval** — Top 4 most relevant chunks retrieved for each question
5. **RAG Chain** — Retrieved context + question fed through a LangChain prompt template
6. **LLM Answer** — Gemini 2.5 Flash generates a grounded answer
7. **Response** — Answer returned to the extension and displayed in the chat

---

## 📁 Project Structure

```
yt_video_chatbot/
├── backend/
│   ├── api.py               # FastAPI backend — full RAG pipeline
│   └── yt_chroma_db/        # ChromaDB persistent storage (auto-generated)
│
└── frontend/
    ├── manifest.json         # Chrome Extension config (Manifest V3)
    ├── popup.html            # Chat UI — dark indigo theme, animations
    └── popup.js              # Extension logic — API calls, chat rendering
```

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| **"Transcript not available"** | The video may not have captions — try a different video |
| **Server not running** | Run `uvicorn api:app --reload --port 8000` |
| **API key error** | Add `GOOGLE_API_KEY` to `backend/.env` |
| **Extension not loading** | Enable Developer mode at `chrome://extensions/` |
| **Slow first request** | HuggingFace embeddings model downloads on first run (~90MB) |
| **Wrong video ID** | Use only the ID part (e.g., `aircAruvnKk`), not the full URL |

---

## 📄 License

MIT License — feel free to modify and use for your own projects.

---

<div align="center">

**Built with ❤️ for Knowledge Seekers** 🎬🤖

*Ask a video anything — powered by RAG + Gemini AI.*

</div>
