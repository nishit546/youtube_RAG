"""
generation/generator.py

Mode-aware answer generation using Google Gemini.

Smart Modes:
  - "ask"         → Default RAG Q&A (uses retrieved context strictly)
  - "summary"     → Concise TL;DR of the entire video
  - "keypoints"   → Bulleted key insights
  - "deep"        → Detailed explanation with examples
  - "quiz"        → Generate quiz questions from content
  - "navigate"    → Locate where a topic is discussed (returns timestamps)

Anti-hallucination: every prompt instructs the model to answer ONLY
from the provided context. If not found, it says "Not in video."
"""

import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)

# ── Singleton LLM ─────────────────────────────────────────────────────────────
_llm: ChatGoogleGenerativeAI | None = None


def get_llm() -> ChatGoogleGenerativeAI:
    """Return shared Gemini LLM instance (lazy-loaded singleton)."""
    global _llm
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    return _llm


# ── Mode-specific prompt templates ────────────────────────────────────────────
_TEMPLATES = {
    "ask": """You are an intelligent assistant for a YouTube video.
Answer ONLY using the context below. If the answer is not in the context, say: "Not in video."
Cite specific timestamps when relevant (e.g., "As mentioned at 2:34...").

Chat History:
{history}

Context (from video transcript):
{context}

Question: {question}

Answer:""",

    "summary": """You are summarizing a YouTube video based on its transcript.
Write a concise TL;DR summary (3–5 sentences) based ONLY on the context below.
Start with "Summary:" and be informative but brief.

Context (from video transcript):
{context}

Provide a concise summary:""",

    "keypoints": """You are extracting key insights from a YouTube video.
Based ONLY on the context below, extract 5–8 bullet-point key takeaways.
Format: • [Key insight]
Only include what is explicitly in the transcript context.

Context (from video transcript):
{context}

Key Points:""",

    "deep": """You are a deep-dive explainer for a YouTube video.
Provide a thorough, detailed explanation based ONLY on the context below.
Include examples from the transcript, explain terminology, and give background where supported.
Cite timestamps where relevant (e.g., "At 3:15, the speaker explains...").

Chat History:
{history}

Context (from video transcript):
{context}

Question: {question}

Detailed Explanation:""",

    "quiz": """You are generating a quiz from a YouTube video transcript.
Based ONLY on the context below, create 5 multiple-choice questions with 4 options each.
Mark the correct answer with ✓.
Format:
Q1. [Question]
   A) ...
   B) ...  ✓
   C) ...
   D) ...

Context (from video transcript):
{context}

Quiz:""",

    "navigate": """You are a video navigation assistant.
The user wants to find where a specific topic, concept, or moment is discussed in the video.
Based ONLY on the context below, identify ALL relevant segments where this is discussed.
Return EACH match as:
  📍 [M:SS – M:SS] — [brief description of what's discussed]

If not found, say: "This topic is not covered in the retrieved segments."

Context (from video transcript):
{context}

Topic to find: {question}

Relevant segments:""",

    "compare": """You are comparing content from multiple YouTube videos.
Answer the user's question by drawing comparisons based ONLY on the contexts below.
Clearly label which video you are referencing for each point.
If a video doesn't cover a topic, say so explicitly.

Chat History:
{history}

Context from multiple videos:
{context}

Question: {question}

Comparative Answer:""",
}


def _get_template(mode: str) -> str:
    return _TEMPLATES.get(mode, _TEMPLATES["ask"])


# ── Main generation function ──────────────────────────────────────────────────
def generate_answer(
    context: str,
    question: str,
    mode: str = "ask",
    history: str = "",
) -> str:
    """
    Generate an answer using Gemini with the specified mode.

    Args:
        context: formatted string of retrieved transcript chunks
        question: user's question or topic
        mode: one of ask | summary | keypoints | deep | quiz | navigate | compare
        history: formatted chat history string (from memory.session)

    Returns:
        Generated answer string
    """
    llm = get_llm()
    parser = StrOutputParser()
    template_str = _get_template(mode)

    # Determine which variables the template needs
    input_vars = ["context"]
    if "{question}" in template_str:
        input_vars.append("question")
    if "{history}" in template_str:
        input_vars.append("history")

    prompt = PromptTemplate(template=template_str, input_variables=input_vars)
    chain = prompt | llm | parser

    inputs = {"context": context}
    if "question" in input_vars:
        inputs["question"] = question
    if "history" in input_vars:
        inputs["history"] = history or "No previous conversation."

    logger.info(f"[generator] Generating answer — mode={mode}")
    return chain.invoke(inputs)
