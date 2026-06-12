#!/usr/bin/env python3
"""
FastAPI server exposing the existing RAG pipeline to the chatbot frontend.

It reuses the project's own modules (place them in this folder):
  - rag_chat.py            (pre_retrieve, build_context, LLM config)
  - openai_deep_research.py (run_deep_research, used as a fallback)

Endpoints:
  GET  /health  -> {"status": "ok", "backend": "...", "model": "..."}
  POST /chat    -> structured EU AI Act classification with citations

Run:
  export OPENAI_API_KEY="sk-..."
  # make sure ChromaDB is up (chroma run --host localhost --port 8000)
  uvicorn server:app --host 0.0.0.0 --port 8000 --reload   # or any port
"""

from __future__ import annotations

import os
import re
import json
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Reuse the existing pipeline. These imports require the project's own files to
# sit next to this one (and their deps installed: chromadb, sentence-transformers...).
from rag_chat import (
    pre_retrieve,
    build_context,
    LLM_BACKEND,
    OPENAI_MODEL,
    OLLAMA_MODEL,
)

try:
    from openai_deep_research import run_deep_research
    DEEP_RESEARCH_AVAILABLE = True
except Exception:  # pragma: no cover - fallback is optional
    run_deep_research = None  # type: ignore
    DEEP_RESEARCH_AVAILABLE = False


ALLOWED_RISK = {"prohibited", "high-risk", "limited-risk", "minimal-risk", "unknown"}

# Comma-separated list of allowed origins for CORS (frontend dev server etc.).
CORS_ORIGINS = [
    o.strip()
    for o in os.environ.get(
        "CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
    ).split(",")
    if o.strip()
]

app = FastAPI(title="EU AI Act Assistant API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    # Plain-text contents of any documents the user attached in the UI.
    document_text: Optional[str] = None
    fast: bool = False


class Source(BaseModel):
    index: int
    source: str
    doc_id: str
    chunk_index: Any = None
    score: Optional[float] = None


class ChatResponse(BaseModel):
    analysis: str
    risk_level: str
    obligations: List[str]
    articles: List[int]
    sources: List[Source]
    used_deep_research: bool = False
    processing_time: float


def _structured_messages(question: str, context_block: str) -> List[Dict[str, str]]:
    """Prompt the model for a strict JSON classification (mirrors batch_rag.py)."""
    system = (
        "You are an AI Act classification assistant. Use ONLY the provided context. "
        "Return a JSON object with this exact schema (no prose, no markdown fences):\n"
        "{\n"
        '  "analysis": string, // concise 3-5 sentence summary with citations,\n'
        "  \"risk_level\": one of ['prohibited','high-risk','limited-risk','minimal-risk','unknown'],\n"
        '  "obligations": array of 3-6 short obligation bullets (each ending with a citation),\n'
        '  "articles": array of unique Article numbers as integers\n'
        "}"
    )
    user = (
        f"Question: {question}\n\n"
        f"Context:\n{context_block}\n\n"
        "Instructions:\n"
        "1. Base every statement on the context and cite like [source; doc_id; chunk_index=K].\n"
        "2. Choose the risk_level per EU AI Act categories.\n"
        "3. Obligations must be actionable compliance steps with citations.\n"
        "4. Articles must be numeric (e.g., 5, 10, 27).\n"
        "5. Output ONLY valid JSON."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _parse_structured(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort parse of the model's JSON answer (mirrors batch_rag.py)."""
    if not text:
        return None
    cleaned = text.strip()
    cleaned = re.sub(r"^```json\s*", "", cleaned, flags=re.IGNORECASE).rstrip("`")
    cleaned = cleaned.strip().rstrip("```").strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to grab the first {...} block.
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    if not isinstance(data, dict):
        return None

    obligations = data.get("obligations") or []
    if isinstance(obligations, str):
        obligations = [o.strip() for o in obligations.split("\n") if o.strip()]
    else:
        obligations = [str(o).strip() for o in obligations if str(o).strip()]

    articles_raw = data.get("articles") or []
    articles: List[int] = []
    if isinstance(articles_raw, list):
        for art in articles_raw:
            if isinstance(art, int):
                articles.append(art)
            else:
                m = re.search(r"(\d+)", str(art))
                if m:
                    articles.append(int(m.group(1)))

    risk = str(data.get("risk_level", "unknown")).strip().lower()
    return {
        "analysis": str(data.get("analysis", "")).strip(),
        "risk_level": risk if risk in ALLOWED_RISK else "unknown",
        "obligations": obligations,
        "articles": sorted(set(articles)),
    }


def _complete(messages: List[Dict[str, str]]) -> str:
    """Non-streaming completion using the configured backend."""
    if LLM_BACKEND == "openai":
        from openai import OpenAI

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable is not set.")
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=0.2,
            stream=False,
        )
        return resp.choices[0].message.content or ""

    # Ollama non-streaming chat.
    import requests

    resp = requests.post(
        "http://localhost:11434/api/chat",
        json={"model": OLLAMA_MODEL, "messages": messages, "stream": False},
        timeout=600,
    )
    resp.raise_for_status()
    return resp.json().get("message", {}).get("content", "")


def _sources_from_metas(metas: List[Dict[str, Any]], scores: List[float]) -> List[Source]:
    sources: List[Source] = []
    for i, meta in enumerate(metas, start=1):
        meta = meta or {}
        score = scores[i - 1] if i - 1 < len(scores) else None
        sources.append(
            Source(
                index=i,
                source=str(meta.get("source", "unknown")),
                doc_id=str(meta.get("doc_id", "")),
                chunk_index=meta.get("chunk_index", "n/a"),
                score=float(score) if score is not None else None,
            )
        )
    return sources


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "backend": LLM_BACKEND,
        "model": OPENAI_MODEL if LLM_BACKEND == "openai" else OLLAMA_MODEL,
        "deep_research": DEEP_RESEARCH_AVAILABLE,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    start = time.time()
    question = req.message.strip()

    docs, metas, scores = pre_retrieve(question, fast=req.fast)

    # Incorporate the user's uploaded document(s) as an extra, citeable source.
    if req.document_text and req.document_text.strip():
        docs = list(docs) + [req.document_text.strip()]
        metas = list(metas) + [
            {"source": "user-upload", "doc_id": "uploaded-document", "chunk_index": 0}
        ]
        scores = list(scores) + [1.0]

    used_deep_research = False

    if not docs:
        # No grounding context: fall back to iterative deep research if available.
        if DEEP_RESEARCH_AVAILABLE and run_deep_research is not None:
            used_deep_research = True
            try:
                research = run_deep_research(question)
                analysis = research if isinstance(research, str) else str(research)
            except Exception as exc:  # pragma: no cover
                analysis = f"Deep research fallback failed: {exc}"
            return ChatResponse(
                analysis=analysis,
                risk_level="unknown",
                obligations=[],
                articles=[],
                sources=[],
                used_deep_research=True,
                processing_time=time.time() - start,
            )
        return ChatResponse(
            analysis="No relevant context was found for this question.",
            risk_level="unknown",
            obligations=[],
            articles=[],
            sources=[],
            used_deep_research=False,
            processing_time=time.time() - start,
        )

    context_block = build_context(docs, metas)
    raw = _complete(_structured_messages(question, context_block))
    parsed = _parse_structured(raw) or {
        "analysis": raw.strip(),
        "risk_level": "unknown",
        "obligations": [],
        "articles": [],
    }

    return ChatResponse(
        analysis=parsed["analysis"],
        risk_level=parsed["risk_level"],
        obligations=parsed["obligations"],
        articles=parsed["articles"],
        sources=_sources_from_metas(metas, scores),
        used_deep_research=used_deep_research,
        processing_time=time.time() - start,
    )
