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

# Reuse the project's own scenario-analysis logic when scenario_batch_rag.py is
# present. This gives the richer scenario schema (role/intended_use/.../risk_level)
# and the exact risk calibration heuristics (e.g. subliminal -> prohibited).
try:
    from scenario_batch_rag import (
        build_scenario_query,
        build_scenario_messages,
        extract_risk_level,
        extract_obligations,
        extract_articles,
        calibrate_risk_level,
    )
    SCENARIO_AVAILABLE = True
except Exception:  # pragma: no cover - degrade to the simple JSON path
    SCENARIO_AVAILABLE = False


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


class Scenario(BaseModel):
    role: str = "Unknown"
    intended_use: str = ""
    system_type: str = ""
    input_data: str = ""
    domain: str = ""
    related_articles: List[int] = []
    obligations: List[str] = []
    risk_level: str = "unknown"


class ChatResponse(BaseModel):
    analysis: str
    risk_level: str
    obligations: List[str]
    articles: List[int]
    sources: List[Source]
    used_deep_research: bool = False
    processing_time: float
    # The full scenario schema, suitable for "Copy JSON" in the UI.
    scenario: Optional[Scenario] = None


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


def _extract_scenario_fields(message: str, document_text: Optional[str]) -> Dict[str, str]:
    """Use the LLM to turn the user's free-text message into scenario fields."""
    blob = message.strip()
    if document_text and document_text.strip():
        blob += "\n\nAttached document:\n" + document_text.strip()[:6000]

    fields = {"role": "", "intended_use": "", "system_type": "", "input_data": "", "domain": ""}
    messages = [
        {
            "role": "system",
            "content": (
                "Extract AI-system scenario fields from the user's text. "
                "Return ONLY a JSON object with keys: role, intended_use, "
                "system_type, input_data, domain. Use \"\" for anything not stated. "
                "role is 'Provider' or 'Deployer' when inferable."
            ),
        },
        {"role": "user", "content": blob},
    ]
    try:
        parsed = json.loads(_strip_fences(_complete(messages)))
        if isinstance(parsed, dict):
            for key in fields:
                value = parsed.get(key)
                if isinstance(value, str):
                    fields[key] = value.strip()
    except Exception:
        pass

    # Always keep the raw question as intended_use if extraction gave nothing.
    if not fields["intended_use"]:
        fields["intended_use"] = message.strip()
    if not fields["role"]:
        fields["role"] = "Provider"
    return fields


def _strip_fences(text: str) -> str:
    cleaned = (text or "").strip()
    cleaned = re.sub(r"^```json\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.strip().strip("`").strip()
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    return match.group(0) if match else cleaned


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


def _clean_obligations(raw: List[str]) -> List[str]:
    """Tidy the obligations parsed from free-form analysis text: strip markdown,
    drop section headers/fragments, and de-duplicate while keeping order."""
    cleaned: List[str] = []
    seen: set = set()
    for item in raw:
        text = re.sub(r"^[\s\*#\-\d\.\)]+", "", str(item)).replace("**", "").strip()
        if len(text) < 25:
            continue
        # Drop leftover fragments that start mid-word/lowercase (e.g. "s: If ...").
        if not text[0].isupper():
            continue
        # Drop bare article references; articles are listed separately.
        if re.match(r"^Article\s+\d", text):
            continue
        # Skip headings like "Specific Obligations and Requirements".
        if ":" not in text and not re.search(r"\b(must|shall|should|ensure|provide|maintain|conduct|implement|register|cease|halt|notify|comply|document|carry out|establish)\b", text, re.IGNORECASE):
            continue
        key = text.lower()[:50]
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(text)
    return cleaned[:12]


def _empty_response(analysis: str, start: float, used_deep_research: bool) -> ChatResponse:
    return ChatResponse(
        analysis=analysis,
        risk_level="unknown",
        obligations=[],
        articles=[],
        sources=[],
        used_deep_research=used_deep_research,
        processing_time=time.time() - start,
    )


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    start = time.time()
    question = req.message.strip()

    # Build the scenario (role/intended_use/...) and the retrieval query. When the
    # project's scenario_batch_rag.py is available we use its exact query builder.
    scenario_fields = _extract_scenario_fields(question, req.document_text)
    if SCENARIO_AVAILABLE:
        query = build_scenario_query(scenario_fields)
    else:
        query = question

    docs, metas, scores = pre_retrieve(query, fast=req.fast)

    # Incorporate the user's uploaded document(s) as an extra, citeable source.
    if req.document_text and req.document_text.strip():
        docs = list(docs) + [req.document_text.strip()]
        metas = list(metas) + [
            {"source": "user-upload", "doc_id": "uploaded-document", "chunk_index": 0}
        ]
        scores = list(scores) + [1.0]

    if not docs:
        # No grounding context: fall back to iterative deep research if available.
        if DEEP_RESEARCH_AVAILABLE and run_deep_research is not None:
            try:
                research = run_deep_research(question)
                analysis = research if isinstance(research, str) else str(research)
            except Exception as exc:  # pragma: no cover
                analysis = f"Deep research fallback failed: {exc}"
            return _empty_response(analysis, start, used_deep_research=True)
        return _empty_response(
            "No relevant context was found for this question.", start, False
        )

    context_block = build_context(docs, metas)
    sources = _sources_from_metas(metas, scores)

    if SCENARIO_AVAILABLE:
        # Mirror scenario_batch_rag: free-form analysis, then extract + calibrate.
        analysis = _complete(build_scenario_messages(query, context_block)).strip()
        generated_risk = extract_risk_level(analysis)
        obligations = _clean_obligations(extract_obligations(analysis))
        articles = extract_articles(analysis)
        risk_level = calibrate_risk_level(generated_risk, scenario_fields)
        if risk_level not in ALLOWED_RISK:
            risk_level = "unknown"
        scenario = Scenario(
            role=scenario_fields.get("role", "Unknown") or "Unknown",
            intended_use=scenario_fields.get("intended_use", ""),
            system_type=scenario_fields.get("system_type", ""),
            input_data=scenario_fields.get("input_data", ""),
            domain=scenario_fields.get("domain", ""),
            related_articles=articles,
            obligations=obligations,
            risk_level=risk_level,
        )
        return ChatResponse(
            analysis=analysis,
            risk_level=risk_level,
            obligations=obligations,
            articles=articles,
            sources=sources,
            used_deep_research=False,
            processing_time=time.time() - start,
            scenario=scenario,
        )

    # Fallback path (scenario_batch_rag.py not present): strict-JSON classification.
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
        sources=sources,
        used_deep_research=False,
        processing_time=time.time() - start,
        scenario=Scenario(
            role=scenario_fields.get("role", "Unknown") or "Unknown",
            intended_use=scenario_fields.get("intended_use", ""),
            system_type=scenario_fields.get("system_type", ""),
            input_data=scenario_fields.get("input_data", ""),
            domain=scenario_fields.get("domain", ""),
            related_articles=parsed["articles"],
            obligations=parsed["obligations"],
            risk_level=parsed["risk_level"],
        ),
    )
