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
    build_messages,
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


class Turn(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    # Plain-text contents of any documents the user attached in the UI.
    document_text: Optional[str] = None
    # Prior conversation turns (oldest first), so follow-ups have context.
    history: List[Turn] = []
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
    # How the message was handled: "classification" (risk card), "qa" (general
    # answer with citations, no card), or "clarification" (a follow-up question).
    mode: str = "classification"


def _structured_messages(question: str, context_block: str) -> List[Dict[str, str]]:
    """Prompt the model for a strict JSON AI Act classification.

    Uses an ordered tier test grounded in the AI Act's structure on top of the
    retrieved context. On the project benchmark this raised accuracy from ~85%
    to ~87% and gave full recall on prohibited cases (e.g. subliminal sensory
    nudging), versus the earlier generic prompt.
    """
    system = (
        "You are an EU AI Act risk classifier. Use the provided context as legal "
        "grounding, plus the well-known structure of the AI Act. Decide the SINGLE "
        "best risk tier by checking tiers IN THIS ORDER and stopping at the first match:\n\n"
        "1) PROHIBITED (Art 5) — the system uses subliminal, manipulative or deceptive "
        "techniques that operate beyond a person's conscious awareness, OR exploits "
        "vulnerabilities (age, disability, social/economic situation), to materially "
        "distort behaviour in a way likely to cause harm. This INCLUDES subtle sensory "
        "cues such as low-frequency/inaudible audio, hidden haptics, or flashed images "
        "intended to influence purchasing or behaviour without awareness. Also: social "
        "scoring; untargeted scraping of facial images; emotion inference in workplace/"
        "education; biometric categorisation of sensitive attributes; real-time remote "
        "biometric identification in public by law enforcement; predictive policing based "
        "solely on profiling.\n"
        "2) HIGH-RISK (Annex III) — biometric identification/categorisation; critical "
        "infrastructure safety; education/vocational access & evaluation; employment, "
        "recruitment, worker management; access to essential private/public services "
        "(credit scoring, insurance, benefits, emergency dispatch); law enforcement; "
        "migration/asylum/border control; administration of justice & democratic processes.\n"
        "3) LIMITED-RISK (Art 50) — the ONLY real obligation is transparency/disclosure: "
        "chatbots and systems that interact with people, or that generate/manipulate "
        "content (text, image, audio, video / deepfakes), or permitted emotion-recognition "
        "where users must simply be informed.\n"
        "4) MINIMAL-RISK — everything else, with no specific obligations (spam filters, "
        "recommendation, inventory/forecasting, games, productivity tools).\n\n"
        "Return ONLY a JSON object (no prose, no markdown fences) with this schema:\n"
        "{\n"
        '  "analysis": string,  // 3-5 sentences justifying the tier, cite like [source; doc_id; chunk_index=K]\n'
        "  \"risk_level\": one of ['prohibited','high-risk','limited-risk','minimal-risk','unknown'],\n"
        '  "obligations": array of 4-8 short actionable compliance steps the role must take (each ending with a citation),\n'
        '  "articles": array of every relevant Article number as integers\n'
        "}\n\n"
        "Obligation rules: NEVER return an empty obligations list. For a PROHIBITED "
        "system, the obligations must include immediately halting/withdrawing the "
        "prohibited practice and redesigning to remove it, plus any data-governance "
        "(Art 10), transparency (Art 50), human-oversight and quality-management "
        "(Art 16) duties that still apply. For HIGH-RISK, cover the Art 9-17 duties "
        "(risk & quality management, data governance, technical documentation, "
        "logging, transparency, human oversight, accuracy/robustness, conformity "
        "assessment and registration). For LIMITED-RISK, cover the Art 50 disclosure "
        "duties. List ALL articles relevant to those obligations, not only the one "
        "that triggered the tier."
    )
    user = (
        f"Scenario: {question}\n\n"
        f"Context:\n{context_block}\n\n"
        "Apply the ordered tier test above and output ONLY valid JSON."
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


def _history_text(history: List["Turn"], limit: int = 6) -> str:
    """Render the recent conversation turns as plain text for context."""
    if not history:
        return ""
    lines: List[str] = []
    for turn in history[-limit:]:
        who = "User" if turn.role == "user" else "Assistant"
        content = (turn.content or "").strip()
        if content:
            lines.append(f"{who}: {content}")
    return "\n".join(lines)


def _extract_scenario_fields(
    message: str, document_text: Optional[str], history: Optional[List["Turn"]] = None
) -> Dict[str, str]:
    """Use the LLM to turn the user's free-text message into scenario fields."""
    blob = ""
    convo = _history_text(history or [])
    if convo:
        blob += "Conversation so far:\n" + convo + "\n\nLatest message:\n"
    blob += message.strip()
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
                "role is 'Provider' or 'Deployer' when inferable. "
                "IMPORTANT: intended_use must preserve the FULL purpose AND the "
                "mechanism/technique described (e.g. 'embeds low-frequency tones to "
                "increase impulse buying', not just 'increase impulse buying'). "
                "Do not drop legally-relevant details such as how the system "
                "influences people, what it detects, or who it targets."
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


def _empty_response(
    analysis: str, start: float, used_deep_research: bool, mode: str = "classification"
) -> ChatResponse:
    return ChatResponse(
        analysis=analysis,
        risk_level="unknown",
        obligations=[],
        articles=[],
        sources=[],
        used_deep_research=used_deep_research,
        processing_time=time.time() - start,
        mode=mode,
    )


def _articles_from_text(text: str) -> List[int]:
    """Pull unique Article numbers from free-form text (QA path fallback)."""
    if SCENARIO_AVAILABLE:
        try:
            return extract_articles(text)
        except Exception:
            pass
    nums = {int(m) for m in re.findall(r"Article\s+(\d+)", text or "")}
    return sorted(nums)


def _route_intent(
    message: str, document_text: Optional[str], history: Optional[List["Turn"]] = None
) -> Dict[str, Any]:
    """Decide how to handle the message: a scenario to classify, a general
    question to answer, or a vague scenario that needs a clarifying question.
    Uses the conversation so far so follow-ups resolve against earlier turns."""
    blob = ""
    convo = _history_text(history or [])
    if convo:
        blob += "Conversation so far:\n" + convo + "\n\nLatest message:\n"
    blob += message.strip()
    if document_text and document_text.strip():
        blob += "\n\n[The user also attached a document for context.]"

    system = (
        "You triage messages for an EU AI Act assistant. Use the conversation so "
        "far to resolve references in the latest message. Return ONLY a JSON object "
        'with keys: "intent" ("scenario" or "question"), "sufficient" (boolean), '
        'and "clarifying_question" (string).\n'
        "- intent='scenario': the latest message (together with the conversation) "
        "describes a SPECIFIC AI system, product, or use-case and the user wants to "
        "know whether it is allowed, its risk level, or its obligations \u2014 EVEN "
        "when phrased as a yes/no question like 'would it be okay?', 'is this "
        "allowed?', 'is this compliant?', or 'what is the risk category here?'.\n"
        "- intent='question': an ABSTRACT or definitional question about the AI Act "
        "that does NOT describe a specific system to evaluate (e.g. 'what is a "
        "high-risk system?', 'what are the penalties?', 'does the Act apply to "
        "military uses?', 'what does Article 5 say?').\n"
        "- sufficient: default to TRUE. Set sufficient=false ONLY when intent is "
        "scenario but neither the message NOR the conversation indicates what the "
        "system DOES or its PURPOSE, so a risk level cannot be guessed at all. If a "
        "concrete function/use is present anywhere in the conversation, it IS "
        "sufficient \u2014 classify it even if minor fields are missing. Do NOT ask "
        "for extra detail just to be thorough.\n"
        "- When sufficient=false, put ONE short, specific follow-up in clarifying_question.\n"
        "- For intent='question', always set sufficient=true and clarifying_question=\"\".\n"
        "Examples:\n"
        '"If the provider offers a background music AI that embeds low-frequency tones to increase impulse buying in supermarkets, would it be okay?" -> '
        '{"intent":"scenario","sufficient":true,"clarifying_question":""}\n'
        '"We screen job applicants with an AI CV-ranking tool." -> '
        '{"intent":"scenario","sufficient":true,"clarifying_question":""}\n'
        '(after a scenario was described) "What is the risk category here?" -> '
        '{"intent":"scenario","sufficient":true,"clarifying_question":""}\n'
        '"I built an AI system. What risk level is it?" -> '
        '{"intent":"scenario","sufficient":false,"clarifying_question":"What does the system do, and what is its intended purpose?"}\n'
        '"Does the AI Act apply to military systems?" -> '
        '{"intent":"question","sufficient":true,"clarifying_question":""}\n'
        '"What are the penalties for non-compliance?" -> '
        '{"intent":"question","sufficient":true,"clarifying_question":""}'
    )
    messages = [{"role": "system", "content": system}, {"role": "user", "content": blob}]

    intent, sufficient, clarifying = "scenario", True, ""
    try:
        parsed = json.loads(_strip_fences(_complete(messages)))
        if isinstance(parsed, dict):
            cand = str(parsed.get("intent", "")).strip().lower()
            if cand in ("scenario", "question"):
                intent = cand
            sufficient = bool(parsed.get("sufficient", True))
            cq = parsed.get("clarifying_question")
            if isinstance(cq, str):
                clarifying = cq.strip()
    except Exception:
        pass

    if intent == "question":
        sufficient, clarifying = True, ""
    return {"intent": intent, "sufficient": sufficient, "clarifying_question": clarifying}


def _retrieve_with_upload(query: str, req: "ChatRequest"):
    """Retrieve grounding chunks and append any user-uploaded document as a
    citeable source."""
    docs, metas, scores = pre_retrieve(query, fast=req.fast)
    if req.document_text and req.document_text.strip():
        docs = list(docs) + [req.document_text.strip()]
        metas = list(metas) + [
            {"source": "user-upload", "doc_id": "uploaded-document", "chunk_index": 0}
        ]
        scores = list(scores) + [1.0]
    return docs, metas, scores


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    start = time.time()
    question = req.message.strip()

    route = _route_intent(question, req.document_text, req.history)

    # The scenario is too vague to classify: ask the user for the missing detail
    # instead of guessing a risk level.
    if route["intent"] == "scenario" and not route["sufficient"]:
        clarifying = route["clarifying_question"] or (
            "Could you tell me a bit more about the AI system \u2014 what it does, "
            "who provides or deploys it, and its intended purpose? That lets me "
            "classify its EU AI Act risk level accurately."
        )
        return _empty_response(clarifying, start, False, mode="clarification")

    # General question: answer directly from the corpus with citations, no card.
    if route["intent"] == "question":
        docs, metas, scores = _retrieve_with_upload(question, req)
        if not docs:
            if DEEP_RESEARCH_AVAILABLE and run_deep_research is not None:
                try:
                    research = run_deep_research(question)
                    analysis = research if isinstance(research, str) else str(research)
                except Exception as exc:  # pragma: no cover
                    analysis = f"Deep research fallback failed: {exc}"
                return _empty_response(analysis, start, True, mode="qa")
            return _empty_response(
                "No relevant context was found for this question.", start, False, mode="qa"
            )
        context_block = build_context(docs, metas)
        sources = _sources_from_metas(metas, scores)
        convo = _history_text(req.history)
        qa_question = (
            f"Conversation so far:\n{convo}\n\nLatest question: {question}"
            if convo
            else question
        )
        analysis = _complete(build_messages(qa_question, context_block)).strip()
        return ChatResponse(
            analysis=analysis,
            risk_level="unknown",
            obligations=[],
            articles=_articles_from_text(analysis),
            sources=sources,
            used_deep_research=False,
            processing_time=time.time() - start,
            scenario=None,
            mode="qa",
        )

    # Otherwise: a scenario with enough detail -> full risk classification.
    # Build the scenario (role/intended_use/...) and the retrieval query. When the
    # project's scenario_batch_rag.py is available we use its exact query builder.
    scenario_fields = _extract_scenario_fields(question, req.document_text, req.history)
    convo = _history_text(req.history or [])
    raw_desc = ((convo + "\n") if convo else "") + question.strip()
    if SCENARIO_AVAILABLE:
        query = build_scenario_query(scenario_fields)
        # The classifier sees the raw description too, so terse field extraction
        # can't hide legally-critical mechanism details (e.g. "low-frequency tones").
        classify_text = f"{raw_desc}\n\n{query}".strip()
    else:
        query = question
        classify_text = question

    docs, metas, scores = _retrieve_with_upload(query, req)

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
        # Mirror batch_rag.py's proven technique (83% on the benchmark): a
        # structured-JSON classification prompt, parsed directly, with regex
        # extraction only as a fallback, then scenario calibration.
        raw = _complete(_structured_messages(classify_text, context_block))
        parsed = _parse_structured(raw)
        analysis = (parsed["analysis"] if parsed else "") or raw.strip()
        generated_risk = parsed["risk_level"] if parsed else "unknown"
        obligations = parsed["obligations"] if parsed else []
        articles = parsed["articles"] if parsed else []

        if (not generated_risk or generated_risk == "unknown") and analysis:
            extracted = extract_risk_level(analysis)
            if extracted and extracted != "unknown":
                generated_risk = extracted
        if not obligations and analysis:
            obligations = extract_obligations(analysis)
        if not articles and analysis:
            articles = extract_articles(analysis)

        obligations = _clean_obligations(obligations)
        risk_level = calibrate_risk_level(generated_risk or "unknown", scenario_fields)
        # Never downgrade a model-asserted 'prohibited'. calibrate_risk_level was
        # tuned for the batch benchmark's rich metadata; with the terse fields a
        # chat turn yields, its metadata inference can wrongly override a correct
        # prohibited classification (e.g. subliminal low-frequency nudging) down
        # to limited/minimal. The classifier sees the full description, so trust it.
        if generated_risk == "prohibited":
            risk_level = "prohibited"
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
