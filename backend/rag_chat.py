#!/usr/bin/env python3
"""
RAG CLI chat using ChromaDB + Ollama (Qwen2.5 Instruct).

Prereqs:
  1) Install Ollama: https://ollama.com/download (macOS: `brew install ollama`)
  2) Start server: `ollama serve`
  3) Pull model:   `ollama pull qwen2.5:7b-instruct`

Run:
  python rag_chat.py

Type a question and press Enter. Ctrl+C to exit.
"""

import sys
import json
import textwrap
import requests
import numpy as np
from typing import Dict, Any, List

import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
import os
import time
import argparse
import shelve
from pathlib import Path

# --- Chroma / Embedding Config ---
CHROMA_HOST = "localhost"
CHROMA_PORT = 8000
COLLECTION_NAME = "research_chunks"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 6
WHERE_FILTER = {"source": {"$eq": "thesis_text_chunks_v1"}}

# --- LLM Config ---
# Choose backend: 'openai' or 'ollama'
LLM_BACKEND = os.environ.get('LLM_BACKEND', 'openai')  # 'openai' or 'ollama'

# OpenAI Config (when LLM_BACKEND='openai')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')
OPENAI_MODEL = os.environ.get('OPENAI_MODEL', 'gpt-5.4-mini')  # or other model id
OPENAI_BASE_URL = os.environ.get('OPENAI_BASE_URL', None)  # e.g. https://openrouter.ai/api/v1 or FAU endpoint

# Ollama Config (when LLM_BACKEND='ollama')
OLLAMA_BASE = "http://localhost:11434"
OLLAMA_CHAT_URL = f"{OLLAMA_BASE}/api/chat"
OLLAMA_GENERATE_URL = f"{OLLAMA_BASE}/api/generate"
OLLAMA_MODEL = os.environ.get('OLLAMA_MODEL', 'qwen2.5:7b-instruct-q4_K_M')  # Local model to use with Ollama
MAX_CONTEXT_CHARS = 8000  # truncate combined context if needed
PRE_TOP_K = 8            # per-variant retrieval size
FINAL_TOP_K = 12         # after rerank
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"

# Simple in-memory caches
VARIANT_CACHE: Dict[str, Dict[str, List[str]]] = {}
RETRIEVE_CACHE: Dict[str, Dict[str, Any]] = {}
CACHE_MAX = 128

# Simple disk cache (optional)
CACHE_DIR = Path.home() / ".cache" / "thesis_rag"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_PATH = str(CACHE_DIR / "rag_cache")  # shelve will create files with extensions

def _disk_cache_get(kind: str, key: str):
    try:
        with shelve.open(CACHE_PATH) as db:
            bucket = db.get(kind, {})
            return bucket.get(key)
    except Exception:
        return None

def _disk_cache_set(kind: str, key: str, value):
    try:
        with shelve.open(CACHE_PATH, writeback=True) as db:
            bucket = db.get(kind)
            if bucket is None:
                bucket = {}
            # prune occasionally
            if len(bucket) > CACHE_MAX:
                bucket.clear()
            bucket[key] = value
            db[kind] = bucket
    except Exception:
        pass


def retrieve(query: str, where: Dict[str, Any] | None = None, top_k: int = TOP_K):
    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    col = client.get_collection(COLLECTION_NAME)
    model = SentenceTransformer(EMBED_MODEL)
    q_emb = model.encode(query, convert_to_numpy=True).tolist()
    res = col.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
        where=where,
    )
    # unwrap first query dim
    documents = res.get("documents", [[]])[0] if res.get("documents") else []
    metadatas = res.get("metadatas", [[]])[0] if res.get("metadatas") else []
    distances = res.get("distances", [[]])[0] if res.get("distances") else []
    return documents, metadatas, distances


# -------- Advanced pre-retrieval pipeline -------- #
def _llm_complete(system_prompt: str, user_prompt: str) -> str:
    """Non-streaming completion using configured backend, returning raw text."""
    if LLM_BACKEND == 'openai':
        from openai import OpenAI, APIConnectionError, APITimeoutError, RateLimitError
        api_key = os.environ.get("OPENAI_API_KEY", OPENAI_API_KEY)
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable is not set.")
        client_kwargs = {"api_key": api_key}
        if OPENAI_BASE_URL:
            client_kwargs["base_url"] = OPENAI_BASE_URL
        client = OpenAI(**client_kwargs)
        last_exception = None
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.3,
                    stream=False,
                    timeout=240,
                    max_tokens=8192,
                )
                return resp.choices[0].message.content or ""
            except (APIConnectionError, APITimeoutError, RateLimitError) as e:
                last_exception = e
                if attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
        raise last_exception
    else:
        # Ollama non-stream generate
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": f"<|system|>\n{system_prompt}\n\n<|user|>\n{user_prompt}\n\n<|assistant|>",
            "stream": False,
        }
        r = requests.post(OLLAMA_GENERATE_URL, json=payload, timeout=600)
        r.raise_for_status()
        data = r.json()
        return data.get("response", "")


def _parse_lines_to_list(text: str, max_items: int | None = None) -> List[str]:
    lines = [l.strip(" -•\t") for l in text.splitlines() if l.strip()]
    if max_items is not None:
        lines = lines[:max_items]
    return lines


def generate_variants(query: str, fast: bool = False) -> Dict[str, List[str]]:
    """Create rewrites, sub-questions, and optionally a HyDE draft for the query.
    Uses a small in-memory cache keyed by (fast, normalized_query).
    """
    key = f"{fast}|{query.strip()}"
    if key in VARIANT_CACHE:
        return VARIANT_CACHE[key]
    disk_hit = _disk_cache_get("variants", key)
    if disk_hit is not None:
        VARIANT_CACHE[key] = disk_hit
        return disk_hit

    sys_prompt = "You create search queries for legal research. Return brief, high-signal lines."
    # 3 rewrites
    rewrites_prompt = (
        "Rewrite the following question into 3 alternative phrasings optimized for retrieval. "
        "One per line, no numbering, <= 18 words each.\n\nQuestion:\n" + query
    )
    rewrites_txt = _llm_complete(sys_prompt, rewrites_prompt)
    rewrites = _parse_lines_to_list(rewrites_txt, max_items=(2 if fast else 3))

    # 3–5 sub-questions
    subq_prompt = (
        "Break the question into 3-5 concrete sub-questions that target different aspects. "
        "One per line, no numbering, <= 18 words each.\n\nQuestion:\n" + query
    )
    subs_txt = _llm_complete(sys_prompt, subq_prompt)
    subqs = _parse_lines_to_list(subs_txt, max_items=(3 if fast else 5))

    # HyDE draft
    hyde_list: List[str] = []
    if not fast:
        hyde_sys = "You draft a short hypothetical answer (HyDE) to help semantic retrieval."
        hyde_user = (
            "Write a 4-6 sentence hypothetical answer to the question, neutral tone, no citations.\n\nQuestion:\n"
            + query
        )
        hyde_text = _llm_complete(hyde_sys, hyde_user)
        if hyde_text.strip():
            hyde_list = [hyde_text]

    result = {"rewrites": rewrites, "subqs": subqs, "hyde": hyde_list}
    if len(VARIANT_CACHE) > CACHE_MAX:
        VARIANT_CACHE.clear()
    VARIANT_CACHE[key] = result
    _disk_cache_set("variants", key, result)
    return result


def chroma_query(ids_client, query_text: str, top_k: int) -> Dict[str, Any]:
    # cache per variant text
    cache_key = f"{query_text.strip()}|k={top_k}"
    if cache_key in RETRIEVE_CACHE:
        return RETRIEVE_CACHE[cache_key]
    disk_hit = _disk_cache_get("retrieve", cache_key)
    if disk_hit is not None:
        RETRIEVE_CACHE[cache_key] = disk_hit
        return disk_hit
    col = ids_client.get_collection(COLLECTION_NAME)
    model = SentenceTransformer(EMBED_MODEL)
    q_emb = model.encode(query_text, convert_to_numpy=True).tolist()
    res = col.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
        where=WHERE_FILTER,
    )
    # unwrap
    docs = res.get("documents", [[]])[0] if res.get("documents") else []
    metas = res.get("metadatas", [[]])[0] if res.get("metadatas") else []
    dists = res.get("distances", [[]])[0] if res.get("distances") else []
    ids = res.get("ids", [[]])[0] if res.get("ids") else []
    out = {"ids": ids, "docs": docs, "metas": metas, "dists": dists}
    if len(RETRIEVE_CACHE) > CACHE_MAX:
        RETRIEVE_CACHE.clear()
    RETRIEVE_CACHE[cache_key] = out
    _disk_cache_set("retrieve", cache_key, out)
    return out


def pre_retrieve(query: str, fast: bool = False) -> tuple[List[str], List[Dict[str, Any]], List[float]]:
    """Advanced retrieval: rewrites + (optional) HyDE + sub-questions -> union -> dedupe -> rerank -> top N."""
    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    variants = generate_variants(query, fast=fast)

    # Collect results across all variants
    pool: Dict[str, Dict[str, Any]] = {}
    variant_texts = [query] + variants.get("rewrites", []) + variants.get("subqs", []) + variants.get("hyde", [])
    for qv in variant_texts:
        if not qv or not qv.strip():
            continue
        res = chroma_query(client, qv.strip(), (6 if fast else PRE_TOP_K))
        for _id, doc, meta, dist in zip(res["ids"], res["docs"], res["metas"], res["dists"]):
            if _id not in pool:
                pool[_id] = {"doc": doc, "meta": meta, "dist": dist}
            else:
                # keep closer distance if using distance metric (smaller is better for cosine)
                if dist is not None and pool[_id].get("dist") is not None and dist < pool[_id]["dist"]:
                    pool[_id].update({"doc": doc, "meta": meta, "dist": dist})

    if not pool:
        return [], [], []

    # Rerank with a cross-encoder using the original query against candidate docs
    # Use CPU for cross-encoder to avoid macOS MPS OOM
    ce = CrossEncoder(RERANKER_MODEL, device='cpu')
    cands = list(pool.items())  # [(id, {doc, meta, dist})]
    pairs = [(query, item[1]["doc"]) for item in cands]
    # Predict with small batch size to limit memory usage
    scores_raw = ce.predict(pairs, batch_size=16)
    scores = scores_raw.tolist() if hasattr(scores_raw, 'tolist') else list(scores_raw)

    # Heuristic boost for primary law / official guidance sources
    def _boost_for_meta(meta: Dict[str, Any]) -> float:
        doc_id = (meta or {}).get("doc_id", "")
        text = str(doc_id).lower()
        primary_terms = [
            "directive", "regulation", "article ", "art. ", "court", "cjeu", "judgment", "case", "tfeu", "charter"
        ]
        official_terms = [
            "commission", "ai office", "edpb", "official", "guidelines", "notice", "opinion"
        ]
        boost = 0.0
        if any(t in text for t in primary_terms):
            boost += 0.15
        if any(t in text for t in official_terms):
            boost += 0.10
        # domain-specific terms
        if "ai act" in text or "cdsm" in text:
            boost += 0.05
        return boost

    scores = [s + _boost_for_meta(cands[i][1]["meta"]) for i, s in enumerate(scores)]

    # Sort by score desc
    ranked_all = sorted(zip(cands, scores), key=lambda x: x[1], reverse=True)

    # Doc-id level dedupe while preserving order by score to create candidate lists
    seen_doc_ids = set()
    cand_docs: List[str] = []
    cand_metas: List[Dict[str, Any]] = []
    cand_scores: List[float] = []
    for (cand, sc) in ranked_all:
        meta = cand[1]["meta"]
        doc_id = (meta or {}).get("doc_id")
        key_id = doc_id or cand[0]
        if key_id in seen_doc_ids:
            continue
        seen_doc_ids.add(key_id)
        cand_docs.append(cand[1]["doc"])
        cand_metas.append(meta)
        cand_scores.append(sc)

    # --- MMR selection to promote diversity among top results ---
    def _normalize(mat: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
        return mat / n

    limit = 10 if fast else FINAL_TOP_K
    if len(cand_docs) <= limit:
        sel_indices = list(range(len(cand_docs)))
    else:
        # Embed candidate docs
        emb_model = SentenceTransformer(EMBED_MODEL)
        doc_embs = emb_model.encode(cand_docs, convert_to_numpy=True)
        doc_embs = _normalize(np.array(doc_embs))
        # MMR greedy selection
        lambda_mult = 0.5
        sel_indices: List[int] = []
        remaining = set(range(len(cand_docs)))
        # start with highest score
        first = int(np.argmax(cand_scores))
        sel_indices.append(first)
        remaining.remove(first)
        while len(sel_indices) < limit and remaining:
            max_idx = None
            max_crit = -1e9
            for i in list(remaining):
                sim_to_sel = 0.0
                for j in sel_indices:
                    sim = float(np.dot(doc_embs[i], doc_embs[j]))  # cosine on normalized vectors
                    if sim > sim_to_sel:
                        sim_to_sel = sim
                crit = lambda_mult * float(cand_scores[i]) - (1 - lambda_mult) * sim_to_sel
                if crit > max_crit:
                    max_crit = crit
                    max_idx = i
            sel_indices.append(max_idx)
            remaining.remove(max_idx)

    docs = [cand_docs[i] for i in sel_indices]
    metas = [cand_metas[i] for i in sel_indices]
    rerank_scores = [cand_scores[i] for i in sel_indices]

    # --- Neighbor window expansion: try to add one adjacent chunk per selected doc ---
    try:
        col = client.get_collection(COLLECTION_NAME)
        def _fetch_neighbor(m: Dict[str, Any], offset: int) -> tuple[str, Dict[str, Any]] | None:
            try:
                did = m.get("doc_id")
                cidx = m.get("chunk_index")
                if did is None or cidx is None:
                    return None
                target_idx = cidx + offset
                res = col.get(
                    where={"doc_id": {"$eq": did}, "chunk_index": {"$eq": target_idx}},
                    include=["documents", "metadatas"],
                    limit=1,
                )
                docs_g = res.get("documents") or []
                metas_g = res.get("metadatas") or []
                if docs_g and metas_g and docs_g[0] and metas_g[0]:
                    return docs_g[0], metas_g[0]
            except Exception:
                return None
            return None

        seen_pair = {(m.get("doc_id"), m.get("chunk_index")) for m in metas}
        aug_docs: List[str] = []
        aug_metas: List[Dict[str, Any]] = []
        for m in metas:
            # Try +1 then -1
            for off in (1, -1):
                maybe = _fetch_neighbor(m, off)
                if maybe is None:
                    continue
                ndoc, nmeta = maybe
                key = (nmeta.get("doc_id"), nmeta.get("chunk_index"))
                if key in seen_pair:
                    continue
                seen_pair.add(key)
                aug_docs.append(ndoc)
                aug_metas.append(nmeta)
                break  # only one neighbor per seed
        # Append neighbors to the end
        docs.extend(aug_docs)
        metas.extend(aug_metas)
    except Exception:
        # neighbor expansion is best-effort
        pass

    return docs, metas, rerank_scores


def build_context(docs: List[str], metas: List[Dict[str, Any]]) -> str:
    """Build a numbered context block with explicit Source headers and citations."""
    items: List[str] = []
    for i, (doc, meta) in enumerate(zip(docs, metas), start=1):
        src = meta.get("source", "unknown")
        doc_id = meta.get("doc_id", "")
        chunk_idx = meta.get("chunk_index", "n/a")
        cite = f"[{src}; {doc_id}; chunk_index={chunk_idx}]"
        snippet = doc.strip().replace("\n", " ")
        items.append(f"Source ({i}): {cite}\n{snippet}")
    context = "\n\n".join(items)
    if len(context) > MAX_CONTEXT_CHARS:
        context = context[:MAX_CONTEXT_CHARS] + "\n...[truncated]"
    return context


def build_messages(question: str, context_block: str) -> List[Dict[str, str]]:
    system = (
        "You are a precise legal research assistant. Answer ONLY the specific question asked. "
        "Use ONLY the provided context. Keep answers concise and directly address the question. "
        "EVERY factual statement MUST end with a citation exactly like [source; doc_id; chunk_index=K]. "
        "Do not add extra information not directly relevant to the question. "
        "Aim for 1-3 sentences maximum unless the question specifically requires more detail."
    )
    user = (
        f"Question: {question}\n\n"
        f"Context: {context_block}\n\n"
        "Provide a concise answer that directly addresses the question. "
        "Include citations for all factual statements. "
        "Do not add extra sections or broader context unless specifically asked."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
 
def chat_openai(messages: List[Dict[str, str]]):
    """Stream response from OpenAI Chat Completions API."""
    from openai import OpenAI
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable is not set.")
        raise RuntimeError("Missing OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            stream=True,
            temperature=0.1,
            max_tokens=300,
        )
        answer_parts: List[str] = []
        print("\nAnswer (OpenAI):\n")
        for chunk in response:
            if getattr(chunk, "choices", None) and chunk.choices[0].delta and chunk.choices[0].delta.content:
                delta = chunk.choices[0].delta.content
                answer_parts.append(delta)
                sys.stdout.write(delta)
                sys.stdout.flush()
        print("\n")
        return "".join(answer_parts)
    except Exception as e:
        print(f"\nError calling OpenAI: {str(e)}")
        raise

def chat_ollama(messages: List[Dict[str, str]]):
    """Stream response from Ollama. Try /api/chat, fall back to /api/generate for older versions."""
    try:
        with requests.post(
            OLLAMA_CHAT_URL,
            json={
                "model": OLLAMA_MODEL,
                "messages": messages,
                "stream": True,
            },
            stream=True,
            timeout=600,
        ) as resp:
            if resp.status_code == 404:
                # Fall back to /api/generate using a single prompt
                raise requests.HTTPError("404 on /api/chat", response=resp)
            resp.raise_for_status()
            return _stream_chat(resp)
    except requests.HTTPError as e:
        # Fallback path
        if getattr(e, "response", None) is not None and e.response.status_code == 404:
            # Convert messages to a single prompt
            sys_prompt = next((m["content"] for m in messages if m.get("role") == "system"), "")
            user_prompt = next((m["content"] for m in messages if m.get("role") == "user"), "")
            prompt = f"<|system|>\n{sys_prompt}\n\n\n{user_prompt}\n\n<|assistant|>"
            with requests.post(
                OLLAMA_GENERATE_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": True,
                },
                stream=True,
                timeout=600,
            ) as resp2:
                resp2.raise_for_status()
                return _stream_generate(resp2)
        else:
            raise


def parse_args():
    parser = argparse.ArgumentParser(description="RAG CLI (Chroma + LLM)")
    parser.add_argument("--fast", action="store_true", help="Use faster pre-retrieval (fewer variants, no HyDE, smaller k)")
    return parser.parse_args()


essay = """\
Welcome to RAG chat! Type your question. We'll retrieve context from ChromaDB and ask an LLM to answer with citations.
Backend: {} | Mode: {{mode}}
""".format(OPENAI_MODEL if LLM_BACKEND == 'openai' else f'Ollama ({OLLAMA_MODEL})')


def main():
    args = parse_args()
    print(essay.replace("{mode}", "FAST" if args.fast else "STANDARD"))
    try:
        while True:
            q = input("Question > ").strip()
            if not q:
                continue
            print("\nRetrieving context (variants{} + rerank)...".format(" (FAST)" if args.fast else " + HyDE"))
            docs, metas, dists = pre_retrieve(q, fast=args.fast)
            if not docs:
                print("No context found. Try another question.")
                continue
            context_block = build_context(docs, metas)

            print("\nAnswer:\n")
            messages = build_messages(q, context_block)
            if LLM_BACKEND == 'openai':
                chat_openai(messages)
            else:  # ollama
                chat_ollama(messages)

            # Show sources at the end for convenience
            print("\nSources:")
            for i, meta in enumerate(metas, start=1):
                src = meta.get("source", "unknown")
                doc_id = meta.get("doc_id", "")
                chunk_idx = meta.get("chunk_index", "n/a")
                print(f"- ({i}) [{src}; {doc_id}; chunk_index={chunk_idx}]")
            print("\n" + "-"*80 + "\n")
    except KeyboardInterrupt:
        print("\nBye.")


if __name__ == "__main__":
    main()
