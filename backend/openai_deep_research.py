#!/usr/bin/env python3
"""
OpenAI Deep Research (Iterative) using Chroma + your existing retrieval.

This script orchestrates an iterative research loop:
- Planner: proposes focused follow-up queries/sub-questions
- Retriever: uses rag_chat.pre_retrieve() to gather context for each follow-up
- Synthesizer: updates a research log and a current answer, enforcing per-sentence citations

Requirements:
  pip install openai>=1.0.0
  (and the deps used by rag_chat.py: sentence-transformers, chromadb, requests)

Usage examples:
  python openai_deep_research.py --rounds 2 "What transparency obligations does the AI Act impose on GPAI providers?"
  python openai_deep_research.py --rounds 1 --fast "What is lawful access under EU TDM?"

Environment:
  export OPENAI_API_KEY="sk-..."

Notes:
- This script reuses retrieval utilities from rag_chat.py (same folder): pre_retrieve, build_context
- Model defaults to a reasoning-capable model; adjust via --model if needed
"""

import os
import sys
import re
import argparse
from typing import List, Dict, Any

import openai
from openai import OpenAI

# Reuse your retrieval pipeline
from rag_chat import pre_retrieve, build_context


DEFAULT_MODEL = os.environ.get( "o4-mini")  # reasoning-capable; set via env or flag
DEFAULT_MAX_CONTEXT_CHARS = int(os.environ.get("DEEP_MAX_CONTEXT", "2800"))
DEFAULT_FOLLOWUPS = int(os.environ.get("DEEP_FOLLOWUPS", "3"))
DEFAULT_PER_SUBQ_DOCS = int(os.environ.get("DEEP_PER_SUBQ_DOCS", "3"))


def get_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set.")
    org = os.environ.get("OPENAI_ORG_ID")
    # Pass organization if provided to ensure we hit the org that has access
    return OpenAI(api_key=api_key, organization=org) if org else OpenAI(api_key=api_key)


def _get_deep_tools() -> list:
    """Return required 'tools' for Deep Research models on /v1/responses.
    Config via env:
      - OPENAI_DEEP_TOOLS: 'web' (default), 'file', or 'both'
      - OPENAI_VECTOR_STORE_ID: required when using file search
    """
    mode = (os.environ.get("OPENAI_DEEP_TOOLS") or "web").strip().lower()
    tools: list = []
    if mode in ("web", "both"):
        tools.append({"type": "web_search_preview"})
    if mode in ("file", "both"):
        vs_id = os.environ.get("OPENAI_VECTOR_STORE_ID")
        # file_search tool can optionally include vector_store_ids
        if vs_id:
            tools.append({"type": "file_search", "vector_store_ids": [vs_id]})
        else:
            tools.append({"type": "file_search"})
    # Fallback: ensure at least one tool
    if not tools:
        tools = [{"type": "web_search_preview"}]
    return tools


def _safe_chat_complete(client: OpenAI, preferred_model: str, messages: List[Dict[str, str]], temperature: float = 0.2, max_tokens: int | None = 400) -> Dict[str, Any]:
    """Try the preferred model; on model_not_found fallback to o4-mini then gpt-5.4-mini."""
    fallback_order = [preferred_model, "o4-mini", "gpt-5.4-mini"]
    last_err: Exception | None = None
    for idx, m in enumerate(fallback_order):
        try:
            resp = client.chat.completions.create(
                model=m,
                messages=messages,
                temperature=temperature,
                stream=False,
                max_tokens=max_tokens,
            )
            return {"resp": resp, "model_used": m}
        except Exception as e:
            # Only fall back on model_not_found / 404
            msg = str(e).lower()
            # Some models (deep-research) only support /v1/responses. Try Responses API.
            if "only supported in v1/responses" in msg or "responses" in msg and "not in v1/chat/completions" in msg:
                try:
                    # Build a single string input to Responses API and include tools=[] per requirements
                    sys_part = next((mobj["content"] for mobj in messages if mobj.get("role") == "system"), "")
                    user_part = next((mobj["content"] for mobj in messages if mobj.get("role") == "user"), "")
                    joined = f"<system>\n{sys_part}\n</system>\n<user>\n{user_part}\n</user>"
                    # Choose required tools for deep-research models
                    tools = _get_deep_tools()
                    r = client.responses.create(
                        model=m,
                        input=joined,
                        tools=tools,
                    )
                    # Prefer output_text if available
                    out_text = getattr(r, "output_text", None)
                    if not out_text:
                        out_text = str(r)
                    return {"resp": r, "model_used": m, "text": out_text, "via": "responses"}
                except Exception as e_resp:
                    last_err = e_resp
                    print(f"[model fallback] {m} requires /v1/responses but call failed: {e_resp}")
                    continue
            # Some models do not accept 'temperature' at all; retry once without it
            if (isinstance(e, openai.BadRequestError) or "unsupported value" in msg) and "temperature" in msg:
                try:
                    resp = client.chat.completions.create(
                        model=m,
                        messages=messages,
                        stream=False,
                        max_tokens=max_tokens,
                    )
                    return {"resp": resp, "model_used": m}
                except Exception as e2:
                    msg2 = str(e2).lower()
                    if isinstance(e2, openai.NotFoundError) or "model_not_found" in msg2 or "404" in msg2:
                        last_err = e2
                        # Verbose note about fallback
                        print(f"[model fallback] {m} unavailable: {e2}")
                        continue
                    raise
            if isinstance(e, openai.NotFoundError) or "model_not_found" in msg or "404" in msg:
                last_err = e
                # Verbose note about fallback
                print(f"[model fallback] {m} unavailable: {e}")
                continue
            raise
    # If all failed, raise the last error
    raise last_err if last_err else RuntimeError("Failed to call OpenAI chat completions.")


def plan_followups(client: OpenAI, model: str, question: str, previous_log: str, limit: int) -> List[str]:
    """Ask the model to propose 3-5 focused follow-up queries. Returns a list of strings."""
    system = (
        "You plan concise follow-up queries for legal research. Keep high signal, no redundancy."
    )
    user = (
        f"Q:\n{question}\n\n"
        f"Log (may be empty):\n{previous_log}\n\n"
        "Output 2-3 lines, <=14 words each. No numbering."
    )
    result = _safe_chat_complete(
        client,
        model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.2,
    )
    resp = result["resp"]
    model_used = result["model_used"]
    print(f"Planner model: {model_used}")
    text = result.get("text") or (getattr(resp, "choices", None) and resp.choices[0].message.content) or ""
    lines = [l.strip(" -•\t") for l in text.splitlines() if l.strip()]
    return lines[:max(1, limit)]


def synthesize_findings(client: OpenAI, model: str, question: str, contexts: str, previous_log: str, verbose: bool = False) -> Dict[str, str]:
    """Ask the model to update the Research Log and produce a Current Answer from provided contexts."""
    if verbose:
        system = (
            "You are a rigorous legal research assistant. Use ONLY the provided context. "
            "Every factual/legal sentence MUST end with [source; doc_id; chunk_index=K]."
        )
        user = (
            f"Q:\n{question}\n\n"
            f"Context:\n{contexts}\n\n"
            f"Prev Log:\n{previous_log}\n\n"
            "Write two labeled sections.\n"
            "Research Log Update: 3–5 headings; 1–3 bullets each; all sentences cited.\n"
            "Current Answer: 2–4 headings; 1–2 bullets each; then 2–3 sentence Summary; all sentences cited."
        )
    else:
        system = (
            "Answer ONLY from the context. Cite every sentence like [source; doc_id; chunk_index=K]."
        )
        user = (
            f"Q:\n{question}\n\n"
            f"Context:\n{contexts}\n\n"
            "Output:\n"
            "- Research Log Update: 3 headings, 1–2 bullets each, all cited.\n"
            "- Current Answer: 2 headings, 1–2 bullets each, then 2-sentence Summary, all cited."
        )
    result = _safe_chat_complete(
        client,
        model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.2,
    )
    resp = result["resp"]
    model_used = result["model_used"]
    print(f"Synthesizer model: {model_used}")
    text = result.get("text") or (getattr(resp, "choices", None) and resp.choices[0].message.content) or ""
    return {"text": text}


def _needs_citation_fix(text: str) -> bool:
    """Heuristic: ensure each bullet or sentence line ends with a citation token [source; doc_id; chunk_index=K]."""
    cite_re = re.compile(r"\[.+?;\s*.+?;\s*chunk_index=.+?\]\s*$")
    lines = [l.rstrip() for l in text.splitlines()]
    violations = 0
    for ln in lines:
        if not ln.strip():
            continue
        # Only enforce on bullet lines or short sentences (ignoring headings)
        if ln.lstrip().startswith(('-', '*', '1)', '2)', '3)', '4)', '5)', '•')) or ln.strip().endswith('.'):
            if not cite_re.search(ln):
                violations += 1
                if violations > 2:
                    return True
    return violations > 0


def repair_citations(client: OpenAI, model: str, original_text: str, contexts: str) -> str:
    """Ask the model to repair formatting so that every sentence ends with a proper citation; do not add new claims."""
    system = (
        "You are a strict copy editor. Repair the formatting and citations ONLY. "
        "Ensure every factual/legal sentence ends with a citation exactly like [source; doc_id; chunk_index=K]. "
        "Do NOT invent content; do NOT change meaning; use ONLY citations present in the provided context. "
        "Keep the same headings and bullets; just add or fix citations."
    )
    user = (
        "Context to use for citations (do not quote beyond this):\n" + contexts + "\n\n" +
        "Original text (repair only; keep structure):\n" + original_text
    )
    result = _safe_chat_complete(
        client,
        model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.2,
    )
    # Use text from Responses API if present, else fall back to chat choices
    return result.get("text") or (getattr(result["resp"], "choices", None) and result["resp"].choices[0].message.content) or original_text


def run_deep_research(question: str, rounds: int = 2, fast: bool = False, model: str = DEFAULT_MODEL, out_path: str | None = None, max_sources: int = 20,
                      followups: int = DEFAULT_FOLLOWUPS, per_subq_docs: int = DEFAULT_PER_SUBQ_DOCS, max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
                      verbose: bool = False) -> Dict[str, Any]:
    client = get_client()
    research_log = ""
    current_answer = ""
    # Collect (meta, score) so we can rank sources at the end
    all_meta_scores: List[tuple[Dict[str, Any], float]] = []
    rounds_completed = 0

    for r in range(1, rounds + 1):
        rounds_completed = r
        print(f"\n=== Round {r}/{rounds}: Planning follow-ups ===")
        followups_list = plan_followups(client, model, question, research_log, limit=followups)
        if not followups_list:
            print("No follow-ups proposed; falling back to main question retrieval.")
            followups_list = [question]
        for i, q in enumerate(followups_list, start=1):
            print(f"  - {q}")

        print(f"\n=== Round {r}/{rounds}: Retrieving contexts ===")
        context_sections: List[str] = []
        for i, subq in enumerate(followups_list, start=1):
            docs, metas, scores = pre_retrieve(subq, fast=fast)
            if not docs:
                continue
            # take only top per_subq_docs to reduce context size
            docs = docs[:per_subq_docs]
            metas = metas[:per_subq_docs]
            scores = (scores or [])[:per_subq_docs]
            ctx = build_context(docs, metas)
            context_sections.append(f"Follow-up ({i}): {subq}\n{ctx}")
            # track meta+score
            for m, sc in zip(metas, (scores or [0.0]*len(metas))):
                try:
                    scf = float(sc)
                except Exception:
                    scf = 0.0
                all_meta_scores.append((m, scf))
        if not context_sections:
            print("No context retrieved; stopping.")
            break
        stitched = "\n\n".join(context_sections)
        # Cap stitched context tokens by characters
        if len(stitched) > max_context_chars:
            stitched = stitched[:max_context_chars] + "\n...[truncated]"

        print(f"\n=== Round {r}/{rounds}: Synthesizing findings ===")
        result = synthesize_findings(client, model, question, stitched, research_log, verbose=verbose)
        block = result["text"].strip()
        # Validate and repair citations if needed
        if _needs_citation_fix(block):
            block = repair_citations(client, model, block, stitched).strip()
        print(block)

        # Update the log and a simple current answer snapshot
        research_log = (research_log + "\n\n" + block).strip()
        current_answer = block

    print("\n=== Final Output ===\n")
    print(current_answer)

    # Dedup by (source, doc_id, chunk_index) keeping best score, and sort
    best_map: Dict[tuple, tuple[Dict[str, Any], float]] = {}
    for m, sc in all_meta_scores:
        key = (m.get("source"), m.get("doc_id"), m.get("chunk_index"))
        prev = best_map.get(key)
        if prev is None or sc > prev[1]:
            best_map[key] = (m, sc)
    ranked = sorted(best_map.values(), key=lambda x: x[1], reverse=True)

    top_ranked = ranked[:max_sources]
    sources_list: List[Dict[str, Any]] = []
    if top_ranked:
        print("\nSources:")
        for i, (meta, sc) in enumerate(top_ranked, start=1):
            src = meta.get("source", "unknown")
            doc_id = meta.get("doc_id", "")
            cidx = meta.get("chunk_index", "n/a")
            print(f"- ({i}) [{src}; {doc_id}; chunk_index={cidx}]")
            sources_list.append({
                "index": i,
                "source": src,
                "doc_id": doc_id,
                "chunk_index": cidx,
                "score": sc,
                "metadata": meta,
            })
    if out_path:
        try:
            header = not os.path.exists(out_path)
            mode = 'a' if not header else 'w'
            with open(out_path, mode, encoding='utf-8') as f:
                if header:
                    f.write("# Research Log and Answer\n\n")
                # Per-question section
                f.write(f"\n---\n\n## Question\n\n{question}\n\n")
                f.write("### Final Current Answer\n\n")
                f.write(current_answer + "\n\n")
                f.write("### Sources\n\n")
                for entry in sources_list:
                    f.write(f"- ({entry['index']}) [{entry['source']}; {entry['doc_id']}; chunk_index={entry['chunk_index']}]\n")
                f.write("\n### Full Research Log\n\n")
                f.write(research_log + "\n")
            print(f"\nSaved report to: {out_path}")
        except Exception as e:
            print(f"\nWarning: could not write report to {out_path}: {e}")
    return {
        "question": question,
        "current_answer": current_answer,
        "research_log": research_log,
        "sources": sources_list,
        "rounds_completed": rounds_completed,
        "fast_mode": fast,
        "model": model,
        "max_sources": max_sources,
        "followups": followups,
        "per_subq_docs": per_subq_docs,
        "max_context_chars": max_context_chars,
        "verbose": verbose,
    }


def parse_args():
    p = argparse.ArgumentParser(description="OpenAI Deep Research (Iterative) using Chroma retrieval")
    p.add_argument("question", type=str, nargs='?', default=None, help="Research question (omit to enter interactively)")
    p.add_argument("--rounds", type=int, default=2, help="Number of research iterations")
    p.add_argument("--fast", action="store_true", help="Use faster retrieval (fewer variants, no HyDE)")
    p.add_argument("--model", type=str, default=DEFAULT_MODEL, help="OpenAI model to use (default from env OPENAI_DEEP_MODEL or o4-mini)")
    p.add_argument("--out", type=str, default=None, help="Write final answer and research log to this Markdown file")
    p.add_argument("--max-sources", type=int, default=20, help="Maximum number of sources to list at the end")
    p.add_argument("--followups", type=int, default=DEFAULT_FOLLOWUPS, help="Max number of planner follow-ups per round (default 3)")
    p.add_argument("--per-subq-docs", type=int, default=DEFAULT_PER_SUBQ_DOCS, help="Docs per follow-up to include in context (default 3)")
    p.add_argument("--max-context-chars", type=int, default=DEFAULT_MAX_CONTEXT_CHARS, help="Cap stitched context length in characters (default 2800)")
    p.add_argument("--verbose", action="store_true", help="Use more detailed prompts/output (higher cost)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        # Continuous loop: keep taking questions until quit
        while True:
            question = args.question
            if not question:
                question = input("Question > ").strip()
            if not question or question.lower() in {"q", "quit", "exit"}:
                break
            run_deep_research(question, rounds=args.rounds, fast=args.fast, model=args.model, out_path=args.out, max_sources=args.max_sources,
                              followups=args.followups, per_subq_docs=args.per_subq_docs, max_context_chars=args.max_context_chars,
                              verbose=args.verbose)
            # After first run, force interactive mode for subsequent questions
            args.question = None
            print("\n(type 'quit' to exit)\n")
    except KeyboardInterrupt:
        print("\nBye.")
