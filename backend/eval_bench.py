#!/usr/bin/env python3
"""Evaluate classification accuracy on scenarios.json.

Modes:
  --method batch   : run the proven batch_rag.process_single_question (baseline)
  --method server  : run the server's scenario classification core

Usage: python eval_bench.py scenarios.json --method batch [--limit N]
"""
import argparse
import json
import time
from collections import Counter


def _install_model_cache():
    """Cache CrossEncoder/SentenceTransformer so rag_chat doesn't reload them
    on every retrieval call (massive speedup for batch evaluation)."""
    import rag_chat
    from functools import lru_cache
    from sentence_transformers import SentenceTransformer, CrossEncoder

    @lru_cache(maxsize=None)
    def _st(name):
        return SentenceTransformer(name)

    @lru_cache(maxsize=None)
    def _ce(name, device="cpu"):
        return CrossEncoder(name, device=device)

    rag_chat.SentenceTransformer = lambda name, *a, **k: _st(name)
    rag_chat.CrossEncoder = lambda name, *a, **k: _ce(name, k.get("device", "cpu"))


def balanced_sample(items, per_class):
    by_class = {}
    for it in items:
        by_class.setdefault(canon(it.get("risk_level")), []).append(it)
    out = []
    for c in ["prohibited", "high", "limited", "minimal"]:
        out.extend(by_class.get(c, [])[:per_class])
    return out


def canon(risk: str) -> str:
    r = (risk or "unknown").strip().lower()
    r = r.replace("-risk", "")
    if r in ("minimal", "minimal-risk"):
        return "minimal"
    if r in ("limited", "limited-risk"):
        return "limited"
    if r in ("high", "high-risk"):
        return "high"
    return r  # prohibited / unknown


def run_batch(items, limit):
    from batch_rag import process_single_question, _coerce_question_entry
    preds = []
    for i, entry in enumerate(items[:limit], 1):
        q = _coerce_question_entry(entry, i)
        res = process_single_question(q, fast=False)
        preds.append(res.get("generated_risk_level", "unknown"))
        print(f"[batch {i}/{min(limit,len(items))}] exp={entry.get('risk_level')} got={preds[-1]}")
    return preds


def run_server(items, limit):
    from rag_chat import pre_retrieve, build_context
    from scenario_batch_rag import (
        build_scenario_query, extract_risk_level, extract_obligations,
        extract_articles, calibrate_risk_level,
    )
    from server import _structured_messages, _parse_structured, _complete, _clean_obligations
    preds = []
    for i, entry in enumerate(items[:limit], 1):
        query = build_scenario_query(entry)
        docs, metas, scores = pre_retrieve(query, fast=False)
        if not docs:
            preds.append("unknown")
            print(f"[server {i}] no docs")
            continue
        context_block = build_context(docs, metas)
        raw = _complete(_structured_messages(query, context_block))
        parsed = _parse_structured(raw)
        analysis = (parsed["analysis"] if parsed else "") or raw.strip()
        risk = parsed["risk_level"] if parsed else "unknown"
        if (not risk or risk == "unknown") and analysis:
            ex = extract_risk_level(analysis)
            if ex and ex != "unknown":
                risk = ex
        generated = risk
        risk = calibrate_risk_level(risk or "unknown", entry)
        if generated == "prohibited":  # mirror server: never downgrade a model prohibited
            risk = "prohibited"
        preds.append(risk)
        print(f"[server {i}/{min(limit,len(items))}] exp={entry.get('risk_level')} got={risk}")
    return preds


def run_old(items, limit):
    """Previous live technique: free-form analysis + regex extraction."""
    from rag_chat import pre_retrieve, build_context
    from scenario_batch_rag import (
        build_scenario_query, build_scenario_messages, extract_risk_level,
        calibrate_risk_level,
    )
    from server import _complete
    preds = []
    for i, entry in enumerate(items[:limit], 1):
        query = build_scenario_query(entry)
        docs, metas, scores = pre_retrieve(query, fast=False)
        if not docs:
            preds.append("unknown")
            print(f"[old {i}] no docs")
            continue
        context_block = build_context(docs, metas)
        analysis = _complete(build_scenario_messages(query, context_block)).strip()
        risk = calibrate_risk_level(extract_risk_level(analysis), entry)
        preds.append(risk)
        print(f"[old {i}/{min(limit,len(items))}] exp={entry.get('risk_level')} got={risk}")
    return preds


def _v2_messages(question, context_block):
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
        '  "analysis": string,  // 3-5 sentences, justify the tier, cite like [source; doc_id; chunk_index=K]\n'
        "  \"risk_level\": one of ['prohibited','high-risk','limited-risk','minimal-risk','unknown'],\n"
        '  "obligations": array of 3-6 short actionable obligation bullets (each ending with a citation),\n'
        '  "articles": array of unique Article numbers as integers\n'
        "}"
    )
    user = (
        f"Scenario: {question}\n\n"
        f"Context:\n{context_block}\n\n"
        "Apply the ordered tier test above and output ONLY valid JSON."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def run_v2(items, limit):
    """Improved classification prompt (tier definitions + ordered test)."""
    from rag_chat import pre_retrieve, build_context
    from scenario_batch_rag import (
        build_scenario_query, extract_risk_level, calibrate_risk_level,
    )
    from server import _parse_structured, _complete
    preds = []
    for i, entry in enumerate(items[:limit], 1):
        query = build_scenario_query(entry)
        docs, metas, scores = pre_retrieve(query, fast=False)
        if not docs:
            preds.append("unknown")
            print(f"[v2 {i}] no docs")
            continue
        context_block = build_context(docs, metas)
        raw = _complete(_v2_messages(query, context_block))
        parsed = _parse_structured(raw)
        analysis = (parsed["analysis"] if parsed else "") or raw.strip()
        risk = parsed["risk_level"] if parsed else "unknown"
        if (not risk or risk == "unknown") and analysis:
            ex = extract_risk_level(analysis)
            if ex and ex != "unknown":
                risk = ex
        risk = calibrate_risk_level(risk or "unknown", entry)
        preds.append(risk)
        print(f"[v2 {i}/{min(limit,len(items))}] exp={entry.get('risk_level')} got={risk}")
    return preds


def run_live(items, limit):
    """Full live path: scenario -> natural sentence -> LLM field extraction ->
    rebuild query -> structured classify -> calibrate. Simulates a user typing
    the scenario into the chat (tests the extraction round-trip)."""
    from rag_chat import pre_retrieve, build_context
    from scenario_batch_rag import (
        build_scenario_query, extract_risk_level, calibrate_risk_level,
    )
    from server import (
        _structured_messages, _parse_structured, _complete, _extract_scenario_fields,
    )
    preds = []
    for i, entry in enumerate(items[:limit], 1):
        user_text = build_scenario_query(entry)  # what a user would type
        fields = _extract_scenario_fields(user_text, None, None)
        query = build_scenario_query(fields)
        docs, metas, scores = pre_retrieve(query, fast=False)
        if not docs:
            preds.append("unknown")
            print(f"[live {i}] no docs")
            continue
        context_block = build_context(docs, metas)
        raw = _complete(_structured_messages(query, context_block))
        parsed = _parse_structured(raw)
        analysis = (parsed["analysis"] if parsed else "") or raw.strip()
        risk = parsed["risk_level"] if parsed else "unknown"
        if (not risk or risk == "unknown") and analysis:
            ex = extract_risk_level(analysis)
            if ex and ex != "unknown":
                risk = ex
        risk = calibrate_risk_level(risk or "unknown", fields)
        preds.append(risk)
        print(f"[live {i}/{min(limit,len(items))}] exp={entry.get('risk_level')} got={risk}")
    return preds


def score(items, preds):
    n = len(preds)
    correct = sum(1 for e, p in zip(items, preds) if canon(e.get("risk_level")) == canon(p))
    print(f"\nAccuracy: {correct}/{n} = {correct/n*100:.2f}%")
    # per-class
    classes = ["prohibited", "high", "limited", "minimal"]
    for c in classes:
        tp = sum(1 for e, p in zip(items, preds) if canon(e.get("risk_level")) == c and canon(p) == c)
        fp = sum(1 for e, p in zip(items, preds) if canon(e.get("risk_level")) != c and canon(p) == c)
        fn = sum(1 for e, p in zip(items, preds) if canon(e.get("risk_level")) == c and canon(p) != c)
        prec = tp / (tp + fp) if tp + fp else 0
        rec = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
        print(f"  {c:11s} P={prec:.3f} R={rec:.3f} F1={f1:.3f} (tp={tp} fp={fp} fn={fn})")
    print("pred dist:", Counter(canon(p) for p in preds))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bench")
    ap.add_argument("--method", choices=["batch", "server", "old", "live", "v2"], required=True)
    ap.add_argument("--limit", type=int, default=10**9)
    ap.add_argument("--per-class", type=int, default=0,
                    help="if >0, take this many scenarios from each risk class")
    args = ap.parse_args()
    _install_model_cache()
    data = json.load(open(args.bench))
    items = data.get("data") or data.get("questions")
    if args.per_class > 0:
        items = balanced_sample(items, args.per_class)
    limit = min(args.limit, len(items))
    t = time.time()
    runner = {"batch": run_batch, "server": run_server, "old": run_old,
              "live": run_live, "v2": run_v2}[args.method]
    preds = runner(items, limit)
    score(items[:limit], preds)
    print(f"elapsed: {time.time()-t:.0f}s")


if __name__ == "__main__":
    main()
