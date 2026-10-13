# extractor.py
# extractor.py
from typing import List, Dict, Any

def _as_meta_dict(meta: Any) -> Dict[str, Any]:
    if meta is None:
        return {}
    if isinstance(meta, dict):
        return meta
    if isinstance(meta, list):
        if len(meta) == 1 and isinstance(meta, dict):
            return meta
        return {"meta_list": meta}
    return {"meta": str(meta)}

def extract_evidence(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    evidence = []
    for h in hits:
        text = h.get("text", "")
        meta = _as_meta_dict(h.get("metadata"))
        quote = text[:400] if isinstance(text, str) else str(text)[:400]
        evidence.append({
            "claim_or_note": "Relevant excerpt",
            "quote": quote,
            "source": meta.get("source", "unknown"),
            "locator": f"chunk_index={meta.get('chunk_index', 'n/a')}",
            "distance": h.get("distance"),
            "length": meta.get("length")
        })
    return evidence

def deduplicate_evidence(evidence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    uniq = []
    for e in evidence:
        key = (e.get("source"), e.get("locator"), (e.get("quote") or "")[:120])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(e)
    return uniq
