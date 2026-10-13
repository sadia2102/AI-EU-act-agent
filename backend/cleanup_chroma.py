#!/usr/bin/env python3
"""
Cleanup script for ChromaDB collection `research_chunks`.

Deletes legacy/dirty records:
- Records with metadata doc_id == "unknown_doc"
- Records missing the "length" field in metadata

Only affects records with source == "thesis_text_chunks_v1" to avoid touching unrelated data.

Usage:
  python cleanup_chroma.py
"""

import chromadb
from typing import List, Dict, Any

CHROMA_HOST = "localhost"
CHROMA_PORT = 8000
COLLECTION_NAME = "research_chunks"
SOURCE_TAG = "thesis_text_chunks_v1"
BATCH_SIZE = 500

def batched(iterable, n):
    for i in range(0, len(iterable), n):
        yield iterable[i:i+n]

def should_delete(meta: Dict[str, Any]) -> bool:
    # Condition 1: explicitly bad doc_id
    if meta.get("doc_id") == "unknown_doc":
        return True
    # Condition 2: missing length field
    if "length" not in meta:
        return True
    return False

def main():
    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    col = client.get_collection(COLLECTION_NAME)

    print(f"Scanning collection '{COLLECTION_NAME}' for dirty records (source='{SOURCE_TAG}')...")

    to_delete_ids: List[str] = []
    total_scanned = 0

    # We will page through the collection using .get with an increasing offset until no more are returned
    offset = 0
    page_size = 1000

    while True:
        result = col.get(
            where={"source": {"$eq": SOURCE_TAG}},
            include=["metadatas"],
            limit=page_size,
            offset=offset,
        )
        ids = result.get("ids") or []
        metas = result.get("metadatas") or []
        if not ids:
            break

        total_scanned += len(ids)

        for _id, meta in zip(ids, metas):
            if isinstance(meta, list):
                # Some clients may return nested; normalize to dict if needed
                meta = meta[0] if meta and isinstance(meta[0], dict) else {}
            if not isinstance(meta, dict):
                meta = {}
            if should_delete(meta):
                to_delete_ids.append(_id)

        print(f"Scanned {total_scanned} records... marked {len(to_delete_ids)} for deletion so far")
        offset += page_size

    if not to_delete_ids:
        print("No records matched cleanup criteria. Nothing to delete.")
        return

    print(f"\nDeleting {len(to_delete_ids)} records in batches of {BATCH_SIZE}...")
    deleted = 0
    for batch in batched(to_delete_ids, BATCH_SIZE):
        col.delete(ids=batch)
        deleted += len(batch)
        print(f"Deleted {deleted}/{len(to_delete_ids)}")

    print("\nCleanup complete.")

if __name__ == "__main__":
    main()
