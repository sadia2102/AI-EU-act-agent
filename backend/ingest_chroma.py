# ingest_chroma.py
import chromadb
from sentence_transformers import SentenceTransformer
import hashlib
from typing import List, Dict, Tuple

CHROMA_HOST = "localhost"
CHROMA_PORT = 8000
COLLECTION_NAME = "research_chunks"
CHUNKS_FILE = "text_chunks.tsv"   # TSV with: doc_id \t text
BATCH_SIZE = 128
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SOURCE_TAG = "thesis_text_chunks_v1"

def load_tsv(path: str) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            doc_id, text = parts
            if text.strip():
                pairs.append((doc_id.strip(), text.strip()))
    return pairs

def stable_id(doc_id: str, text: str) -> str:
    h = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return f"{doc_id}_{h}"

def batched(iterable, n):
    for i in range(0, len(iterable), n):
        yield iterable[i:i+n]

def main():
    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    col = client.get_or_create_collection(COLLECTION_NAME)
    pairs = load_tsv(CHUNKS_FILE)
    print(f"Loaded {len(pairs)} chunks from {CHUNKS_FILE}")
    if not pairs:
        print("No chunks to ingest. Exiting.")
        return

    model = SentenceTransformer(EMBED_MODEL)

    processed = 0
    for batch_num, batch in enumerate(batched(pairs, BATCH_SIZE)):
        docs: List[str] = []
        metas: List[Dict] = []
        ids: List[str] = []
        for doc_id, text in batch:
            docs.append(text)
            metadata = {
                "source": SOURCE_TAG,
                "doc_id": doc_id,
                "chunk_index": processed,
                "length": len(text),
            }
            metas.append(metadata)
            ids.append(stable_id(doc_id, text))
            processed += 1
        
        # Generate embeddings
        embs = model.encode(docs, convert_to_numpy=True).tolist()
        
        # Upsert to ChromaDB
        col.upsert(ids=ids, documents=docs, embeddings=embs, metadatas=metas)
        print(f"Ingested batch {batch_num + 1}: {processed}/{len(pairs)} chunks")
        
        # Show sample metadata for first batch
        if batch_num == 0:
            print(f"Sample metadata from first batch: {metas[0]}")

    print("Ingestion complete and persisted on server volume.")
    
    # Verify ingestion by checking a few documents
    print("\nVerifying ingestion...")
    try:
        result = col.get(limit=3, include=["documents", "metadatas"])
        docs = result.get("documents", [])
        metas = result.get("metadatas", [])
        print(f"Verification: Found {len(docs)} documents in collection")
        if metas:
            print(f"Sample metadata: {metas[0]}")
    except Exception as e:
        print(f"Verification failed: {e}")

if __name__ == "__main__":
    main()
