# retrieve_chroma.py
import chromadb
from sentence_transformers import SentenceTransformer
from typing import Optional, Dict, Any

CHROMA_HOST = "localhost"
CHROMA_PORT = 8000
COLLECTION_NAME = "research_chunks"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5

def citation(meta: Dict[str, Any]) -> str:
    """Generate a citation string from metadata."""
    src = meta.get("source", "unknown")
    doc = meta.get("doc_id", "")
    loc = f"chunk_index={meta.get('chunk_index', 'n/a')}"
    parts = [src]
    if doc:
        parts.append(doc)
    parts.append(loc)
    return "[" + "; ".join(parts) + "]"

def retrieve(query: str, where: Optional[Dict[str, Any]] = None, top_k: int = TOP_K):
    """Retrieve documents from ChromaDB with proper metadata handling."""
    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    col = client.get_collection(COLLECTION_NAME)
    model = SentenceTransformer(EMBED_MODEL)
    q_emb = model.encode(query, convert_to_numpy=True).tolist()
    res = col.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
        where=where
    )
    return res

if __name__ == "__main__":
    print("Enter a query (Ctrl+C to exit):")
    try:
        while True:
            q = input("> ").strip()
            if not q:
                continue

            # Only search the new ingested corpus that has provenance
            where_filter = {"source": {"$eq": "thesis_text_chunks_v1"}}  # metadata filter

            # Ask for more hits to avoid seeing just a single result
            res = retrieve(q, where=where_filter, top_k=10)

            # ChromaDB returns results in nested lists format
            # Extract the first (and only) query result
            documents = res.get("documents", [[]])[0] if res.get("documents") else []
            metadatas = res.get("metadatas", [[]])[0] if res.get("metadatas") else []
            distances = res.get("distances", [[]])[0] if res.get("distances") else []

            if not documents:
                print("No results.\n")
                continue

            print("\nTop results:")
            for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances), start=1):
                # Generate citation using the citation function
                cite = citation(meta) if meta else "[unknown source]"
                
                # Format distance
                dist_str = f"{dist:.4f}" if dist is not None else "n/a"
                
                # Print result with metadata
                print(f"#{i} distance={dist_str} {cite}")
                print(f"Document: {doc[:500]}{'...' if len(doc) > 500 else ''}")
                print(f"Full metadata: {meta}")
                print("-" * 80)
                print()
    except KeyboardInterrupt:
        print("\nBye.")
