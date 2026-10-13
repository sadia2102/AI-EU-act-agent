"""One-off ingest: load text_chunks.tsv into the research_chunks collection,
matching the metadata schema the pipeline expects
(source=thesis_text_chunks_v1, doc_id, chunk_index, length)."""

import chromadb
from sentence_transformers import SentenceTransformer

TSV = "text_chunks.tsv"
COLLECTION = "research_chunks"
SOURCE = "thesis_text_chunks_v1"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

client = chromadb.HttpClient(host="localhost", port=8000)
try:
    client.delete_collection(COLLECTION)
except Exception:
    pass
col = client.create_collection(COLLECTION)

model = SentenceTransformer(EMBED_MODEL)

docs, metas, ids = [], [], []
per_doc_idx = {}
with open(TSV, encoding="utf-8") as f:
    for line in f:
        line = line.rstrip("\n")
        if not line.strip():
            continue
        parts = line.split("\t", 1)
        if len(parts) != 2:
            continue
        doc_id, text = parts[0].strip(), parts[1]
        idx = per_doc_idx.get(doc_id, 0)
        per_doc_idx[doc_id] = idx + 1
        docs.append(text)
        metas.append({
            "source": SOURCE,
            "doc_id": doc_id,
            "chunk_index": idx,
            "length": len(text),
        })
        ids.append(f"{doc_id}__{idx}")

print(f"Read {len(docs)} chunks across {len(per_doc_idx)} docs. Embedding...")
embeddings = model.encode(docs, convert_to_numpy=True, batch_size=64, show_progress_bar=True).tolist()

B = 256
for i in range(0, len(docs), B):
    col.add(
        ids=ids[i:i + B],
        documents=docs[i:i + B],
        embeddings=embeddings[i:i + B],
        metadatas=metas[i:i + B],
    )
    print(f"  added {min(i + B, len(docs))}/{len(docs)}")

print("Collection count:", col.count())
