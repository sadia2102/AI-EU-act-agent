# inspect_chroma.py
import chromadb

CHROMA_HOST = "localhost"
CHROMA_PORT = 8000
COLLECTION_NAME = "research_chunks"

client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
col = client.get_collection(COLLECTION_NAME)

print("Count:", col.count())

# Request allowed include keys (omit 'ids' here)
out = col.get(limit=5, include=["metadatas", "documents"])  # allowed include keys

ids = out.get("ids", [])               # ids are always present in the response object
metas = out.get("metadatas", [])
docs = out.get("documents", [])

for i, (id_, meta, doc) in enumerate(zip(ids, metas, docs), start=1):
    print(f"\n#{i} id={id_}")
    print("metadata:", meta)
    print("doc[:120]:", (doc or "")[:120])
