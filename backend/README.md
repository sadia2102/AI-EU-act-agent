# Backend API (EU AI Act Assistant)

A thin [FastAPI](https://fastapi.tiangolo.com/) wrapper around the existing RAG
pipeline so the Next.js frontend can call it over HTTP. It reuses your own
modules — it does **not** reimplement retrieval or the LLM call.

## 1. Put your pipeline files here

Copy your existing scripts into this `backend/` folder next to `server.py`:

```
backend/
  server.py                 # provided (the API)
  requirements.txt          # provided
  rag_chat.py               # YOURS  (pre_retrieve, build_context, LLM config)
  scenario_batch_rag.py     # YOURS  (risk classification + obligations + articles)
  openai_deep_research.py   # YOURS  (run_deep_research — optional fallback)
  batch_rag.py              # YOURS  (optional)
```

> **`scenario_batch_rag.py` is what produces the full scenario output**
> (role / intended_use / system_type / input_data / domain / related_articles /
> obligations / risk_level), reusing its `build_scenario_query`,
> `build_scenario_messages`, `extract_*`, and `calibrate_risk_level`. If it's not
> present, the API degrades to a simpler strict-JSON classification.

> **Security:** remove the hardcoded `OPENAI_API_KEY = "sk-..."` line from
> `rag_chat.py`. The code already falls back to `os.environ["OPENAI_API_KEY"]`.
> Rotate that key — it was committed in plaintext.

## 2. Install & configure

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

export OPENAI_API_KEY="sk-..."           # required (OpenAI backend)
# export CORS_ORIGINS="http://localhost:3000"   # optional, this is the default
```

Make sure ChromaDB is running and populated (the collection `research_chunks`
on `localhost:8000`, as configured in `rag_chat.py`). Point `--path` at the
folder that holds your existing data:

```bash
chroma run --path "/path/to/your/chroma_db" --host localhost --port 8000
```

Verify the collection actually has your chunks (must print a non-zero number):

```bash
python -c "import chromadb; c=chromadb.HttpClient(host='localhost',port=8000); print(c.get_collection('research_chunks').count())"
```

### Rebuild the index from `text_chunks.tsv`

If the count is `0` or the collection is missing, repopulate it from your TSV
(tab-separated `doc_id<TAB>text` lines). Copy `text_chunks.tsv` into `backend/`
and run, while `chroma run` is up:

```bash
python ingest_tsv.py
```

This creates the `research_chunks` collection with the metadata the pipeline
expects (`source=thesis_text_chunks_v1`, `doc_id`, `chunk_index`, `length`) and
embeds with `all-MiniLM-L6-v2`.

## 3. Run the API

`rag_chat.py` points Chroma at port 8000, so run the API on a **different port**
(e.g. 8008):

```bash
uvicorn server:app --host 0.0.0.0 --port 8008 --reload
```

Quick check:

```bash
curl localhost:8008/health
curl -X POST localhost:8008/chat -H 'Content-Type: application/json' \
  -d '{"message":"What transparency obligations apply to GPAI providers?"}'
```

## 4. Point the frontend at it

In the repo root, create `.env.local`:

```
NEXT_PUBLIC_API_URL=http://localhost:8008
```

Then `npm run dev`. When `NEXT_PUBLIC_API_URL` is set the UI calls this API; if
it is unset or unreachable, the UI falls back to its built-in mock so the demo
still works offline.

## API

`POST /chat`

```jsonc
// request
{ "message": "string", "document_text": "optional string", "fast": false }

// response
{
  "analysis": "string with [source; doc_id; chunk_index=K] citations",
  "risk_level": "prohibited | high-risk | limited-risk | minimal-risk | unknown",
  "obligations": ["...", "..."],
  "articles": [5, 10, 27],
  "sources": [{ "index": 1, "source": "...", "doc_id": "...", "chunk_index": 3, "score": 0.7 }],
  "used_deep_research": false,
  "processing_time": 2.5,
  // full scenario schema (the UI's "Copy JSON" button copies this)
  "scenario": {
    "role": "Provider",
    "intended_use": "...",
    "system_type": "...",
    "input_data": "...",
    "domain": "...",
    "related_articles": [5, 10, 27],
    "obligations": ["..."],
    "risk_level": "prohibited"
  }
}
```

The server first asks the LLM to extract the scenario fields from the user's
message (and any attached document), builds the retrieval query with
`build_scenario_query`, runs retrieval + analysis, then extracts obligations /
articles and calibrates the risk level — mirroring `scenario_batch_rag.py`.

Uploaded documents: the frontend sends the plain text of attached files as
`document_text`; the server adds it as an extra citeable source
(`user-upload`) so the model can classify the user's own scenario.
