# EU AI Act Assistant

A chatbot-style frontend where the assistant can also accept documents. Built
with **Next.js (App Router) + TypeScript + Tailwind CSS**.

It can run in two modes:

- **Demo mode (default):** assistant replies are simulated locally
  (`src/lib/mockAgent.ts`); uploaded files stay in browser memory.
- **Connected mode:** set `NEXT_PUBLIC_API_URL` and the UI calls the RAG backend
  in [`backend/`](./backend), which classifies the EU AI Act **risk level** and
  returns **obligations**, **articles**, and **citations**. If the backend is
  unreachable it falls back to demo mode automatically.

## Features

- Chat interface with user/assistant message bubbles, timestamps, and a typing
  indicator.
- Document upload via the paperclip button **or** drag-and-drop onto the
  composer (PDF, DOCX, TXT, MD, CSV, JSON…), shown as removable attachment chips.
- Attachments are rendered inside the sent message; the mock agent acknowledges
  and "references" them.
- Suggested prompts on the empty state, auto-growing input, Enter-to-send
  (Shift+Enter for newline), and responsive light/dark styling.

## Getting Started

```bash
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

## Project structure

```
src/
  app/
    layout.tsx        # root layout + metadata
    page.tsx          # renders <ChatApp />
  components/
    ChatApp.tsx       # state, message flow, empty state
    Composer.tsx      # text input + file upload + drag & drop
    MessageBubble.tsx # a single message + attachments
    AttachmentChip.tsx
    TypingIndicator.tsx
    icons.tsx
    ClassificationCard.tsx # risk badge + obligations + articles + citations
  lib/
    types.ts          # ChatMessage / Attachment / Classification types
    mockAgent.ts      # simulated responses (demo mode)
    api.ts            # calls the backend /chat endpoint (connected mode)
    format.ts         # byte/time formatting helpers
backend/
  server.py           # FastAPI wrapper around the existing RAG pipeline
  requirements.txt
  README.md           # how to run the backend
```

## Connecting the backend

The chat UI talks to a FastAPI `POST /chat` endpoint that reuses the project's
existing RAG pipeline (`rag_chat.py` + Chroma + OpenAI/Ollama). See
[`backend/README.md`](./backend/README.md) for setup. In short:

```bash
cp .env.local.example .env.local   # NEXT_PUBLIC_API_URL=http://localhost:8008
```

With `NEXT_PUBLIC_API_URL` set, `ChatApp.send()` calls the API and renders the
risk classification, obligations, articles, and citations. Attached text
documents are sent as `document_text` so the agent can classify the user's own
scenario.

## Scripts

- `npm run dev` – start the dev server
- `npm run build` – production build
- `npm run lint` – ESLint

## Thesis evaluation

The `thesis/` directory contains the final multi-model comparison report and LaTeX/Word tables. The evaluation scripts, benchmark data, and model outputs are in `backend/`. See [`thesis/README.md`](./thesis/README.md) for reproduction steps.
