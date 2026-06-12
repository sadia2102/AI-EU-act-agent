# EU AI Act Assistant

A chatbot-style frontend where the assistant can also accept documents. Built
with **Next.js (App Router) + TypeScript + Tailwind CSS**.

> This is a **frontend design / demo**: assistant replies are simulated locally
> (`src/lib/mockAgent.ts`) and uploaded files are kept in browser memory only —
> nothing is sent to a server.

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
  lib/
    types.ts          # ChatMessage / Attachment types
    mockAgent.ts      # simulated responses
    format.ts         # byte/time formatting helpers
```

## Wiring up a real backend

Replace the simulated reply in `ChatApp.send()` with a call to your API and
upload `Attachment` files to your storage/LLM of choice. The UI is structured so
only that one function needs to change.

## Scripts

- `npm run dev` – start the dev server
- `npm run build` – production build
- `npm run lint` – ESLint
