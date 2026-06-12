"use client";

import { useRef, useState, type KeyboardEvent } from "react";
import type { Attachment } from "@/lib/types";
import { AttachmentChip } from "./AttachmentChip";
import { PaperclipIcon, SendIcon } from "./icons";

const ACCEPTED =
  ".pdf,.doc,.docx,.txt,.md,.csv,.rtf,.json,application/pdf,text/plain";

interface PendingFile extends Attachment {
  file: File;
}

interface ComposerProps {
  disabled?: boolean;
  onSend: (
    text: string,
    attachments: Attachment[],
    documentText?: string,
  ) => void;
}

const TEXT_READABLE = /^(text\/|application\/(json|xml|csv))/;
const TEXT_EXTENSIONS = /\.(txt|md|markdown|csv|json|xml|rtf|log)$/i;
const MAX_DOC_CHARS = 20000;

async function readDocumentText(files: PendingFile[]): Promise<string> {
  const readable = files.filter(
    (p) => TEXT_READABLE.test(p.type) || TEXT_EXTENSIONS.test(p.name),
  );
  if (readable.length === 0) return "";
  const parts = await Promise.all(
    readable.map(async (p) => {
      const content = (await p.file.text()).slice(0, MAX_DOC_CHARS);
      return `--- ${p.name} ---\n${content}`;
    }),
  );
  return parts.join("\n\n");
}

let idCounter = 0;
function nextId() {
  idCounter += 1;
  return `att-${Date.now()}-${idCounter}`;
}

export function Composer({ disabled, onSend }: ComposerProps) {
  const [text, setText] = useState("");
  const [pending, setPending] = useState<PendingFile[]>([]);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  function addFiles(files: FileList | null) {
    if (!files) return;
    const next: PendingFile[] = Array.from(files).map((file) => ({
      id: nextId(),
      name: file.name,
      size: file.size,
      type: file.type || "application/octet-stream",
      file,
    }));
    setPending((prev) => [...prev, ...next]);
  }

  function removeAttachment(id: string) {
    setPending((prev) => prev.filter((p) => p.id !== id));
  }

  function resetTextarea() {
    if (textareaRef.current) textareaRef.current.style.height = "auto";
  }

  async function handleSend() {
    const trimmed = text.trim();
    if (disabled || (!trimmed && pending.length === 0)) return;
    const attachments: Attachment[] = pending.map((p) => ({
      id: p.id,
      name: p.name,
      size: p.size,
      type: p.type,
    }));
    const documentText = await readDocumentText(pending);
    onSend(trimmed, attachments, documentText || undefined);
    setText("");
    setPending([]);
    resetTextarea();
  }

  function handleKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  const canSend = !disabled && (text.trim().length > 0 || pending.length > 0);

  return (
    <div className="border-t border-black/10 bg-white/70 px-4 py-3 backdrop-blur dark:border-white/10 dark:bg-black/30">
      <div className="mx-auto max-w-3xl">
        <div
          onDragOver={(e) => {
            e.preventDefault();
            setIsDragging(true);
          }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setIsDragging(false);
            addFiles(e.dataTransfer.files);
          }}
          className={`rounded-2xl border bg-white shadow-sm transition dark:bg-white/[.06] ${
            isDragging
              ? "border-blue-500 ring-2 ring-blue-500/30"
              : "border-black/10 dark:border-white/15"
          }`}
        >
          {pending.length > 0 && (
            <div className="flex flex-wrap gap-2 border-b border-black/5 px-3 pb-2.5 pt-3 dark:border-white/10">
              {pending.map((p) => (
                <AttachmentChip
                  key={p.id}
                  attachment={p}
                  onRemove={removeAttachment}
                />
              ))}
            </div>
          )}

          <div className="flex items-end gap-2 p-2">
            <button
              type="button"
              aria-label="Attach documents"
              title="Attach documents"
              disabled={disabled}
              onClick={() => fileInputRef.current?.click()}
              className="rounded-xl p-2 text-gray-500 transition hover:bg-black/5 hover:text-gray-700 disabled:opacity-40 dark:text-gray-400 dark:hover:bg-white/10 dark:hover:text-gray-200"
            >
              <PaperclipIcon className="h-5 w-5" />
            </button>
            <input
              ref={fileInputRef}
              type="file"
              multiple
              accept={ACCEPTED}
              className="hidden"
              onChange={(e) => {
                addFiles(e.target.files);
                e.target.value = "";
              }}
            />

            <textarea
              ref={textareaRef}
              rows={1}
              value={text}
              disabled={disabled}
              placeholder={
                isDragging ? "Drop documents here…" : "Ask about the EU AI Act, or attach a document…"
              }
              onChange={(e) => {
                setText(e.target.value);
                e.target.style.height = "auto";
                e.target.style.height = `${Math.min(e.target.scrollHeight, 160)}px`;
              }}
              onKeyDown={handleKeyDown}
              className="max-h-40 flex-1 resize-none bg-transparent py-2 text-sm outline-none placeholder:text-gray-400 disabled:opacity-50"
            />

            <button
              type="button"
              aria-label="Send message"
              title="Send message"
              disabled={!canSend}
              onClick={handleSend}
              className="rounded-xl bg-blue-600 p-2 text-white transition hover:bg-blue-700 disabled:cursor-not-allowed disabled:bg-gray-300 dark:disabled:bg-white/15"
            >
              <SendIcon className="h-5 w-5" />
            </button>
          </div>
        </div>
        <p className="mt-1.5 px-1 text-center text-[11px] text-gray-400">
          Press Enter to send, Shift+Enter for a new line. Demo UI — responses are
          simulated.
        </p>
      </div>
    </div>
  );
}
