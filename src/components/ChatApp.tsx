"use client";

import { useEffect, useRef, useState } from "react";
import type { Attachment, ChatMessage, Classification } from "@/lib/types";
import { generateMockReply, replyDelay } from "@/lib/mockAgent";
import { fetchAgentReply, isApiConfigured } from "@/lib/api";
import { Composer } from "./Composer";
import { MessageBubble } from "./MessageBubble";
import { TypingIndicator } from "./TypingIndicator";
import { SparkIcon } from "./icons";

const SUGGESTIONS = [
  "What are the risk categories?",
  "What obligations apply to high-risk systems?",
  "What are the rules for general-purpose AI?",
  "What are the penalties for non-compliance?",
];

let msgCounter = 0;
function newId(role: string) {
  msgCounter += 1;
  return `${role}-${Date.now()}-${msgCounter}`;
}

function createUserMessage(
  text: string,
  attachments: Attachment[],
): ChatMessage {
  return {
    id: newId("user"),
    role: "user",
    content: text,
    attachments,
    createdAt: Date.now(),
  };
}

function createAssistantMessage(
  content: string,
  classification?: Classification,
): ChatMessage {
  return {
    id: newId("assistant"),
    role: "assistant",
    content,
    classification,
    createdAt: Date.now(),
  };
}

export function ChatApp() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isTyping, setIsTyping] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({
      top: scrollRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [messages, isTyping]);

  async function send(
    text: string,
    attachments: Attachment[],
    documentText?: string,
  ) {
    const history = messages.map((m) => ({
      role: m.role,
      content: m.content,
    }));
    setMessages((prev) => [...prev, createUserMessage(text, attachments)]);
    setIsTyping(true);

    if (isApiConfigured()) {
      try {
        const reply = await fetchAgentReply(text, documentText, history);
        setMessages((prev) => [
          ...prev,
          createAssistantMessage(reply.content, reply.classification),
        ]);
        setIsTyping(false);
        return;
      } catch {
        // Backend unreachable: fall through to the built-in mock so the demo works.
      }
    }

    await new Promise((resolve) => window.setTimeout(resolve, replyDelay()));
    setMessages((prev) => [
      ...prev,
      createAssistantMessage(generateMockReply(text, attachments)),
    ]);
    setIsTyping(false);
  }

  const isEmpty = messages.length === 0;

  return (
    <div className="flex h-full flex-col bg-gradient-to-b from-gray-50 to-gray-100 dark:from-[#0c0c10] dark:to-[#08080b]">
      <header className="flex items-center gap-3 border-b border-black/10 bg-white/70 px-4 py-3 backdrop-blur dark:border-white/10 dark:bg-black/30">
        <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-indigo-500 to-violet-600 text-white">
          <SparkIcon className="h-5 w-5" />
        </div>
        <div className="leading-tight">
          <h1 className="text-sm font-semibold">EU AI Act Assistant</h1>
          <p className="text-xs text-gray-500 dark:text-gray-400">
            Ask questions or upload documents for review
          </p>
        </div>
        <span className="ml-auto flex items-center gap-1.5 rounded-full bg-green-500/10 px-2.5 py-1 text-xs font-medium text-green-600 dark:text-green-400">
          <span className="h-1.5 w-1.5 rounded-full bg-green-500" />
          Online
        </span>
      </header>

      <div ref={scrollRef} className="flex-1 overflow-y-auto px-4 py-6">
        <div className="mx-auto max-w-3xl">
          {isEmpty ? (
            <div className="flex flex-col items-center justify-center py-10 text-center">
              <div className="mb-4 flex h-14 w-14 items-center justify-center rounded-2xl bg-gradient-to-br from-indigo-500 to-violet-600 text-white">
                <SparkIcon className="h-7 w-7" />
              </div>
              <h2 className="text-lg font-semibold">
                How can I help with the EU AI Act?
              </h2>
              <p className="mt-1 max-w-md text-sm text-gray-500 dark:text-gray-400">
                Ask a question, or attach a document (PDF, DOCX, TXT…) and I&apos;ll
                reference it in my answer.
              </p>
              <div className="mt-6 grid w-full max-w-xl grid-cols-1 gap-2 sm:grid-cols-2">
                {SUGGESTIONS.map((s) => (
                  <button
                    key={s}
                    type="button"
                    onClick={() => send(s, [])}
                    className="rounded-xl border border-black/10 bg-white px-4 py-3 text-left text-sm transition hover:border-blue-400 hover:bg-blue-50 dark:border-white/10 dark:bg-white/[.04] dark:hover:border-blue-500/60 dark:hover:bg-white/[.08]"
                  >
                    {s}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="flex flex-col gap-5">
              {messages.map((m) => (
                <MessageBubble key={m.id} message={m} />
              ))}
              {isTyping && <TypingIndicator />}
            </div>
          )}
        </div>
      </div>

      <Composer disabled={isTyping} onSend={send} />
    </div>
  );
}
