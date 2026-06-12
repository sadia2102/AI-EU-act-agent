import type { ChatMessage } from "@/lib/types";
import { formatTime } from "@/lib/format";
import { AttachmentChip } from "./AttachmentChip";
import { SparkIcon } from "./icons";

function renderContent(text: string) {
  // Lightweight **bold** and *italic* rendering for the demo.
  const parts = text.split(/(\*\*[^*]+\*\*|\*[^*]+\*)/g);
  return parts.map((part, i) => {
    if (part.startsWith("**") && part.endsWith("**")) {
      return <strong key={i}>{part.slice(2, -2)}</strong>;
    }
    if (part.startsWith("*") && part.endsWith("*")) {
      return <em key={i}>{part.slice(1, -1)}</em>;
    }
    return <span key={i}>{part}</span>;
  });
}

export function MessageBubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === "user";
  return (
    <div className={`flex gap-3 ${isUser ? "flex-row-reverse" : "flex-row"}`}>
      <div
        className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-full text-xs font-semibold ${
          isUser
            ? "bg-blue-600 text-white"
            : "bg-gradient-to-br from-indigo-500 to-violet-600 text-white"
        }`}
        aria-hidden
      >
        {isUser ? "You" : <SparkIcon className="h-4 w-4" />}
      </div>
      <div className={`flex max-w-[78%] flex-col ${isUser ? "items-end" : "items-start"}`}>
        <div
          className={`rounded-2xl px-4 py-2.5 text-sm leading-relaxed shadow-sm ${
            isUser
              ? "rounded-tr-sm bg-blue-600 text-white"
              : "rounded-tl-sm bg-white text-gray-900 ring-1 ring-black/5 dark:bg-white/[.08] dark:text-gray-100 dark:ring-white/10"
          }`}
        >
          {message.content && (
            <p className="whitespace-pre-wrap">{renderContent(message.content)}</p>
          )}
          {message.attachments && message.attachments.length > 0 && (
            <div className={`flex flex-wrap gap-2 ${message.content ? "mt-2" : ""}`}>
              {message.attachments.map((a) => (
                <AttachmentChip key={a.id} attachment={a} variant="message" />
              ))}
            </div>
          )}
        </div>
        <span className="mt-1 px-1 text-[11px] opacity-50">
          {formatTime(message.createdAt)}
        </span>
      </div>
    </div>
  );
}
