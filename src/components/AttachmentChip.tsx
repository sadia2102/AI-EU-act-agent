import type { Attachment } from "@/lib/types";
import { formatBytes } from "@/lib/format";
import { CloseIcon, FileIcon } from "./icons";

interface AttachmentChipProps {
  attachment: Attachment;
  onRemove?: (id: string) => void;
  variant?: "composer" | "message";
}

export function AttachmentChip({
  attachment,
  onRemove,
  variant = "composer",
}: AttachmentChipProps) {
  const isComposer = variant === "composer";
  return (
    <div
      className={`flex items-center gap-2 rounded-lg border px-2.5 py-1.5 text-sm ${
        isComposer
          ? "border-black/10 bg-black/[.03] dark:border-white/15 dark:bg-white/[.06]"
          : "border-white/20 bg-white/10"
      }`}
    >
      <FileIcon className="h-4 w-4 shrink-0 opacity-70" />
      <div className="min-w-0">
        <p className="max-w-[10rem] truncate font-medium leading-tight">
          {attachment.name}
        </p>
        <p className="text-xs opacity-60">{formatBytes(attachment.size)}</p>
      </div>
      {onRemove && (
        <button
          type="button"
          aria-label={`Remove ${attachment.name}`}
          onClick={() => onRemove(attachment.id)}
          className="ml-1 rounded p-0.5 opacity-60 transition hover:bg-black/10 hover:opacity-100 dark:hover:bg-white/10"
        >
          <CloseIcon className="h-3.5 w-3.5" />
        </button>
      )}
    </div>
  );
}
