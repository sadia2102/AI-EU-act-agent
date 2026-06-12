import type { Classification, RiskLevel } from "@/lib/types";

const RISK_STYLES: Record<RiskLevel, { label: string; className: string }> = {
  prohibited: {
    label: "Prohibited",
    className: "bg-red-500/15 text-red-700 dark:text-red-300 ring-red-500/30",
  },
  "high-risk": {
    label: "High risk",
    className:
      "bg-orange-500/15 text-orange-700 dark:text-orange-300 ring-orange-500/30",
  },
  "limited-risk": {
    label: "Limited risk",
    className:
      "bg-amber-500/15 text-amber-700 dark:text-amber-300 ring-amber-500/30",
  },
  "minimal-risk": {
    label: "Minimal risk",
    className:
      "bg-green-500/15 text-green-700 dark:text-green-300 ring-green-500/30",
  },
  unknown: {
    label: "Unknown",
    className: "bg-gray-500/15 text-gray-600 dark:text-gray-300 ring-gray-500/30",
  },
};

export function RiskBadge({ level }: { level: RiskLevel }) {
  const style = RISK_STYLES[level] ?? RISK_STYLES.unknown;
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-semibold ring-1 ${style.className}`}
    >
      <span className="h-1.5 w-1.5 rounded-full bg-current" />
      {style.label}
    </span>
  );
}

export function ClassificationCard({
  classification,
}: {
  classification: Classification;
}) {
  const { riskLevel, obligations, articles, sources, usedDeepResearch } =
    classification;

  const hasDetails =
    obligations.length > 0 || articles.length > 0 || sources.length > 0;
  if (riskLevel === "unknown" && !hasDetails) return null;

  return (
    <div className="mt-3 flex flex-col gap-3 border-t border-black/5 pt-3 dark:border-white/10">
      <div className="flex flex-wrap items-center gap-2">
        <RiskBadge level={riskLevel} />
        {usedDeepResearch && (
          <span className="rounded-full bg-violet-500/15 px-2 py-0.5 text-[11px] font-medium text-violet-600 ring-1 ring-violet-500/30 dark:text-violet-300">
            deep research fallback
          </span>
        )}
      </div>

      {obligations.length > 0 && (
        <div>
          <p className="mb-1 text-xs font-semibold uppercase tracking-wide opacity-60">
            Obligations
          </p>
          <ul className="list-disc space-y-1 pl-4 text-sm">
            {obligations.map((o, i) => (
              <li key={i}>{o}</li>
            ))}
          </ul>
        </div>
      )}

      {articles.length > 0 && (
        <div className="flex flex-wrap items-center gap-1.5">
          <span className="text-xs font-semibold uppercase tracking-wide opacity-60">
            Articles
          </span>
          {articles.map((a) => (
            <span
              key={a}
              className="rounded-md bg-black/5 px-1.5 py-0.5 text-xs font-medium dark:bg-white/10"
            >
              Art. {a}
            </span>
          ))}
        </div>
      )}

      {sources.length > 0 && (
        <div>
          <p className="mb-1 text-xs font-semibold uppercase tracking-wide opacity-60">
            Citations
          </p>
          <ul className="space-y-0.5 text-xs opacity-80">
            {sources.map((s) => (
              <li key={s.index} className="font-mono">
                ({s.index}) [{s.source}; {s.doc_id}; chunk_index=
                {String(s.chunk_index ?? "n/a")}]
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
