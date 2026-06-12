import type {
  Classification,
  ReplyMode,
  RiskLevel,
  Scenario,
  Source,
} from "./types";

const API_URL = process.env.NEXT_PUBLIC_API_URL;

export function isApiConfigured(): boolean {
  return Boolean(API_URL);
}

interface ChatApiResponse {
  analysis: string;
  risk_level: RiskLevel;
  obligations: string[];
  articles: number[];
  sources: Source[];
  used_deep_research?: boolean;
  scenario?: Scenario | null;
  mode?: ReplyMode;
}

export interface AgentReply {
  content: string;
  classification?: Classification;
}

export async function fetchAgentReply(
  message: string,
  documentText?: string,
): Promise<AgentReply> {
  if (!API_URL) {
    throw new Error("NEXT_PUBLIC_API_URL is not set");
  }

  const res = await fetch(`${API_URL}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      message,
      document_text: documentText ?? null,
    }),
  });

  if (!res.ok) {
    throw new Error(`API responded with ${res.status}`);
  }

  const data: ChatApiResponse = await res.json();
  return {
    content: data.analysis,
    classification: {
      riskLevel: data.risk_level,
      obligations: data.obligations ?? [],
      articles: data.articles ?? [],
      sources: data.sources ?? [],
      usedDeepResearch: data.used_deep_research,
      scenario: data.scenario ?? null,
      mode: data.mode ?? "classification",
    },
  };
}
