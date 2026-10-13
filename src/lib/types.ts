export type Role = "user" | "assistant";

export type RiskLevel =
  | "prohibited"
  | "high-risk"
  | "limited-risk"
  | "minimal-risk"
  | "unknown";

export interface Attachment {
  id: string;
  name: string;
  size: number;
  type: string;
}

export interface Source {
  index: number;
  source: string;
  doc_id: string;
  chunk_index?: string | number | null;
  score?: number | null;
}

export interface Scenario {
  role: string;
  intended_use: string;
  system_type: string;
  input_data: string;
  domain: string;
  related_articles: number[];
  obligations: string[];
  risk_level: RiskLevel;
}

export type ReplyMode = "classification" | "qa" | "clarification";

export interface Classification {
  riskLevel: RiskLevel;
  obligations: string[];
  articles: number[];
  sources: Source[];
  usedDeepResearch?: boolean;
  scenario?: Scenario | null;
  mode?: ReplyMode;
}

export interface ChatMessage {
  id: string;
  role: Role;
  content: string;
  attachments?: Attachment[];
  classification?: Classification;
  createdAt: number;
}
