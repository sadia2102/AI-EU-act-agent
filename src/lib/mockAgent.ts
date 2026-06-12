import type { Attachment } from "./types";

const TOPIC_REPLIES: { match: RegExp; reply: string }[] = [
  {
    match: /risk|categor|classif/i,
    reply:
      "The EU AI Act sorts systems into four risk tiers: **unacceptable** (banned), **high-risk** (strict obligations), **limited** (transparency duties), and **minimal** (largely unregulated). Tell me what your system does and I can help place it.",
  },
  {
    match: /high.?risk|annex|obligation/i,
    reply:
      "High-risk systems (Annex III) must meet requirements such as a risk-management system, data governance, technical documentation, logging, human oversight, and a conformity assessment before going to market.",
  },
  {
    match: /gpai|general.?purpose|foundation|llm/i,
    reply:
      "General-purpose AI (GPAI) providers have transparency and documentation duties; models posing *systemic risk* face added evaluation, incident-reporting, and cybersecurity obligations.",
  },
  {
    match: /fine|penalt|enforc/i,
    reply:
      "Penalties scale with severity: up to **€35M or 7%** of global turnover for prohibited-practice breaches, and lower tiers for other violations.",
  },
  {
    match: /timeline|date|when|enter into force|deadline/i,
    reply:
      "Key dates: the Act entered into force in Aug 2024, bans on prohibited systems applied 6 months later, GPAI rules at 12 months, and most high-risk obligations phase in over 24–36 months.",
  },
];

function summarizeAttachments(attachments: Attachment[]): string {
  if (attachments.length === 0) return "";
  const names = attachments.map((a) => `"${a.name}"`).join(", ");
  return `I've received ${attachments.length} document${
    attachments.length > 1 ? "s" : ""
  } (${names}). I'd scan ${
    attachments.length > 1 ? "them" : "it"
  } for AI-system descriptions, intended purpose, and data sources, then map findings to the relevant EU AI Act obligations. `;
}

export function generateMockReply(
  userText: string,
  attachments: Attachment[],
): string {
  const docPart = summarizeAttachments(attachments);
  const topic = TOPIC_REPLIES.find((t) => t.match.test(userText));

  if (topic) return docPart + topic.reply;

  if (docPart && !userText.trim()) {
    return (
      docPart +
      "Ask me anything — e.g. *what risk category does this fall under?* or *what documentation is required?*"
    );
  }

  return (
    docPart +
    "I'm a demo assistant for the EU AI Act. Try asking about risk categories, high-risk obligations, GPAI rules, penalties, or the compliance timeline — or drop in a document and I'll reference it."
  );
}

export function replyDelay(): number {
  return 700 + Math.random() * 700;
}
