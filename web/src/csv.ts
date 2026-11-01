import type { FeedbackLabel } from "./api/client";

const COLUMNS: (keyof FeedbackLabel)[] = [
  "incident_id",
  "title",
  "predicted_service",
  "actual_service",
  "outcome",
  "confidence",
  "decision",
  "labelled_by",
  "created_at",
  "is_demo",
  "note",
];

function cell(value: unknown): string {
  const text = value === null || value === undefined ? "" : String(value);
  // A leading formula character would run in a spreadsheet, so it is neutralised.
  const safe = /^[=+\-@\t\r]/.test(text) && typeof value === "string" ? `'${text}` : text;
  return /[",\n\r]/.test(safe) ? `"${safe.replaceAll('"', '""')}"` : safe;
}

export function labelsToCsv(items: FeedbackLabel[]): string {
  const rows = items.map((item) => COLUMNS.map((column) => cell(item[column])).join(","));
  return `${[COLUMNS.join(","), ...rows].join("\r\n")}\r\n`;
}

export function downloadCsv(text: string, name: string) {
  const url = URL.createObjectURL(new Blob([text], { type: "text/csv;charset=utf-8" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  link.click();
  URL.revokeObjectURL(url);
}
