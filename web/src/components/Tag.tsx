import styles from "./primitives.module.css";

type Tone = "neutral" | "signal" | "warn" | "ok" | "bad" | "link";

const LABELS: Record<string, { label: string; tone: Tone }> = {
  critical: { label: "Critical", tone: "bad" },
  high: { label: "High", tone: "signal" },
  medium: { label: "Medium", tone: "warn" },
  low: { label: "Low", tone: "neutral" },
  open: { label: "Open", tone: "link" },
  needs_feedback: { label: "Needs feedback", tone: "warn" },
  closed: { label: "Closed", tone: "neutral" },
  confident: { label: "Confident", tone: "ok" },
  abstained: { label: "Abstained", tone: "warn" },
  uncalibrated: { label: "Uncalibrated", tone: "neutral" },
  top_1: { label: "Top 1", tone: "ok" },
  top_3: { label: "Top 3", tone: "link" },
  missed: { label: "Missed", tone: "bad" },
  unknown: { label: "Unknown", tone: "neutral" },
};

/** A status, severity, decision or outcome. The text always carries the meaning. */
export function Tag({ value, tone }: { value: string; tone?: Tone }) {
  const known = LABELS[value];
  const name = tone ?? known?.tone ?? "neutral";
  return <span className={`${styles.tag} ${styles[name]}`}>{known?.label ?? value}</span>;
}
