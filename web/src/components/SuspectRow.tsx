import type { IncidentDetail } from "../api/client";
import { formatPercent } from "../format";
import styles from "./decision.module.css";

type Ranked = IncidentDetail["diagnosis"]["ranking"][number];

function note(item: Ranked): string {
  const { onset_rank: onset, missing_frac: missing } = item.evidence;
  const parts = [onset == null ? "no sustained anomaly" : `onset rank ${onset}`];
  if (missing == null) parts.push("telemetry unknown");
  else if (missing > 0) parts.push(`${formatPercent(missing)} missing`);
  return parts.join(", ");
}

interface SuspectRowProps {
  item: Ranked;
  rank: number;
  selected: boolean;
  onSelect: (service: string) => void;
}

export function SuspectRow({ item, rank, selected, onSelect }: SuspectRowProps) {
  return (
    <button
      type="button"
      className={styles.suspect}
      aria-pressed={selected}
      onClick={() => onSelect(item.service)}
    >
      <span className={styles.score}>{rank}</span>
      <span>
        <span className={styles.suspectName}>{item.service}</span>
        <span className={styles.suspectNote}>{note(item)}</span>
      </span>
      <span className={styles.score}>{item.score.toFixed(2)}</span>
    </button>
  );
}
