import styles from "./charts.module.css";

export interface Segment {
  label: string;
  value: number;
  color: string;
}

/** Shares of a whole. The legend carries each count, so colour is never alone. */
export function StackedBars({ segments, label }: { segments: Segment[]; label: string }) {
  const total = segments.reduce((sum, segment) => sum + segment.value, 0);
  const summary = segments.map((segment) => `${segment.label} ${segment.value}`).join(", ");
  return (
    <div>
      <div className={styles.stack} role="img" aria-label={`${label}: ${summary}`}>
        {segments
          .filter((segment) => segment.value > 0)
          .map((segment) => (
            <span
              key={segment.label}
              style={{ flexGrow: segment.value / Math.max(total, 1), background: segment.color }}
            />
          ))}
      </div>
      <ul className={styles.stackLegend}>
        {segments.map((segment) => (
          <li key={segment.label}>
            <span className={styles.swatch} style={{ background: segment.color }} />
            {segment.label}: {segment.value}
          </li>
        ))}
      </ul>
    </div>
  );
}
