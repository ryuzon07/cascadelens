import { formatSigned } from "../format";
import styles from "./charts.module.css";

interface Item {
  label: string;
  contribution: number;
}

/** Signed contributions to a score. They explain the ranking, not the cause. */
export function ContributionBars({ items, label }: { items: Item[]; label: string }) {
  const largest = Math.max(...items.map((item) => Math.abs(item.contribution)), 1e-9);
  return (
    <ul style={{ margin: 0, padding: 0, listStyle: "none" }} aria-label={label}>
      {items.map((item) => {
        const width = (Math.abs(item.contribution) / largest) * 50;
        const up = item.contribution >= 0;
        return (
          <li key={item.label} className={styles.contribution}>
            <span>{item.label}</span>
            <span className={styles.contributionTrack} aria-hidden="true">
              <span
                className={`${styles.contributionBar} ${up ? styles.up : styles.down}`}
                style={
                  up ? { left: "50%", width: `${width}%` } : { right: "50%", width: `${width}%` }
                }
              />
            </span>
            <span className={styles.number}>{formatSigned(item.contribution)}</span>
          </li>
        );
      })}
    </ul>
  );
}
