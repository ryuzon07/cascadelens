import { formatUnit } from "../format";
import styles from "./charts.module.css";

interface ConfidenceGaugeProps {
  confidence: number;
  threshold: number | null;
}

/** Confidence against the abstention threshold. The numbers stay in text beside the bar. */
export function ConfidenceGauge({ confidence, threshold }: ConfidenceGaugeProps) {
  const low = threshold !== null && confidence < threshold;
  const fill = low ? `${styles.gaugeFill} ${styles.gaugeFillLow}` : styles.gaugeFill;
  return (
    <div className={styles.gauge}>
      <div className={styles.gaugeLabels}>
        <span>Confidence {formatUnit(confidence)}</span>
        <span>{threshold === null ? "No threshold" : `Threshold ${formatUnit(threshold)}`}</span>
      </div>
      <div
        className={styles.gaugeTrack}
        role="meter"
        aria-label="Confidence"
        aria-valuemin={0}
        aria-valuemax={1}
        aria-valuenow={confidence}
        aria-valuetext={formatUnit(confidence)}
      >
        <div className={fill} style={{ width: `${confidence * 100}%` }} />
        {threshold !== null && (
          <span className={styles.gaugeMark} style={{ left: `calc(${threshold * 100}% - 1px)` }} />
        )}
      </div>
    </div>
  );
}
