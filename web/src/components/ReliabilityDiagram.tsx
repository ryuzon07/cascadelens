import type { ModelReport } from "../api/client";
import { PENDING } from "./Card";
import { EmptyState } from "./States";
import styles from "./charts.module.css";

type Bin = NonNullable<ModelReport["reliability"]>[number];

const SIZE = 220;
const PAD = 28;

/** Accuracy against confidence. Missing bins stay pending, never invented. */
export function ReliabilityDiagram({ bins }: { bins: Bin[] | null | undefined }) {
  if (!bins) {
    return (
      <EmptyState title={PENDING}>
        The committed results hold reliability figures, but not the bins behind them.
      </EmptyState>
    );
  }
  const place = (value: number) => PAD + value * (SIZE - 2 * PAD);
  const flip = (value: number) => SIZE - place(value);
  const summary = bins
    .map(
      (bin) =>
        `confidence ${bin.confidence.toFixed(2)}, accuracy ${bin.accuracy.toFixed(2)}, ${bin.cases} cases`,
    )
    .join("; ");
  return (
    <svg
      className={styles.svg}
      style={{ maxWidth: SIZE }}
      viewBox={`0 0 ${SIZE} ${SIZE}`}
      role="img"
      aria-label={`Reliability diagram: ${summary}`}
    >
      <line x1={place(0)} y1={flip(0)} x2={place(1)} y2={flip(1)} stroke="var(--rule-strong)" />
      {bins.map((bin) => (
        <circle
          key={bin.confidence}
          cx={place(bin.confidence)}
          cy={flip(bin.accuracy)}
          r={4}
          fill="var(--ink)"
        />
      ))}
      <text x={SIZE / 2} y={SIZE - 6} textAnchor="middle">
        confidence
      </text>
      <text x={4} y={14}>
        accuracy
      </text>
    </svg>
  );
}
