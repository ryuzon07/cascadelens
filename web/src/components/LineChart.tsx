import { useState } from "react";
import type { ServiceEvidence } from "../api/client";
import styles from "./charts.module.css";

type Series = ServiceEvidence["series"][number];

const WIDTH = 320;
const HEIGHT = 110;
const PAD = { left: 8, right: 8, top: 16, bottom: 16 };

function paths(points: Series["points"], x: (ts: number) => number, y: (v: number) => number) {
  const runs: string[] = [];
  let run = "";
  for (const point of points) {
    if (point.value === null || point.value === undefined) {
      if (run) runs.push(run);
      run = "";
    } else {
      run += `${run ? "L" : "M"}${x(point.ts).toFixed(1)} ${y(point.value).toFixed(1)}`;
    }
  }
  if (run) runs.push(run);
  return runs;
}

/** One metric over the window. Missing samples leave a gap, and nothing is interpolated. */
export function LineChart({ series }: { series: Series }) {
  const [hover, setHover] = useState<Series["points"][number] | null>(null);
  const values = series.points.flatMap((point) => point.value ?? []);
  const times = series.points.map((point) => point.ts);
  const missing = series.points.length - values.length;
  const summary = `${series.metric}: ${series.points.length} points, ${missing} missing`;
  if (values.length === 0) {
    return (
      <figure style={{ margin: 0 }}>
        <figcaption className={styles.chartTitle}>{series.metric}</figcaption>
        <p>No samples in this window.</p>
      </figure>
    );
  }
  const [low, high] = [Math.min(...values), Math.max(...values)];
  const [first, last] = [Math.min(...times), Math.max(...times)];
  const x = (ts: number) =>
    PAD.left + ((ts - first) / Math.max(last - first, 1)) * (WIDTH - PAD.left - PAD.right);
  const y = (value: number) =>
    HEIGHT -
    PAD.bottom -
    ((value - low) / Math.max(high - low, 1e-9)) * (HEIGHT - PAD.top - PAD.bottom);

  function nearest(position: number) {
    const target = first + (position * (last - first)) / 1;
    return series.points.reduce((best, point) =>
      Math.abs(point.ts - target) < Math.abs(best.ts - target) ? point : best,
    );
  }

  return (
    <figure style={{ margin: 0 }}>
      <figcaption className={styles.chartTitle}>{series.metric}</figcaption>
      <svg
        className={styles.svg}
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        role="img"
        aria-label={summary}
        onMouseLeave={() => setHover(null)}
        onMouseMove={(event) => {
          const box = event.currentTarget.getBoundingClientRect();
          setHover(nearest((event.clientX - box.left) / Math.max(box.width, 1)));
        }}
      >
        <line
          x1={PAD.left}
          x2={WIDTH - PAD.right}
          y1={HEIGHT - PAD.bottom}
          y2={HEIGHT - PAD.bottom}
          stroke="var(--rule)"
        />
        {paths(series.points, x, y).map((d) => (
          <path key={d} d={d} fill="none" stroke="var(--ink)" strokeWidth={2} />
        ))}
        <text x={PAD.left} y={11}>
          max {high.toPrecision(3)}
        </text>
        <text x={PAD.left} y={HEIGHT - 3}>
          min {low.toPrecision(3)}
        </text>
        {hover && (
          <>
            <line
              x1={x(hover.ts)}
              x2={x(hover.ts)}
              y1={PAD.top}
              y2={HEIGHT - PAD.bottom}
              stroke="var(--link)"
            />
            <text x={WIDTH - PAD.right} y={11} textAnchor="end">
              +{hover.ts - first}s: {hover.value?.toPrecision(3) ?? "missing"}
            </text>
          </>
        )}
      </svg>
    </figure>
  );
}
