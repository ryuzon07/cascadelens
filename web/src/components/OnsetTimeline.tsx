import type { IncidentTimeline } from "../api/client";
import styles from "./charts.module.css";

const WIDTH = 640;
const ROW = 24;
const LABEL = 110;
const TOP = 18;

function describe(row: IncidentTimeline["services"][number], start: number): string {
  if (!row.has_telemetry) return `${row.service}: no telemetry`;
  const onset = row.onset === null ? "no anomaly" : `anomalous from +${row.onset - start}s`;
  const stopped =
    row.telemetry_stopped === null
      ? ""
      : `, telemetry stopped at +${row.telemetry_stopped - start}s`;
  return `${row.service}: ${onset}${stopped}`;
}

/** Each service's anomalous period against the window. Earliest onset comes first. */
export function OnsetTimeline({ timeline }: { timeline: IncidentTimeline }) {
  const { window_start: start, window_end: end, baseline_end: baseline } = timeline;
  const x = (ts: number) => LABEL + ((ts - start) / Math.max(end - start, 1)) * (WIDTH - LABEL - 8);
  const rows = [...timeline.services].sort(
    (a, b) =>
      (a.onset ?? Number.MAX_SAFE_INTEGER) - (b.onset ?? Number.MAX_SAFE_INTEGER) ||
      a.service.localeCompare(b.service),
  );
  const height = TOP + rows.length * ROW + 4;
  return (
    <figure style={{ margin: 0 }}>
      <svg
        className={styles.svg}
        viewBox={`0 0 ${WIDTH} ${height}`}
        role="img"
        aria-label={rows.map((row) => describe(row, start)).join("; ")}
      >
        <rect
          x={LABEL}
          y={TOP}
          width={x(baseline) - LABEL}
          height={height - TOP}
          fill="var(--raised)"
        />
        <text x={LABEL} y={11}>
          baseline
        </text>
        <text x={WIDTH - 8} y={11} textAnchor="end">
          +{end - start}s
        </text>
        {rows.map((row, index) => {
          const y = TOP + index * ROW;
          return (
            <g key={row.service}>
              <text x={0} y={y + 16}>
                {row.service}
              </text>
              <line x1={LABEL} x2={WIDTH - 8} y1={y + 12} y2={y + 12} stroke="var(--rule)" />
              {row.onset !== null && (
                <rect
                  x={x(row.onset)}
                  y={y + 6}
                  width={Math.max(x(row.anomalous_until ?? row.onset) - x(row.onset), 3)}
                  height={12}
                  fill="var(--signal)"
                />
              )}
              {row.telemetry_stopped !== null && (
                <path
                  d={`M${x(row.telemetry_stopped)} ${y + 5}v14`}
                  stroke="var(--ink)"
                  strokeWidth={2}
                  strokeDasharray="3 2"
                />
              )}
            </g>
          );
        })}
      </svg>
      <figcaption className={styles.gaugeLabels}>
        Shaded: baseline. Solid bar: anomalous. Dashed mark: telemetry stopped.
      </figcaption>
    </figure>
  );
}
