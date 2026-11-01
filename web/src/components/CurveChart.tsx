import styles from "./charts.module.css";

export interface Curve {
  label: string;
  points: { x: number; y: number }[];
}

interface CurveChartProps {
  label: string;
  xLabel: string;
  yLabel: string;
  series: Curve[];
  threshold?: number | null;
  formatX?: (value: number) => string;
}

const WIDTH = 560;
const HEIGHT = 220;
const PAD = { left: 40, right: 12, top: 24, bottom: 34 };
const STROKES = [
  { color: "var(--ink)", dash: undefined },
  { color: "var(--signal)", dash: "6 3" },
  { color: "var(--link)", dash: "2 3" },
  { color: "var(--warn)", dash: "8 3 2 3" },
];

/** Lines on a 0 to 1 axis. Each line has its own dash, so colour is never alone. */
export function CurveChart({
  label,
  xLabel,
  yLabel,
  series,
  threshold,
  formatX = String,
}: CurveChartProps) {
  const xs = series.flatMap((curve) => curve.points.map((point) => point.x));
  const [low, high] = [Math.min(...xs), Math.max(...xs)];
  const x = (value: number) =>
    PAD.left + ((value - low) / Math.max(high - low, 1e-9)) * (WIDTH - PAD.left - PAD.right);
  const y = (value: number) => HEIGHT - PAD.bottom - value * (HEIGHT - PAD.top - PAD.bottom);
  const summary = series
    .map(
      (curve) =>
        `${curve.label}: ${curve.points
          .map((point) => `${formatX(point.x)} ${point.y.toFixed(3)}`)
          .join(", ")}`,
    )
    .join("; ");
  return (
    <figure style={{ margin: 0, maxWidth: WIDTH }}>
      <svg
        className={styles.svg}
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        role="img"
        aria-label={`${label}. ${summary}`}
      >
        {[0, 0.5, 1].map((tick) => (
          <g key={tick}>
            <line
              x1={PAD.left}
              x2={WIDTH - PAD.right}
              y1={y(tick)}
              y2={y(tick)}
              stroke="var(--rule)"
            />
            <text x={PAD.left - 6} y={y(tick) + 4} textAnchor="end">
              {tick}
            </text>
          </g>
        ))}
        {threshold != null && (
          <g>
            <line
              x1={PAD.left}
              x2={WIDTH - PAD.right}
              y1={y(threshold)}
              y2={y(threshold)}
              stroke="var(--bad)"
              strokeDasharray="4 2"
            />
            <text x={WIDTH - PAD.right} y={y(threshold) + 13} textAnchor="end">
              threshold
            </text>
          </g>
        )}
        {series.map((curve, index) => {
          const stroke = STROKES[index % STROKES.length]!;
          const d = curve.points
            .map(
              (point, place) =>
                `${place ? "L" : "M"}${x(point.x).toFixed(1)} ${y(point.y).toFixed(1)}`,
            )
            .join("");
          return (
            <g key={curve.label}>
              <path
                d={d}
                fill="none"
                stroke={stroke.color}
                strokeWidth={2}
                strokeDasharray={stroke.dash}
              />
              {curve.points.map((point) => (
                <circle key={point.x} cx={x(point.x)} cy={y(point.y)} r={3} fill={stroke.color} />
              ))}
            </g>
          );
        })}
        <text x={PAD.left} y={HEIGHT - 18}>
          {formatX(low)}
        </text>
        <text x={WIDTH - PAD.right} y={HEIGHT - 18} textAnchor="end">
          {formatX(high)}
        </text>
        <text x={(PAD.left + WIDTH - PAD.right) / 2} y={HEIGHT - 4} textAnchor="middle">
          {xLabel}
        </text>
        <text x={4} y={9}>
          {yLabel}
        </text>
      </svg>
      {series.length > 1 && (
        <ul className={styles.legend}>
          {series.map((curve, index) => {
            const stroke = STROKES[index % STROKES.length]!;
            return (
              <li key={curve.label}>
                <svg width={22} height={6} aria-hidden="true">
                  <line
                    x1={0}
                    x2={22}
                    y1={3}
                    y2={3}
                    stroke={stroke.color}
                    strokeWidth={2}
                    strokeDasharray={stroke.dash}
                  />
                </svg>
                {curve.label}
              </li>
            );
          })}
        </ul>
      )}
    </figure>
  );
}
