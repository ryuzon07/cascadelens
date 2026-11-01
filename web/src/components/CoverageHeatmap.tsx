import type { IncidentCoverage } from "../api/client";
import { formatPercent } from "../format";
import styles from "./charts.module.css";
import primitives from "./primitives.module.css";

function tone(coverage: number): string | undefined {
  if (coverage >= 1) return styles.full;
  return coverage > 0 ? styles.partial : styles.none;
}

/** Telemetry coverage. Every cell states its share, so colour is never alone. */
export function CoverageHeatmap({ coverage }: { coverage: IncidentCoverage }) {
  const cells = new Map(coverage.cells.map((cell) => [`${cell.service}/${cell.family}`, cell]));
  return (
    // The box scrolls on narrow screens, so keyboard users must be able to focus it.
    // eslint-disable-next-line jsx-a11y/no-noninteractive-tabindex
    <div className={primitives.tableBox} tabIndex={0} role="region" aria-label="Telemetry coverage">
      <table className={`${primitives.table} ${styles.heatmap}`}>
        <caption className="visually-hidden">Share of expected samples present</caption>
        <thead>
          <tr>
            <th scope="col">Service</th>
            {coverage.families.map((family) => (
              <th key={family} scope="col">
                {family}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {coverage.services.map((service) => (
            <tr key={service}>
              <th scope="row">{service}</th>
              {coverage.families.map((family) => {
                const share = cells.get(`${service}/${family}`)?.coverage ?? 0;
                return (
                  <td key={family} className={tone(share)}>
                    {formatPercent(share)}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
