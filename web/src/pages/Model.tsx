import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { api, queryKeys, type ModelReport } from "../api/client";
import { Card, CardBody, CardHeader, PENDING } from "../components/Card";
import { CurveChart } from "../components/CurveChart";
import { DataTable, type Column } from "../components/DataTable";
import { Select } from "../components/Fields";
import { QueryView } from "../components/QueryView";
import { ReliabilityDiagram } from "../components/ReliabilityDiagram";
import { EmptyState } from "../components/States";
import { Tag } from "../components/Tag";
import { formatPercent } from "../format";
import styles from "./pages.module.css";

type Result = ModelReport["results"][number];
type Robustness = ModelReport["robustness"][number];

const METHOD_NAMES: Record<string, string> = {
  ranker: "Ranker",
  ranker_augmented: "Ranker, augmented",
  lightgbm_flat: "LightGBM flat",
  lightgbm_flat_masking: "LightGBM flat, masking",
};
const PROPORTIONS = /Top-|Avg@/;
const INTENDED = [
  "A read-only aid that ranks likely origin services for one incident window, for an engineer to check.",
  "It gives a confidence and declines to answer when confidence is below the threshold.",
];
const LIMITS = [
  "Results come from injected benchmark faults, not production incidents.",
  "Perturbation results are simulated.",
  "It is not proof of causation, and it does not model changing topology.",
  "The served model comes from one training fold, not from all the data.",
  "It must not trigger remediation, and the API must not be exposed publicly.",
];

function value(row: Result, number: number | null | undefined): string {
  if (number === null || number === undefined) return PENDING;
  return PROPORTIONS.test(row.name) ? formatPercent(number) : number.toFixed(3);
}

const RESULT_COLUMNS: Column<Result>[] = [
  { key: "name", header: "Result", render: (row) => row.name },
  {
    key: "estimate",
    header: "Estimate",
    render: (row) => <span className="mono">{value(row, row.estimate)}</span>,
  },
  {
    key: "interval",
    header: "95% interval",
    render: (row) => (
      <span className={`mono ${styles.nowrap}`}>
        {row.lower == null || row.upper == null
          ? PENDING
          : `${value(row, row.lower)} to ${value(row, row.upper)}`}
      </span>
    ),
  },
  { key: "source", header: "Source", render: (row) => <code>{row.source}</code> },
];

function hasPending(report: ModelReport): boolean {
  return (
    !report.result_status ||
    report.results.some((row) => row.estimate === null || row.estimate === undefined) ||
    report.checks.some((check) => check.passed === null || check.passed === undefined) ||
    !report.reliability
  );
}

function checkTag(passed: boolean | null | undefined) {
  if (passed === null || passed === undefined) return <Tag value={PENDING} />;
  return passed ? <Tag value="Passed" tone="ok" /> : <Tag value="Flagged" tone="bad" />;
}

function pointLabel(point: string): string {
  return point.split(":").slice(1).join(":") || point;
}

function RobustnessCurves({ rows }: { rows: Robustness[] }) {
  const families = [...new Set(rows.map((row) => row.perturbation))]
    .filter((name) => name !== "clean")
    .sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
  const [family, setFamily] = useState("P2");
  const chosen = families.includes(family) ? family : families[0];
  if (!chosen) return <EmptyState title={PENDING}>No robustness results are committed.</EmptyState>;
  const shown = rows.filter((row) => row.perturbation === chosen || row.perturbation === "clean");
  const points = [...new Set(shown.map((row) => row.point))];
  const methods = [...new Set(shown.map((row) => row.method))];
  const series = methods.map((method) => ({
    label: METHOD_NAMES[method] ?? method,
    points: shown
      .filter((row) => row.method === method)
      .map((row) => ({ x: points.indexOf(row.point), y: row.estimate })),
  }));
  const columns: Column<string>[] = [
    { key: "point", header: "Grid point", render: (point) => pointLabel(point) },
    ...methods.map((method) => ({
      key: method,
      header: METHOD_NAMES[method] ?? method,
      render: (point: string) => {
        const row = shown.find((item) => item.point === point && item.method === method);
        return row ? (
          <span className={`mono ${styles.nowrap}`}>
            {formatPercent(row.estimate)} [{formatPercent(row.lower)}, {formatPercent(row.upper)}]
          </span>
        ) : (
          PENDING
        );
      },
    })),
  ];
  return (
    <div className={styles.stack}>
      <div className={styles.tags}>
        <Select
          label="Perturbation"
          value={chosen}
          onChange={(event) => setFamily(event.target.value)}
          options={families.map((name) => ({ value: name, label: name }))}
        />
        <Tag value="Simulated" tone="warn" />
      </div>
      <CurveChart
        label={`Top-1 under ${chosen}, simulated`}
        xLabel="Grid point, in the order of the table. 0 is clean."
        yLabel="Top-1"
        series={series}
      />
      <DataTable
        caption={`Top-1 under ${chosen}`}
        columns={columns}
        rows={points}
        rowKey={(point) => point}
      />
    </div>
  );
}

function Report({ report }: { report: ModelReport }) {
  const facts = [
    ["Version", report.model_version],
    ["Method", report.method],
    ["Split", `${report.split}, fold ${report.fold}, seed ${report.seed}`],
    ["Result status", report.result_status ?? PENDING],
    ["Calibrated", report.calibrated ? "yes" : "no"],
    ["Frozen config", report.frozen_config_hash?.slice(0, 12) ?? "not frozen"],
    ["Data checksum", report.training_data_checksum.slice(0, 12)],
  ];
  return (
    <>
      <Card>
        <CardHeader title="Model" />
        <CardBody>
          <dl className={styles.facts}>
            {facts.map(([name, text]) => (
              <div key={name}>
                <dt>{name}</dt>
                <dd>{text}</dd>
              </div>
            ))}
          </dl>
        </CardBody>
      </Card>
      {hasPending(report) && (
        <p className={styles.notice} role="note">
          Some values are shown as {PENDING} because no committed result holds them.
          {!report.frozen_config_hash && " No config is frozen, so results are exploratory."}
        </p>
      )}
      <Card>
        <CardHeader title="Results" />
        <DataTable
          caption="Results from committed result files"
          columns={RESULT_COLUMNS}
          rows={report.results}
          rowKey={(row) => row.name}
        />
      </Card>
      <div className={styles.pair}>
        <Card>
          <CardHeader title="Calibration" />
          <CardBody>
            <ReliabilityDiagram bins={report.reliability} />
          </CardBody>
        </Card>
        <Card>
          <CardHeader title={`Risk and coverage: ${report.calibrated_method}`} />
          <CardBody>
            {report.risk_coverage.length === 0 ? (
              <EmptyState title={PENDING}>No coverage results are committed.</EmptyState>
            ) : (
              <CurveChart
                label="Top-1 on kept cases against coverage"
                xLabel="Coverage"
                yLabel="Top-1 kept"
                formatX={formatPercent}
                series={[
                  {
                    label: "Top-1 on kept cases",
                    points: report.risk_coverage.map((point) => ({
                      x: point.coverage,
                      y: point.estimate,
                    })),
                  },
                ]}
              />
            )}
          </CardBody>
        </Card>
      </div>
      <Card>
        <CardHeader title="Robustness" />
        <CardBody>
          <RobustnessCurves rows={report.robustness} />
        </CardBody>
      </Card>
      <Card>
        <CardHeader title="Leakage and calibration checks" />
        <CardBody>
          <ul className={styles.labelList}>
            {report.checks.map((check) => (
              <li key={check.name} className={styles.check}>
                {check.name} {checkTag(check.passed)} <code>{check.source}</code>
              </li>
            ))}
          </ul>
        </CardBody>
      </Card>
      <div className={styles.pair}>
        <Card>
          <CardHeader title="Intended use" />
          <CardBody>
            <ul className={styles.notes}>
              {INTENDED.map((line) => (
                <li key={line}>{line}</li>
              ))}
            </ul>
          </CardBody>
        </Card>
        <Card>
          <CardHeader title="Limits" />
          <CardBody>
            <ul className={styles.notes}>
              {LIMITS.map((line) => (
                <li key={line}>{line}</li>
              ))}
            </ul>
          </CardBody>
        </Card>
      </div>
    </>
  );
}

export function Model() {
  const model = useQuery({ queryKey: queryKeys.model, queryFn: ({ signal }) => api.model(signal) });
  return (
    <div className={styles.page}>
      <h1 className="visually-hidden">Model</h1>
      <QueryView query={model} label="the model" rows={6}>
        {(report) => <Report report={report} />}
      </QueryView>
    </div>
  );
}
