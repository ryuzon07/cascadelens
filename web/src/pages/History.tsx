import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { Link } from "react-router-dom";
import { api, queryKeys, type FeedbackLabel, type HistoryReport } from "../api/client";
import { Button } from "../components/Button";
import { Card, CardBody, CardHeader, KpiTile } from "../components/Card";
import { DataTable, type Column } from "../components/DataTable";
import { Select } from "../components/Fields";
import { QueryView } from "../components/QueryView";
import { StackedBars } from "../components/StackedBars";
import { EmptyState } from "../components/States";
import { Tag } from "../components/Tag";
import { downloadCsv, labelsToCsv } from "../csv";
import { formatPercent, formatTime, formatUnit } from "../format";
import styles from "./pages.module.css";

const DAY = 86_400;
const RANGES = [
  { value: "", label: "All time" },
  { value: "30", label: "Last 30 days" },
  { value: "90", label: "Last 90 days" },
];
const COLUMNS: Column<FeedbackLabel>[] = [
  {
    key: "incident",
    header: "Incident",
    render: (row) => (
      <Link className={`mono ${styles.nowrap}`} to={`/incidents/${row.incident_id}`}>
        {row.incident_id}
      </Link>
    ),
    sortValue: (row) => row.incident_id,
  },
  { key: "title", header: "Title", render: (row) => row.title },
  {
    key: "predicted",
    header: "Predicted #1",
    render: (row) => <span className="mono">{row.predicted_service}</span>,
  },
  {
    key: "actual",
    header: "Confirmed cause",
    render: (row) =>
      row.actual_service ? <span className="mono">{row.actual_service}</span> : "Not known",
  },
  { key: "outcome", header: "Outcome", render: (row) => <Tag value={row.outcome} /> },
  {
    key: "confidence",
    header: "Confidence",
    render: (row) => <span className="mono">{formatUnit(row.confidence)}</span>,
    sortValue: (row) => row.confidence,
  },
  { key: "labelled", header: "Labelled by", render: (row) => row.labelled_by },
  {
    key: "when",
    header: "Labelled",
    render: (row) => <span className={styles.nowrap}>{formatTime(row.created_at)}</span>,
    sortValue: (row) => row.created_at,
  },
];

function rateHint(rate: HistoryReport["top_1"]): string {
  if (!rate) return "No labels yet";
  const interval = `${formatPercent(rate.lower)} to ${formatPercent(rate.upper)}`;
  return `95% interval ${interval}, ${rate.labels} labelled`;
}

function Report({ report }: { report: HistoryReport }) {
  return (
    <>
      <div className={styles.kpis}>
        <KpiTile
          label="Top-1 match"
          value={report.top_1 ? formatPercent(report.top_1.estimate) : null}
          hint={rateHint(report.top_1)}
        />
        <KpiTile
          label="Top-3 match"
          value={report.top_3 ? formatPercent(report.top_3.estimate) : null}
          hint={rateHint(report.top_3)}
        />
        <KpiTile label="Labelled incidents" value={String(report.items.length)} />
      </div>
      <p className={styles.meta}>
        These rates cover only incidents someone labelled, so they are not a fair accuracy estimate.
        {report.includes_demo && " They include sample data."}
      </p>
      <Card>
        <CardHeader title="Outcomes by week" />
        <CardBody>
          {report.weeks.length === 0 ? (
            <EmptyState title="No labelled weeks yet" />
          ) : (
            <div className={styles.stack}>
              {report.weeks.map((week) => (
                <div key={week.week_start}>
                  <p className={styles.meta}>Week of {formatTime(week.week_start).slice(0, 10)}</p>
                  <StackedBars
                    label={`Outcomes, week of ${formatTime(week.week_start).slice(0, 10)}`}
                    segments={[
                      { label: "Top 1", value: week.top_1, color: "var(--ok)" },
                      { label: "Top 3", value: week.top_3, color: "var(--link)" },
                      { label: "Missed", value: week.missed, color: "var(--bad)" },
                      { label: "Unknown", value: week.unknown, color: "var(--rule-strong)" },
                    ]}
                  />
                </div>
              ))}
            </div>
          )}
        </CardBody>
      </Card>
      <Card>
        <CardHeader
          title="Labelled incidents"
          actions={
            <Button
              disabled={report.items.length === 0}
              onClick={() => downloadCsv(labelsToCsv(report.items), "cascadelens-labels.csv")}
            >
              Export CSV
            </Button>
          }
        />
        {report.items.length === 0 ? (
          <CardBody>
            <EmptyState title="No labels yet">Label an incident from its detail page.</EmptyState>
          </CardBody>
        ) : (
          <DataTable
            caption="Labelled incidents"
            columns={COLUMNS}
            rows={report.items}
            rowKey={(row) => row.incident_id}
          />
        )}
      </Card>
    </>
  );
}

export function History() {
  const [range, setRange] = useState("");
  const [since, setSince] = useState<number | undefined>(undefined);
  const history = useQuery({
    queryKey: [...queryKeys.history, since],
    queryFn: ({ signal }) => api.history(since, signal),
  });
  return (
    <div className={styles.page}>
      <h1 className="visually-hidden">History and feedback</h1>
      <div>
        <Select
          label="Time range"
          options={RANGES}
          value={range}
          onChange={(event) => {
            const days = event.target.value;
            setRange(days);
            setSince(days ? Math.floor(Date.now() / 1000) - Number(days) * DAY : undefined);
          }}
        />
      </div>
      <QueryView query={history} label="history" rows={5}>
        {(report) => <Report report={report} />}
      </QueryView>
    </div>
  );
}
