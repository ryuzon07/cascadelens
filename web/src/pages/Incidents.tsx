import { useInfiniteQuery, useQuery } from "@tanstack/react-query";
import { useMemo, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { api, queryKeys, type IncidentQuery, type IncidentSummary } from "../api/client";
import { Button } from "../components/Button";
import { Card, CardBody, CardHeader, KpiTile } from "../components/Card";
import { DataTable, type Column } from "../components/DataTable";
import { Select } from "../components/Fields";
import { StackedBars } from "../components/StackedBars";
import { EmptyState, ErrorState, Skeleton, StaleIndicator } from "../components/States";
import { Tabs } from "../components/Tabs";
import { Tag } from "../components/Tag";
import { formatPercent, formatTime, formatUnit } from "../format";
import styles from "./pages.module.css";

const DAY = 86_400;
const PAGE = 50;
const STATUS_TABS = [
  { id: "all", label: "All" },
  { id: "open", label: "Open" },
  { id: "needs_feedback", label: "Needs feedback" },
];
const SEVERITIES = [
  { value: "", label: "Any" },
  { value: "critical", label: "Critical" },
  { value: "high", label: "High" },
  { value: "medium", label: "Medium" },
  { value: "low", label: "Low" },
];
const RANGES = [
  { value: "", label: "All time" },
  { value: "1", label: "Last 24 hours" },
  { value: "7", label: "Last 7 days" },
  { value: "30", label: "Last 30 days" },
];

const COLUMNS: Column<IncidentSummary>[] = [
  {
    key: "id",
    header: "ID",
    render: (row) => (
      <Link className={`mono ${styles.nowrap}`} to={`/incidents/${row.id}`}>
        {row.id}
      </Link>
    ),
    sortValue: (row) => row.id,
  },
  { key: "title", header: "Title", render: (row) => row.title, sortValue: (row) => row.title },
  { key: "severity", header: "Severity", render: (row) => <Tag value={row.severity} /> },
  { key: "status", header: "Status", render: (row) => <Tag value={row.status} /> },
  {
    key: "started",
    header: "Started",
    render: (row) => <span className={styles.nowrap}>{formatTime(row.window_start)}</span>,
    sortValue: (row) => row.window_start,
  },
  {
    key: "suspect",
    header: "Top suspect",
    render: (row) => (
      <span className={styles.bar}>
        <span className="mono">{row.top_service}</span>
        <span className={styles.barTrack} aria-hidden="true">
          <span className={styles.barFill} style={{ width: `${row.confidence * 100}%` }} />
        </span>
        <span className="mono">{formatUnit(row.confidence)}</span>
      </span>
    ),
    sortValue: (row) => row.confidence,
  },
  { key: "decision", header: "Decision", render: (row) => <Tag value={row.decision} /> },
  { key: "owner", header: "Owner", render: (row) => row.owner ?? "Unassigned" },
];

function daysAgo(days: number): number {
  return Math.floor(Date.now() / 1000) - days * DAY;
}

function matches(row: IncidentSummary, text: string): boolean {
  const wanted = text.toLowerCase();
  return [row.id, row.title, row.top_service].some((value) => value.toLowerCase().includes(wanted));
}

function Kpis({ rows }: { rows: IncidentSummary[] }) {
  const [since] = useState(() => daysAgo(30));
  const history = useQuery({
    queryKey: [...queryKeys.history, since],
    queryFn: ({ signal }) => api.history(since, signal),
  });
  const top3 = history.data?.top_3;
  const abstained = rows.filter((row) => row.decision === "abstained").length;
  return (
    <div className={styles.kpis}>
      <KpiTile
        label="Open incidents"
        value={String(rows.filter((row) => row.status !== "closed").length)}
      />
      <KpiTile
        label="Time to first lead"
        value={null}
        hint="Median needs alert times, which incidents do not hold yet"
      />
      <KpiTile
        label="Top-3 match, 30 days"
        value={top3 ? formatPercent(top3.estimate) : null}
        hint={top3 ? `${top3.labels} labelled` : "No labels yet"}
      />
      <KpiTile
        label="Abstain rate"
        value={rows.length ? formatPercent(abstained / rows.length) : null}
        hint={`${rows.length} loaded`}
      />
    </div>
  );
}

function shares(rows: IncidentSummary[]) {
  const count = (decision: string) => rows.filter((row) => row.decision === decision).length;
  return [
    { label: "Confident", value: count("confident"), color: "var(--ink)" },
    { label: "Abstained", value: count("abstained"), color: "var(--warn)" },
    { label: "Uncalibrated", value: count("uncalibrated"), color: "var(--rule-strong)" },
  ];
}

export function Incidents() {
  const [params] = useSearchParams();
  const [status, setStatus] = useState("all");
  const [severity, setSeverity] = useState("");
  const [service, setService] = useState("");
  const [range, setRange] = useState("");
  const [since, setSince] = useState<number | undefined>(undefined);
  const text = params.get("q") ?? "";

  const query: IncidentQuery = useMemo(
    () => ({
      status: status === "all" ? undefined : status,
      severity,
      service,
      since,
      limit: PAGE,
    }),
    [status, severity, service, since],
  );
  const list = useInfiniteQuery({
    queryKey: queryKeys.incidents(query),
    queryFn: ({ pageParam, signal }) => api.incidents({ ...query, cursor: pageParam }, signal),
    initialPageParam: undefined as string | undefined,
    getNextPageParam: (page) => page.next_cursor ?? undefined,
  });
  const everything = useQuery({
    queryKey: queryKeys.incidents({ limit: 200 }),
    queryFn: ({ signal }) => api.incidents({ limit: 200 }, signal),
  });

  const all = everything.data?.items ?? [];
  const rows = (list.data?.pages.flatMap((page) => page.items) ?? []).filter((row) =>
    text ? matches(row, text) : true,
  );
  const waiting = all.filter((row) => row.status === "needs_feedback");
  const services = [...new Set(all.map((row) => row.top_service))].sort();

  return (
    <div className={styles.page}>
      <h1 className="visually-hidden">Incidents</h1>
      <Kpis rows={all} />
      <div className={styles.stack}>
        <Card>
          <CardHeader title="Incidents" actions={<StaleIndicator active={list.isRefetching} />} />
          <Tabs label="Status" items={STATUS_TABS} selected={status} onSelect={setStatus} />
          <div className={styles.filters}>
            <Select
              label="Severity"
              options={SEVERITIES}
              value={severity}
              onChange={(event) => setSeverity(event.target.value)}
            />
            <Select
              label="Top suspect"
              options={[
                { value: "", label: "Any" },
                ...services.map((name) => ({ value: name, label: name })),
              ]}
              value={service}
              onChange={(event) => setService(event.target.value)}
            />
            <Select
              label="Time range"
              options={RANGES}
              value={range}
              onChange={(event) => {
                setRange(event.target.value);
                setSince(event.target.value ? daysAgo(Number(event.target.value)) : undefined);
              }}
            />
          </div>
          {list.isPending && <Skeleton label="Loading incidents" rows={5} />}
          {list.isError && <ErrorState error={list.error} onRetry={() => void list.refetch()} />}
          {list.isSuccess && rows.length === 0 && (
            <EmptyState title="No incidents match">
              {text ? `Nothing matches "${text}".` : "Try wider filters."}
            </EmptyState>
          )}
          {rows.length > 0 && (
            <DataTable caption="Incidents" columns={COLUMNS} rows={rows} rowKey={(row) => row.id} />
          )}
          {list.hasNextPage && (
            <CardBody>
              <Button onClick={() => void list.fetchNextPage()} disabled={list.isFetchingNextPage}>
                Load more
              </Button>
            </CardBody>
          )}
        </Card>
        <div className={styles.pair}>
          <Card>
            <CardHeader title="Waiting for your label" />
            <CardBody>
              {waiting.length === 0 ? (
                <EmptyState title="Nothing is waiting" />
              ) : (
                <ul className={styles.labelList}>
                  {waiting.map((row) => (
                    <li key={row.id}>
                      <Link to={`/incidents/${row.id}`}>{row.title}</Link>
                    </li>
                  ))}
                </ul>
              )}
            </CardBody>
          </Card>
          <Card>
            <CardHeader title="Decision share" />
            <CardBody>
              <StackedBars label="Decisions" segments={shares(all)} />
            </CardBody>
          </Card>
        </div>
      </div>
    </div>
  );
}
