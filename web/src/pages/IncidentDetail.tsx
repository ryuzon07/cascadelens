import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api, queryKeys, type IncidentDetail as Incident } from "../api/client";
import { Button } from "../components/Button";
import { Card, CardBody, CardHeader } from "../components/Card";
import { ContributionBars } from "../components/ContributionBars";
import { CoverageHeatmap } from "../components/CoverageHeatmap";
import { DecisionCard } from "../components/DecisionCard";
import { DependencyGraph, GraphLegend } from "../components/DependencyGraph";
import { FeedbackForm } from "../components/FeedbackForm";
import { LineChart } from "../components/LineChart";
import { OnsetTimeline } from "../components/OnsetTimeline";
import { QueryView } from "../components/QueryView";
import { EmptyState } from "../components/States";
import { SuspectRow } from "../components/SuspectRow";
import { Tabs } from "../components/Tabs";
import { Tag } from "../components/Tag";
import chartStyles from "../components/charts.module.css";
import decisionStyles from "../components/decision.module.css";
import { formatTime, formatUnit } from "../format";
import styles from "./pages.module.css";

const TABS = [
  { id: "overview", label: "Overview" },
  { id: "evidence", label: "Evidence" },
  { id: "timeline", label: "Timeline" },
  { id: "what-if", label: "What-if" },
  { id: "raw", label: "Raw data" },
];
const TOPOLOGY: Record<string, string> = {
  provided: "provided with the case",
  inferred: "inferred from metrics",
  none: "not available",
};
const TOP_3 = "Ranker Top-3 (clean)";

function ShareLink() {
  const [state, setState] = useState("");
  async function copy() {
    try {
      await navigator.clipboard.writeText(window.location.href);
      setState("Link copied.");
    } catch {
      setState("Copy this page's address to share it.");
    }
  }
  return (
    <>
      <Button onClick={() => void copy()}>Share link</Button>
      <span role="status" className={styles.meta}>
        {state}
      </span>
    </>
  );
}

function Header({ incident }: { incident: Incident }) {
  const client = useQueryClient();
  const rerun = useMutation({
    mutationFn: () => api.rerun(incident.id),
    onSuccess: () =>
      Promise.all([
        client.invalidateQueries({ queryKey: queryKeys.incident(incident.id) }),
        client.invalidateQueries({ queryKey: ["incidents"] }),
        client.invalidateQueries({ queryKey: queryKeys.history }),
      ]),
  });
  return (
    <header className={styles.header}>
      <div>
        <div className={styles.tags}>
          <span className="mono">{incident.id}</span>
          <Tag value={incident.severity} />
          <Tag value={incident.status} />
        </div>
        <h1 className={styles.title}>{incident.title}</h1>
        <p className={styles.meta}>
          {formatTime(incident.window_start)} to {formatTime(incident.window_end)} ·{" "}
          {incident.service_count} services · {incident.sample_count} samples · topology{" "}
          {TOPOLOGY[incident.topology_source]}
        </p>
      </div>
      <div className={styles.actions}>
        <ShareLink />
        <Button
          variant="primary"
          onClick={() => rerun.mutate()}
          disabled={rerun.isPending}
          aria-label="Re-run diagnosis"
        >
          Re-run
        </Button>
        {rerun.isError && <span role="alert">{rerun.error.message}</span>}
      </div>
    </header>
  );
}

function Suspects({
  incident,
  selected,
  onSelect,
}: {
  incident: Incident;
  selected: string;
  onSelect: (service: string) => void;
}) {
  const leads = incident.decision === "abstained";
  return (
    <Card>
      <CardHeader title={leads ? "Leads to check" : "Ranked suspects"} />
      {incident.diagnosis.ranking.length === 0 ? (
        <EmptyState title="No service could be ranked" />
      ) : (
        <ol className={decisionStyles.suspects}>
          {incident.diagnosis.ranking.map((item, index) => (
            <li key={item.service}>
              <SuspectRow
                item={item}
                rank={index + 1}
                selected={item.service === selected}
                onSelect={onSelect}
              />
            </li>
          ))}
        </ol>
      )}
    </Card>
  );
}

function GraphCard({
  id,
  selected,
  onSelect,
}: {
  id: string;
  selected: string;
  onSelect: (service: string) => void;
}) {
  const graph = useQuery({
    queryKey: queryKeys.graph(id),
    queryFn: ({ signal }) => api.graph(id, signal),
  });
  return (
    <Card>
      <CardHeader title="Dependency graph" />
      <CardBody>
        <QueryView query={graph} label="the graph">
          {(data) =>
            data.edges.length === 0 ? (
              <>
                <EmptyState title="No call graph">
                  This case has no recorded or inferred calls, so graph evidence is unavailable.
                </EmptyState>
                <DependencyGraph graph={data} selected={selected} onSelect={onSelect} />
              </>
            ) : (
              <>
                <DependencyGraph graph={data} selected={selected} onSelect={onSelect} />
                <GraphLegend />
              </>
            )
          }
        </QueryView>
      </CardBody>
    </Card>
  );
}

function TimelineCard({ id }: { id: string }) {
  const timeline = useQuery({
    queryKey: queryKeys.timeline(id),
    queryFn: ({ signal }) => api.timeline(id, signal),
  });
  return (
    <Card>
      <CardHeader title="Onset timeline" />
      <CardBody>
        <QueryView query={timeline} label="the timeline">
          {(data) => <OnsetTimeline timeline={data} />}
        </QueryView>
      </CardBody>
    </Card>
  );
}

function EvidenceCards({ id, service }: { id: string; service: string }) {
  const evidence = useQuery({
    queryKey: queryKeys.evidence(id, service),
    queryFn: ({ signal }) => api.evidence(id, service, signal),
    enabled: Boolean(service),
  });
  return (
    <>
      <Card>
        <CardHeader title={`Feature contributions: ${service}`} />
        <CardBody>
          <QueryView query={evidence} label="the evidence">
            {(data) => (
              <>
                <ContributionBars
                  label="Contribution by feature group"
                  items={data.groups.map((item) => ({
                    label: item.group,
                    contribution: item.contribution,
                  }))}
                />
                <h3 className={styles.meta} style={{ marginTop: 12 }}>
                  Largest features
                </h3>
                <ContributionBars label="Contribution by feature" items={data.features} />
              </>
            )}
          </QueryView>
        </CardBody>
      </Card>
      <Card>
        <CardHeader title={`Metrics: ${service}`} />
        <CardBody>
          <QueryView query={evidence} label="the metrics">
            {(data) =>
              data.series.length === 0 ? (
                <EmptyState title="No metrics for this service" />
              ) : (
                <div className={chartStyles.chartGrid}>
                  {data.series.map((series) => (
                    <LineChart key={series.metric} series={series} />
                  ))}
                </div>
              )
            }
          </QueryView>
        </CardBody>
      </Card>
    </>
  );
}

function useCoverage(id: string) {
  return useQuery({
    queryKey: queryKeys.coverage(id),
    queryFn: ({ signal }) => api.coverage(id, signal),
  });
}

function Overview({
  incident,
  selected,
  onSelect,
}: {
  incident: Incident;
  selected: string;
  onSelect: (service: string) => void;
}) {
  const coverage = useCoverage(incident.id);
  const model = useQuery({ queryKey: queryKeys.model, queryFn: ({ signal }) => api.model(signal) });
  const top3 = model.data?.results.find((row) => row.name === TOP_3)?.estimate ?? null;
  return (
    <div className={styles.split}>
      <div className={styles.stack}>
        <DecisionCard
          incident={incident}
          telemetryCoverage={coverage.data?.overall ?? null}
          benchmarkTop3={top3}
        />
        <GraphCard id={incident.id} selected={selected} onSelect={onSelect} />
        <TimelineCard id={incident.id} />
        <EvidenceCards id={incident.id} service={selected} />
      </div>
      <div className={styles.stack}>
        <Suspects incident={incident} selected={selected} onSelect={onSelect} />
        <Card>
          <CardHeader title="Feedback" />
          <CardBody>
            <FeedbackForm key={incident.id} incident={incident} />
          </CardBody>
        </Card>
      </div>
    </div>
  );
}

function Evidence({
  incident,
  selected,
  onSelect,
}: {
  incident: Incident;
  selected: string;
  onSelect: (service: string) => void;
}) {
  const coverage = useCoverage(incident.id);
  return (
    <div className={styles.split}>
      <div className={styles.stack}>
        <EvidenceCards id={incident.id} service={selected} />
        <Card>
          <CardHeader title="Telemetry coverage" />
          <CardBody>
            <QueryView query={coverage} label="coverage">
              {(data) => <CoverageHeatmap coverage={data} />}
            </QueryView>
          </CardBody>
        </Card>
      </div>
      <Suspects incident={incident} selected={selected} onSelect={onSelect} />
    </div>
  );
}

function RawData({ incident }: { incident: Incident }) {
  return (
    <div className={styles.stack}>
      <Card>
        <CardHeader title="Notes" />
        <CardBody>
          <ul className={styles.notes}>
            {incident.diagnosis.notes.map((note) => (
              <li key={note}>{note}</li>
            ))}
          </ul>
          <p className={styles.meta}>
            Model {incident.diagnosis.model_version}, diagnosed {formatTime(incident.diagnosed_at)},
            confidence {formatUnit(incident.confidence)}.
          </p>
        </CardBody>
      </Card>
      <Card>
        <CardHeader title="Diagnosis" />
        <CardBody>
          {/* The block scrolls, so keyboard users must be able to focus it. */}
          {/* eslint-disable-next-line jsx-a11y/no-noninteractive-tabindex */}
          <pre className={styles.raw} tabIndex={0} aria-label="Diagnosis as JSON">
            {JSON.stringify(incident.diagnosis, null, 2)}
          </pre>
        </CardBody>
      </Card>
    </div>
  );
}

function Loaded({ incident }: { incident: Incident }) {
  const [tab, setTab] = useState("overview");
  const [chosen, setChosen] = useState<string | null>(null);
  const selected = chosen ?? incident.top_service;
  const shared = { incident, selected, onSelect: setChosen };
  return (
    <div className={styles.page}>
      <Header incident={incident} />
      <Tabs label="Incident views" items={TABS} selected={tab} onSelect={setTab} />
      {tab === "overview" && <Overview {...shared} />}
      {tab === "evidence" && <Evidence {...shared} />}
      {tab === "timeline" && <TimelineCard id={incident.id} />}
      {tab === "what-if" && (
        <Card>
          <CardHeader title="What-if" />
          <CardBody>
            <p>
              <Link to={`/incidents/${incident.id}/what-if`}>Open the what-if lab</Link> to simulate
              damage to a copy of this case. It never changes the incident.
            </p>
          </CardBody>
        </Card>
      )}
      {tab === "raw" && <RawData incident={incident} />}
    </div>
  );
}

export function IncidentDetail() {
  const { id = "" } = useParams();
  const incident = useQuery({
    queryKey: queryKeys.incident(id),
    queryFn: ({ signal }) => api.incident(id, signal),
  });
  return (
    <QueryView query={incident} label="the incident" rows={6}>
      {(data) => <Loaded key={data.id} incident={data} />}
    </QueryView>
  );
}
