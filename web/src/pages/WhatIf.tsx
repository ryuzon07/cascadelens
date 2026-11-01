import { useMutation, useQuery } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import {
  api,
  queryKeys,
  type IncidentDetail,
  type WhatIfRequest,
  type WhatIfResult,
} from "../api/client";
import { Button } from "../components/Button";
import { Card, CardBody, CardHeader } from "../components/Card";
import { CurveChart } from "../components/CurveChart";
import { DataTable, type Column } from "../components/DataTable";
import { Select, Slider, TextInput } from "../components/Fields";
import { QueryView } from "../components/QueryView";
import { EmptyState } from "../components/States";
import { Tag } from "../components/Tag";
import { formatUnit } from "../format";
import styles from "./pages.module.css";

type Scenario = WhatIfResult["original"];
type Change = WhatIfResult["rank_changes"][number];
type Kind = "telemetry" | "window" | "edges";

interface Control {
  label: string;
  perturbation: WhatIfRequest["perturbation"];
  parameter: string;
  min: number;
  max: number;
  step: number;
  initial: number;
  format: (value: number) => string;
}

const percent = (value: number) => `${Math.round(value * 100)}%`;
const seconds = (value: number) => `${value > 0 ? "+" : ""}${value} s`;

const CONTROLS: Record<Kind, Control> = {
  telemetry: {
    label: "Telemetry dropped",
    perturbation: "P1",
    parameter: "rate",
    min: 0.1,
    max: 0.9,
    step: 0.1,
    initial: 0.3,
    format: percent,
  },
  window: {
    label: "Window shift",
    perturbation: "P4",
    parameter: "offset_seconds",
    min: -120,
    max: 120,
    step: 30,
    initial: 60,
    format: seconds,
  },
  edges: {
    label: "Edges removed",
    perturbation: "P6",
    parameter: "rate",
    min: 0.1,
    max: 0.9,
    step: 0.1,
    initial: 0.3,
    format: percent,
  },
};
const PRESETS: { label: string; kind: Kind; value: number }[] = [
  { label: "Heavy telemetry loss", kind: "telemetry", value: 0.5 },
  { label: "Late window", kind: "window", value: 120 },
  { label: "Early window", kind: "window", value: -120 },
  { label: "Half the graph gone", kind: "edges", value: 0.5 },
];
const CHANGE_COLUMNS: Column<Change>[] = [
  {
    key: "service",
    header: "Service",
    render: (row) => <span className="mono">{row.service}</span>,
  },
  { key: "original", header: "Original", render: (row) => row.original_rank ?? "Not ranked" },
  { key: "scenario", header: "Scenario", render: (row) => row.scenario_rank ?? "Not ranked" },
  {
    key: "change",
    header: "Change",
    render: (row) => {
      if (row.change === null || row.change === undefined) return "-";
      if (row.change === 0) return "No change";
      return `${row.change > 0 ? "Up" : "Down"} ${Math.abs(row.change)}`;
    },
  },
];

function ScenarioCard({ title, scenario }: { title: string; scenario: Scenario }) {
  return (
    <Card>
      <CardHeader title={title} actions={<Tag value={scenario.decision} />} />
      <CardBody>
        <p>
          {scenario.decision === "abstained" ? "Leading lead" : "Top suspect"}:{" "}
          <span className="mono">{scenario.ranking[0]?.service ?? "None ranked"}</span>
        </p>
        <p className={styles.meta}>Confidence {formatUnit(scenario.confidence)}</p>
      </CardBody>
    </Card>
  );
}

function Result({ result }: { result: WhatIfResult }) {
  const control = Object.values(CONTROLS).find((item) => item.perturbation === result.perturbation);
  return (
    <>
      <div className={styles.tags}>
        <Tag value="Simulated" tone="warn" />
        <span className={styles.meta}>
          {result.name}, seed {result.seed}. The stored incident did not change.
        </span>
      </div>
      <div className={styles.pair}>
        <ScenarioCard title="Original" scenario={result.original} />
        <ScenarioCard title="Scenario" scenario={result.scenario} />
      </div>
      <Card>
        <CardHeader title="Rank changes" />
        <DataTable
          caption="Rank changes"
          columns={CHANGE_COLUMNS}
          rows={result.rank_changes}
          rowKey={(row) => row.service}
        />
      </Card>
      <Card>
        <CardHeader title="Confidence against intensity" />
        <CardBody>
          {result.curve.length === 0 ? (
            <EmptyState title="No intensity scale">This scenario has a single setting.</EmptyState>
          ) : (
            <CurveChart
              label="Confidence against intensity, simulated"
              xLabel={control?.label ?? "Intensity"}
              yLabel="confidence"
              threshold={result.abstain_threshold}
              formatX={control?.format}
              series={[
                {
                  label: "Confidence",
                  points: result.curve.map((point) => ({
                    x: point.intensity,
                    y: point.confidence,
                  })),
                },
              ]}
            />
          )}
        </CardBody>
      </Card>
    </>
  );
}

function Lab({ incident }: { incident: IncidentDetail }) {
  const [kind, setKind] = useState<Kind>("telemetry");
  const [value, setValue] = useState(CONTROLS.telemetry.initial);
  const [seed, setSeed] = useState(0);
  const controller = useRef<AbortController | null>(null);
  const control = CONTROLS[kind];
  const noGraph = kind === "edges" && incident.topology_source === "none";

  useEffect(() => () => controller.current?.abort(), []);

  const run = useMutation({
    mutationFn: () => {
      controller.current = new AbortController();
      return api.whatIf(
        incident.id,
        {
          perturbation: control.perturbation,
          params: { [control.parameter]: value },
          seed,
        },
        controller.current.signal,
      );
    },
  });
  const aborted = run.error?.name === "AbortError";

  function choose(next: Kind, amount: number) {
    setKind(next);
    setValue(amount);
  }

  return (
    <div className={styles.page}>
      <header>
        <h1 className={styles.title}>What-if lab</h1>
        <p className={styles.meta}>
          <Link to={`/incidents/${incident.id}`}>{incident.id}</Link>: {incident.title}. Each run
          damages a copy of the case with one simulated change.
        </p>
      </header>
      <Card>
        <CardHeader title="Scenario" />
        <CardBody>
          <div className={styles.page}>
            <div className={styles.actions}>
              {PRESETS.map((preset) => (
                <Button key={preset.label} onClick={() => choose(preset.kind, preset.value)}>
                  {preset.label}
                </Button>
              ))}
            </div>
            <div className={styles.pair}>
              <Select
                label="Change to simulate"
                value={kind}
                onChange={(event) => {
                  const next = event.target.value as Kind;
                  choose(next, CONTROLS[next].initial);
                }}
                options={Object.entries(CONTROLS).map(([id, item]) => ({
                  value: id,
                  label: item.label,
                }))}
              />
              <Slider
                label={control.label}
                value={value}
                min={control.min}
                max={control.max}
                step={control.step}
                onChange={setValue}
                format={control.format}
              />
              <TextInput
                label="Seed"
                type="number"
                min={0}
                value={seed}
                onChange={(event) => setSeed(Math.max(0, Number(event.target.value) || 0))}
              />
            </div>
            {noGraph && <p role="alert">This incident has no call graph to change.</p>}
            <div className={styles.actions}>
              <Button
                variant="primary"
                onClick={() => run.mutate()}
                disabled={run.isPending || noGraph}
              >
                Run
              </Button>
              {run.isPending && (
                <>
                  <Button onClick={() => controller.current?.abort()}>Cancel</Button>
                  <span role="status" className={styles.meta}>
                    Running scenario…
                  </span>
                </>
              )}
              {aborted && (
                <span role="status" className={styles.meta}>
                  Cancelled.
                </span>
              )}
            </div>
            {run.isError && !aborted && <p role="alert">{run.error.message}</p>}
          </div>
        </CardBody>
      </Card>
      {run.data && !run.isPending && <Result result={run.data} />}
    </div>
  );
}

export function WhatIf() {
  const { id = "" } = useParams();
  const incident = useQuery({
    queryKey: queryKeys.incident(id),
    queryFn: ({ signal }) => api.incident(id, signal),
  });
  return (
    <QueryView query={incident} label="the incident">
      {(data) => <Lab incident={data} />}
    </QueryView>
  );
}

const CHOOSER = { limit: 200 };

export function WhatIfChooser() {
  const incidents = useQuery({
    queryKey: queryKeys.incidents(CHOOSER),
    queryFn: ({ signal }) => api.incidents(CHOOSER, signal),
  });
  return (
    <Card>
      <CardHeader title="What-if lab" />
      <CardBody>
        <QueryView query={incidents} label="incidents">
          {(data) =>
            data.items.length === 0 ? (
              <EmptyState title="No incidents yet">A scenario starts from an incident.</EmptyState>
            ) : (
              <>
                <h1 className={styles.meta}>Choose an incident to simulate on.</h1>
                <ul className={styles.labelList}>
                  {data.items.map((item) => (
                    <li key={item.id}>
                      <Link to={`/incidents/${item.id}/what-if`}>
                        {item.id}: {item.title}
                      </Link>
                    </li>
                  ))}
                </ul>
              </>
            )
          }
        </QueryView>
      </CardBody>
    </Card>
  );
}
