import { useState, type ReactNode } from "react";
import { ApiError } from "../api/client";
import { Button } from "../components/Button";
import { Card, CardBody, CardHeader, KpiTile } from "../components/Card";
import { ConfidenceGauge } from "../components/ConfidenceGauge";
import { ContributionBars } from "../components/ContributionBars";
import { CoverageHeatmap } from "../components/CoverageHeatmap";
import { DataTable } from "../components/DataTable";
import { DecisionCard } from "../components/DecisionCard";
import { DependencyGraph, GraphLegend } from "../components/DependencyGraph";
import { Disclaimer, SampleDataBadge } from "../components/Disclaimer";
import { FeedbackForm } from "../components/FeedbackForm";
import { Select, Slider, TextInput, Toggle } from "../components/Fields";
import { CurveChart } from "../components/CurveChart";
import { LineChart } from "../components/LineChart";
import { OnsetTimeline } from "../components/OnsetTimeline";
import { ReliabilityDiagram } from "../components/ReliabilityDiagram";
import { StackedBars } from "../components/StackedBars";
import { EmptyState, ErrorState, Skeleton, StaleIndicator } from "../components/States";
import { SuspectRow } from "../components/SuspectRow";
import { Tabs } from "../components/Tabs";
import { Tag } from "../components/Tag";
import {
  coverage,
  graph,
  incidentDetail,
  incidents,
  serviceEvidence,
  timeline,
} from "../mocks/fixtures";
import styles from "./Components.module.css";

const TAGS = [
  "critical",
  "high",
  "medium",
  "low",
  "open",
  "needs_feedback",
  "closed",
  "confident",
  "abstained",
  "uncalibrated",
  "top_1",
  "top_3",
  "missed",
  "unknown",
];
const CONFIDENT = incidentDetail(incidents[0]!);
const ABSTAINED = incidentDetail(incidents[1]!);
const SERIES = serviceEvidence("alpha").series[0]!;
const ROWS = [
  { name: "alpha", count: 3 },
  { name: "beta", count: 1 },
  { name: "gamma", count: 2 },
];

function Sample({ title, children }: { title: string; children: ReactNode }) {
  return (
    <Card>
      <CardHeader title={title} />
      <CardBody>
        <div className={styles.row}>{children}</div>
      </CardBody>
    </Card>
  );
}

/** Every component in every state, with placeholder content. Hidden in production builds. */
export function Components() {
  const [tab, setTab] = useState("one");
  const [level, setLevel] = useState(30);
  const [enabled, setEnabled] = useState(true);
  return (
    <div className={styles.sheet}>
      <h1>Components</h1>
      <Sample title="Buttons">
        <Button variant="primary">Primary</Button>
        <Button>Secondary</Button>
        <Button variant="ghost">Ghost</Button>
        <Button variant="icon" aria-label="Icon button">
          +
        </Button>
        <Button disabled>Disabled</Button>
      </Sample>
      <Sample title="Tags">
        {TAGS.map((value) => (
          <Tag key={value} value={value} />
        ))}
        <SampleDataBadge />
      </Sample>
      <Sample title="KPI tiles">
        <KpiTile label="With a value" value="12" hint="Hint text" />
        <KpiTile label="Pending" value={null} />
      </Sample>
      <Sample title="Tabs">
        <Tabs
          label="Example tabs"
          selected={tab}
          onSelect={setTab}
          items={[
            { id: "one", label: "First" },
            { id: "two", label: "Second" },
            { id: "three", label: "Third" },
          ]}
        />
      </Sample>
      <Sample title="Controls">
        <TextInput label="Text input" placeholder="Placeholder" />
        <Select
          label="Select"
          options={[
            { value: "a", label: "Option A" },
            { value: "b", label: "Option B" },
          ]}
        />
        <Slider
          label="Slider"
          value={level}
          min={0}
          max={100}
          step={10}
          onChange={setLevel}
          format={(value) => `${value}%`}
        />
        <Toggle label="Toggle" checked={enabled} onChange={setEnabled} />
        <Toggle label="Disabled toggle" checked={false} onChange={() => {}} disabled />
      </Sample>
      <Sample title="Data table">
        <DataTable
          caption="Example table"
          rows={ROWS}
          rowKey={(row) => row.name}
          columns={[
            {
              key: "name",
              header: "Name",
              render: (row) => row.name,
              sortValue: (row) => row.name,
            },
            {
              key: "count",
              header: "Count",
              render: (row) => row.count,
              sortValue: (row) => row.count,
            },
          ]}
        />
      </Sample>
      <Sample title="States">
        <EmptyState title="Empty state">Nothing to show.</EmptyState>
        <ErrorState error={new ApiError(0, "network_error", "")} onRetry={() => {}} />
        <Skeleton label="Loading example" />
        <StaleIndicator active />
      </Sample>
      <Sample title="Confidence gauge">
        <ConfidenceGauge confidence={0.9} threshold={0.5} />
        <ConfidenceGauge confidence={0.3} threshold={0.5} />
        <ConfidenceGauge confidence={0.9} threshold={null} />
      </Sample>
      <Sample title="Decision cards">
        <DecisionCard incident={CONFIDENT} telemetryCoverage={1} benchmarkTop3={null} />
        <DecisionCard incident={ABSTAINED} telemetryCoverage={0.8} benchmarkTop3={null} />
      </Sample>
      <Sample title="Suspects">
        <SuspectRow item={CONFIDENT.diagnosis.ranking[0]!} rank={1} selected onSelect={() => {}} />
        <SuspectRow
          item={CONFIDENT.diagnosis.ranking[1]!}
          rank={2}
          selected={false}
          onSelect={() => {}}
        />
      </Sample>
      <Sample title="Charts">
        <ContributionBars
          label="Example contributions"
          items={[
            { label: "Up", contribution: 2 },
            { label: "Down", contribution: -1 },
          ]}
        />
        <LineChart series={SERIES} />
        <StackedBars
          label="Example shares"
          segments={[
            { label: "First", value: 3, color: "var(--ink)" },
            { label: "Second", value: 1, color: "var(--warn)" },
          ]}
        />
        <CoverageHeatmap coverage={coverage} />
        <OnsetTimeline timeline={timeline} />
        <CurveChart
          label="Example curve"
          xLabel="Intensity"
          yLabel="Confidence"
          threshold={0.5}
          series={[
            {
              label: "Example",
              points: [
                { x: 0, y: 0.9 },
                { x: 1, y: 0.4 },
              ],
            },
          ]}
        />
        <ReliabilityDiagram bins={null} />
        <ReliabilityDiagram
          bins={[
            { confidence: 0.3, accuracy: 0.25, cases: 10 },
            { confidence: 0.8, accuracy: 0.85, cases: 20 },
          ]}
        />
      </Sample>
      <Sample title="Dependency graph">
        <DependencyGraph graph={graph} selected="gamma" />
        <GraphLegend />
      </Sample>
      <Sample title="Feedback form">
        <FeedbackForm incident={CONFIDENT} />
      </Sample>
      <Sample title="Disclaimer">
        <Disclaimer />
      </Sample>
    </div>
  );
}
