import { useState, type ReactNode } from "react";
import { ApiError } from "../api/client";
import { Button } from "../components/Button";
import { Card, CardBody, CardHeader, KpiTile } from "../components/Card";
import { DataTable } from "../components/DataTable";
import { Disclaimer, SampleDataBadge } from "../components/Disclaimer";
import { Select, Slider, TextInput, Toggle } from "../components/Fields";
import { EmptyState, ErrorState, Skeleton, StaleIndicator } from "../components/States";
import { Tabs } from "../components/Tabs";
import { Tag } from "../components/Tag";
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
      <Sample title="Disclaimer">
        <Disclaimer />
      </Sample>
    </div>
  );
}
