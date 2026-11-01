import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { Link } from "react-router-dom";
import { api, queryKeys, type IncidentSummary } from "../api/client";
import { Card, CardBody, CardHeader } from "../components/Card";
import { DependencyGraph, GraphLegend } from "../components/DependencyGraph";
import { Select } from "../components/Fields";
import { QueryView } from "../components/QueryView";
import { EmptyState } from "../components/States";
import styles from "./pages.module.css";

const QUERY = { limit: 200 };

function Map({ incidents }: { incidents: IncidentSummary[] }) {
  const [chosen, setChosen] = useState("");
  const id = chosen || incidents[0]?.id || "";
  const graph = useQuery({
    queryKey: queryKeys.graph(id),
    queryFn: ({ signal }) => api.graph(id, signal),
    enabled: Boolean(id),
  });
  if (!id) return <EmptyState title="No incidents yet">The map draws from an incident.</EmptyState>;
  return (
    <>
      <div className={styles.filters}>
        <Select
          label="Incident"
          value={id}
          onChange={(event) => setChosen(event.target.value)}
          options={incidents.map((item) => ({
            value: item.id,
            label: `${item.id}: ${item.title}`,
          }))}
        />
        <Link to={`/incidents/${id}`}>Open incident</Link>
      </div>
      <CardBody>
        <QueryView query={graph} label="the graph">
          {(data) => (
            <>
              <DependencyGraph graph={data} />
              <GraphLegend />
            </>
          )}
        </QueryView>
      </CardBody>
    </>
  );
}

/** One incident's call graph. There is no live topology, because nothing connects to systems. */
export function ServiceMap() {
  const incidents = useQuery({
    queryKey: queryKeys.incidents(QUERY),
    queryFn: ({ signal }) => api.incidents(QUERY, signal),
  });
  return (
    <Card>
      <CardHeader title="Service map" />
      <QueryView query={incidents} label="incidents">
        {(data) => <Map incidents={data.items} />}
      </QueryView>
    </Card>
  );
}
