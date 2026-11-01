import cytoscape, { type ElementDefinition, type StylesheetJson } from "cytoscape";
import dagre from "cytoscape-dagre";
import { useEffect, useRef } from "react";
import type { IncidentGraph } from "../api/client";
import styles from "./charts.module.css";

const SIGNAL = "#C8381A";
const SIGNAL_BG = "#F7E1D8";
const SIGNAL_2 = "#D9805F";
const WARN = "#8A5A00";
const RULE = "#DAD6CB";
const RULE_STRONG = "#C4BEAF";
const INK = "#1B1B18";
const MUTED = "#57544B";
const LINK = "#1E4F9C";
const FONT = "IBM Plex Mono, monospace";

try {
  cytoscape.use(dagre);
} catch {
  // Registering twice throws, for example on a hot reload. The layout falls back below.
}

const STYLE: StylesheetJson = [
  {
    selector: "node",
    style: {
      shape: "rectangle",
      width: "label",
      height: 30,
      padding: "8px",
      label: "data(label)",
      "text-valign": "center",
      "text-halign": "center",
      "text-wrap": "wrap",
      "font-family": FONT,
      "font-size": 12,
      color: INK,
      "background-color": "#FFFFFF",
      "border-width": 1,
      "border-color": RULE,
    },
  },
  {
    selector: 'node[state = "top_suspect"]',
    style: { "background-color": SIGNAL, "border-color": SIGNAL, color: "#FFFFFF" },
  },
  {
    selector: 'node[state = "suspect"]',
    style: { "background-color": SIGNAL_BG, "border-color": SIGNAL_2 },
  },
  { selector: 'node[state = "anomalous"]', style: { "border-color": WARN, "border-width": 2 } },
  {
    selector: 'node[state = "no_telemetry"]',
    style: { "border-style": "dashed", "border-color": MUTED, color: MUTED },
  },
  { selector: "node.selected", style: { "outline-width": 2, "outline-color": LINK } },
  {
    selector: "edge",
    style: {
      width: 1,
      "line-color": RULE_STRONG,
      "target-arrow-color": RULE_STRONG,
      "target-arrow-shape": "triangle",
      "curve-style": "bezier",
    },
  },
  { selector: "edge[?inferred]", style: { "line-style": "dashed" } },
  {
    selector: "edge[?anomalous]",
    style: { width: 2.5, "line-color": SIGNAL, "target-arrow-color": SIGNAL },
  },
];

export function graphElements(graph: IncidentGraph): ElementDefinition[] {
  return [
    ...graph.nodes.map((node) => ({
      data: {
        id: node.service,
        state: node.state,
        label: node.has_telemetry ? node.service : `${node.service}\nno data`,
      },
    })),
    ...graph.edges.map((edge) => ({
      data: {
        id: `${edge.caller}->${edge.callee}`,
        source: edge.caller,
        target: edge.callee,
        inferred: edge.inferred,
        anomalous: edge.on_anomalous_path,
      },
    })),
  ];
}

function summary(graph: IncidentGraph): string {
  const anomalous = graph.edges.filter((edge) => edge.on_anomalous_path).length;
  return `Dependency graph: ${graph.nodes.length} services, ${graph.edges.length} calls, ${anomalous} on the anomalous path. Use the ranked list to select a service with the keyboard.`;
}

interface DependencyGraphProps {
  graph: IncidentGraph;
  selected?: string;
  onSelect?: (service: string) => void;
}

export function DependencyGraph({ graph, selected, onSelect }: DependencyGraphProps) {
  const container = useRef<HTMLDivElement>(null);
  const instance = useRef<cytoscape.Core | null>(null);
  const select = useRef(onSelect);
  useEffect(() => {
    select.current = onSelect;
  }, [onSelect]);

  useEffect(() => {
    if (!container.current) return;
    const cy = cytoscape({
      container: container.current,
      elements: graphElements(graph),
      style: STYLE,
      userPanningEnabled: false,
      userZoomingEnabled: false,
      boxSelectionEnabled: false,
    });
    const options = { name: "dagre", rankDir: "TB", nodeSep: 40, rankSep: 50 };
    try {
      cy.layout(options as cytoscape.LayoutOptions).run();
    } catch {
      cy.layout({ name: "breadthfirst", directed: true, spacingFactor: 1.2 }).run();
    }
    cy.fit(undefined, 24);
    cy.on("tap", "node", (event) => select.current?.(event.target.id() as string));
    instance.current = cy;
    return () => {
      cy.destroy();
      instance.current = null;
    };
  }, [graph]);

  useEffect(() => {
    const cy = instance.current;
    if (!cy) return;
    cy.nodes().removeClass("selected");
    if (selected) cy.getElementById(selected).addClass("selected");
  }, [selected, graph]);

  return (
    <div className={styles.graphPanel}>
      <div ref={container} className={styles.graph} role="img" aria-label={summary(graph)} />
    </div>
  );
}

const NODES = [
  { label: "Top suspect", style: { background: SIGNAL, borderColor: SIGNAL } },
  { label: "Suspect 2-3", style: { background: SIGNAL_BG, borderColor: SIGNAL_2 } },
  { label: "Anomalous", style: { borderColor: WARN, borderWidth: 2 } },
  { label: "Healthy", style: {} },
  { label: "No telemetry", style: { borderStyle: "dashed", borderColor: MUTED } },
];

export function GraphLegend() {
  return (
    <ul className={styles.legend} aria-label="Graph legend">
      {NODES.map((node) => (
        <li key={node.label}>
          <span className={styles.legendNode} style={node.style} />
          {node.label}
        </li>
      ))}
      <li>
        <span className={styles.legendEdge} style={{ borderColor: SIGNAL }} />
        Anomalous call path
      </li>
      <li>
        <span className={styles.legendEdge} style={{ borderTopStyle: "dashed" }} />
        Inferred call
      </li>
    </ul>
  );
}
