import type {
  FeedbackLabel,
  HistoryReport,
  IncidentCoverage,
  IncidentDetail,
  IncidentGraph,
  IncidentPage,
  IncidentSummary,
  IncidentTimeline,
  IntegrationList,
  ModelReport,
  ServiceEvidence,
  SettingsView,
  WhatIfResult,
} from "../api/client";

/** Synthetic data for developing without the service. Model results stay null on purpose. */
export const DISCLAIMER = "ranking of likely origin services, not proof of causation";
const START = 1_700_000_000;
const END = START + 445;

function summary(id: string, overrides: Partial<IncidentSummary>): IncidentSummary {
  return {
    id,
    title: "Sample incident",
    severity: "medium",
    status: "open",
    owner: "on-call",
    is_demo: true,
    window_start: START,
    window_end: END,
    created_at: START + 600,
    top_service: "alpha",
    top_score: 1,
    confidence: 0.9,
    decision: "confident",
    disclaimer: DISCLAIMER,
    ...overrides,
  };
}

export const incidents: IncidentSummary[] = [
  summary("INC-0002", { title: "Sample slowdown", severity: "high" }),
  summary("INC-0001", {
    title: "Sample gap in telemetry",
    status: "needs_feedback",
    confidence: 0.3,
    decision: "abstained",
    top_service: "beta",
  }),
];

export const incidentPage: IncidentPage = {
  items: incidents,
  next_cursor: null,
  disclaimer: DISCLAIMER,
};

function evidence(service: string, score: number) {
  return {
    service,
    score,
    evidence: {
      onset_rank: 1,
      upstream_anom_ratio: null,
      missing_frac: 0,
      top_groups: [{ group: "signal", contribution: score }],
    },
  };
}

export function incidentDetail(item: IncidentSummary): IncidentDetail {
  return {
    ...item,
    service_count: 3,
    sample_count: 1080,
    topology_source: "provided",
    diagnosed_at: item.created_at,
    abstain_threshold: 0.5,
    diagnosis: {
      case_id: "sample",
      ranking: [evidence(item.top_service, 1), evidence("gamma", -2)],
      confidence: item.confidence,
      calibrated: true,
      abstain: item.decision === "abstained",
      notes: ["telemetry missing for 0 of 3 services", "topology provided with the case"],
      model_version: "sample",
      disclaimer: DISCLAIMER,
      timing: { total_seconds: 0.1 },
    },
  };
}

export const graph: IncidentGraph = {
  nodes: [
    { service: "alpha", state: "top_suspect", score: 1, has_telemetry: true },
    { service: "beta", state: "healthy", score: null, has_telemetry: true },
    { service: "gamma", state: "suspect", score: -2, has_telemetry: true },
  ],
  edges: [
    { caller: "beta", callee: "gamma", inferred: false, on_anomalous_path: false },
    { caller: "gamma", callee: "alpha", inferred: false, on_anomalous_path: true },
  ],
  topology_source: "provided",
  disclaimer: DISCLAIMER,
};

export const timeline: IncidentTimeline = {
  window_start: START,
  window_end: END,
  baseline_end: START + 111,
  services: [
    {
      service: "alpha",
      onset: START + 245,
      anomalous_until: END,
      telemetry_stopped: null,
      has_telemetry: true,
    },
  ],
  disclaimer: DISCLAIMER,
};

export const coverage: IncidentCoverage = {
  services: ["alpha"],
  families: ["cpu", "latency"],
  cells: [
    { service: "alpha", family: "cpu", coverage: 1, samples: 90 },
    { service: "alpha", family: "latency", coverage: 0.5, samples: 45 },
  ],
  overall: 0.75,
  disclaimer: DISCLAIMER,
};

export function serviceEvidence(service: string): ServiceEvidence {
  return {
    service,
    rank: 1,
    score: 1,
    groups: [{ group: "signal", contribution: 1 }],
    features: [
      {
        feature: "signal_sample",
        label: "Sample feature",
        group: "signal",
        value: 1,
        contribution: 1,
      },
    ],
    series: [
      {
        metric: "cpu",
        family: "cpu",
        points: [
          { ts: START, value: 1 },
          { ts: END, value: 2 },
        ],
      },
    ],
    disclaimer: DISCLAIMER,
  };
}

export const whatIf: WhatIfResult = {
  perturbation: "P1",
  name: "random_metric_loss",
  realism: "simulated",
  parameters: { rate: 0.3, level: "series" },
  seed: 0,
  original: {
    ranking: [{ service: "alpha", rank: 1, score: 1 }],
    confidence: 0.9,
    decision: "confident",
  },
  scenario: {
    ranking: [{ service: "alpha", rank: 1, score: 0.5 }],
    confidence: 0.6,
    decision: "confident",
  },
  rank_changes: [{ service: "alpha", original_rank: 1, scenario_rank: 1, change: 0 }],
  intensity_parameter: "rate",
  curve: [
    { intensity: 0, confidence: 0.9, top_service: "alpha", decision: "confident" },
    { intensity: 0.3, confidence: 0.6, top_service: "alpha", decision: "confident" },
  ],
  abstain_threshold: 0.5,
  disclaimer: DISCLAIMER,
};

export const label: FeedbackLabel = {
  incident_id: "INC-0002",
  title: "Sample slowdown",
  correct: true,
  predicted_service: "alpha",
  actual_service: "alpha",
  outcome: "top_1",
  confidence: 0.9,
  decision: "confident",
  note: "",
  labelled_by: "local user",
  created_at: START + 900,
  is_demo: true,
  disclaimer: DISCLAIMER,
};

export const history: HistoryReport = {
  items: [],
  weeks: [],
  top_1: null,
  top_3: null,
  includes_demo: false,
  disclaimer: DISCLAIMER,
};

export const model: ModelReport = {
  model_version: "sample",
  method: "two_stage_ranker",
  split: "group_run",
  seed: 0,
  fold: 0,
  calibrated: true,
  abstain_threshold: 0.5,
  training_data_checksum: "sample",
  feature_config_hash: "sample",
  config_hash: "sample",
  frozen_config_hash: null,
  result_status: null,
  results: [
    {
      name: "Ranker Top-1 (clean)",
      estimate: null,
      lower: null,
      upper: null,
      source: "experiments/robustness/results.json",
    },
  ],
  checks: [{ name: "Identity probe", passed: null, source: "experiments/probes/identity.json" }],
  disclaimer: DISCLAIMER,
};

export const settings: SettingsView = {
  abstain_threshold: null,
  window_seconds: null,
  use_provided_topology: true,
  model_threshold: 0.5,
  effective_threshold: 0.5,
};

export const integrations: IntegrationList = {
  items: [{ name: "Prometheus", kind: "metrics", status: "configuration only", configured: false }],
  read_only:
    "CascadeLens is read-only: these records are configuration only and connect to nothing.",
};
