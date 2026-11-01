import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { describe, expect, it, vi } from "vitest";
import { App } from "../App";
import { graphElements } from "../components/DependencyGraph";
import * as fixtures from "../mocks/fixtures";
import { renderApp } from "../test/render";
import { server } from "../test/setup";

const taps: ((event: { target: { id: () => string } }) => void)[] = [];

vi.mock("cytoscape", () => {
  const collection = { removeClass: vi.fn(), addClass: vi.fn() };
  const core = vi.fn(() => ({
    layout: () => ({ run: vi.fn() }),
    fit: vi.fn(),
    on: (_: string, __: string, handler: (typeof taps)[number]) => taps.push(handler),
    destroy: vi.fn(),
    nodes: () => collection,
    getElementById: () => collection,
  }));
  return { default: Object.assign(core, { use: vi.fn() }) };
});

describe("incidents overview", () => {
  it("lists incidents with their decisions", async () => {
    renderApp(<App />, "/incidents");
    const table = await screen.findByRole("table", { name: "Incidents" });
    expect(within(table).getByRole("link", { name: "INC-0002" })).toHaveAttribute(
      "href",
      "/incidents/INC-0002",
    );
    expect(within(table).getByText("Abstained")).toBeInTheDocument();
    expect(within(table).getByText("Confident")).toBeInTheDocument();
  });

  it("shows pending instead of inventing a KPI", async () => {
    renderApp(<App />, "/incidents");
    await screen.findByRole("table", { name: "Incidents" });
    const pending = await screen.findAllByText("[pending]");
    expect(pending.length).toBeGreaterThanOrEqual(2);
    expect(screen.getByText("No labels yet")).toBeInTheDocument();
  });

  it("derives KPIs from the incidents", async () => {
    renderApp(<App />, "/incidents");
    expect(await screen.findByText("50.0%")).toBeInTheDocument();
    expect(screen.getByRole("img", { name: /Decisions/ })).toHaveAccessibleName(
      "Decisions: Confident 1, Abstained 1, Uncalibrated 0",
    );
  });

  it("lists what is waiting for a label", async () => {
    renderApp(<App />, "/incidents");
    const card = (await screen.findByRole("heading", { name: "Waiting for your label" })).closest(
      "section",
    )!;
    expect(await within(card).findByRole("link")).toHaveTextContent("Sample gap in telemetry");
  });

  it("sends filters to the API", async () => {
    const seen: string[] = [];
    server.use(
      http.get("*/api/v1/incidents", ({ request }) => {
        seen.push(new URL(request.url).search);
        return HttpResponse.json(fixtures.incidentPage);
      }),
    );
    renderApp(<App />, "/incidents");
    await screen.findByRole("table", { name: "Incidents" });
    await userEvent.selectOptions(screen.getByLabelText("Severity"), "high");
    await userEvent.click(screen.getByRole("tab", { name: "Needs feedback" }));
    await waitFor(() =>
      expect(seen.some((text) => text.includes("status=needs_feedback"))).toBe(true),
    );
    expect(seen.some((text) => text.includes("severity=high"))).toBe(true);
  });

  it("filters by the search box", async () => {
    renderApp(<App />, "/incidents");
    await screen.findByRole("table", { name: "Incidents" });
    await userEvent.type(screen.getByRole("searchbox"), "telemetry{Enter}");
    await waitFor(() =>
      expect(screen.queryByRole("link", { name: "INC-0002" })).not.toBeInTheDocument(),
    );
    expect(screen.getByRole("link", { name: "INC-0001" })).toBeInTheDocument();
  });

  it("shows an empty state", async () => {
    server.use(
      http.get("*/api/v1/incidents", () =>
        HttpResponse.json({ ...fixtures.incidentPage, items: [] }),
      ),
    );
    renderApp(<App />, "/incidents");
    expect(await screen.findByText("No incidents match")).toBeInTheDocument();
  });

  it("shows an error with a retry", async () => {
    server.use(http.get("*/api/v1/incidents", () => HttpResponse.error()));
    renderApp(<App />, "/incidents");
    expect(await screen.findByRole("alert")).toHaveTextContent("could not be reached");
    server.resetHandlers();
    await userEvent.click(screen.getByRole("button", { name: "Retry" }));
    expect(await screen.findByRole("table", { name: "Incidents" })).toBeInTheDocument();
  });

  it("loads further pages", async () => {
    server.use(
      http.get("*/api/v1/incidents", ({ request }) =>
        new URL(request.url).searchParams.get("cursor")
          ? HttpResponse.json({
              ...fixtures.incidentPage,
              items: [{ ...fixtures.incidents[0]!, id: "INC-0000" }],
            })
          : HttpResponse.json({ ...fixtures.incidentPage, next_cursor: "1" }),
      ),
    );
    renderApp(<App />, "/incidents");
    await userEvent.click(await screen.findByRole("button", { name: "Load more" }));
    expect(await screen.findByRole("link", { name: "INC-0000" })).toBeInTheDocument();
  });
});

describe("incident detail", () => {
  it("shows a confident incident", async () => {
    renderApp(<App />, "/incidents/INC-0002");
    expect(await screen.findByRole("heading", { name: "Sample slowdown" })).toBeInTheDocument();
    expect(screen.getByRole("region", { name: "Decision" })).toHaveTextContent(
      "Most likely origin",
    );
    expect(screen.getByRole("heading", { name: "Ranked suspects" })).toBeInTheDocument();
    expect(await screen.findByRole("img", { name: /Dependency graph/ })).toBeInTheDocument();
  });

  it("shows an abstained incident as leads", async () => {
    renderApp(<App />, "/incidents/INC-0001");
    expect(await screen.findByRole("heading", { name: "Leads to check" })).toBeInTheDocument();
    expect(screen.getByRole("region", { name: "Decision" })).toHaveTextContent(
      "No confident answer",
    );
  });

  it("selects a suspect from the list", async () => {
    let asked = "";
    server.use(
      http.get("*/api/v1/incidents/:id/evidence/:service", ({ params }) => {
        asked = String(params.service);
        return HttpResponse.json(fixtures.serviceEvidence(asked));
      }),
    );
    renderApp(<App />, "/incidents/INC-0002");
    await userEvent.click(await screen.findByRole("button", { name: /gamma/ }));
    expect(await screen.findByText("Feature contributions: gamma")).toBeInTheDocument();
    await waitFor(() => expect(asked).toBe("gamma"));
  });

  it("selects a suspect from the graph", async () => {
    taps.length = 0;
    renderApp(<App />, "/incidents/INC-0002");
    await screen.findByRole("img", { name: /Dependency graph/ });
    await waitFor(() => expect(taps.length).toBeGreaterThan(0));
    taps.at(-1)!({ target: { id: () => "gamma" } });
    expect(await screen.findByText("Metrics: gamma")).toBeInTheDocument();
  });

  it("re-runs the diagnosis", async () => {
    let calls = 0;
    server.use(
      http.post("*/api/v1/incidents/:id/rerun", () => {
        calls += 1;
        return HttpResponse.json(fixtures.incidentDetail(fixtures.incidents[0]!));
      }),
    );
    renderApp(<App />, "/incidents/INC-0002");
    await userEvent.click(await screen.findByRole("button", { name: "Re-run diagnosis" }));
    await waitFor(() => expect(calls).toBe(1));
  });

  it("copies a share link", async () => {
    renderApp(<App />, "/incidents/INC-0002");
    await userEvent.click(await screen.findByRole("button", { name: "Share link" }));
    expect(await screen.findByText(/Link copied|Copy this page/)).toBeInTheDocument();
  });

  it("opens every tab", async () => {
    renderApp(<App />, "/incidents/INC-0002");
    await userEvent.click(await screen.findByRole("tab", { name: "Evidence" }));
    expect(await screen.findByRole("region", { name: "Telemetry coverage" })).toBeInTheDocument();
    await userEvent.click(screen.getByRole("tab", { name: "Timeline" }));
    expect(await screen.findByRole("heading", { name: "Onset timeline" })).toBeInTheDocument();
    await userEvent.click(screen.getByRole("tab", { name: "What-if" }));
    expect(screen.getByRole("link", { name: "Open the what-if lab" })).toHaveAttribute(
      "href",
      "/incidents/INC-0002/what-if",
    );
    await userEvent.click(screen.getByRole("tab", { name: "Raw data" }));
    expect(screen.getByLabelText("Diagnosis as JSON")).toHaveTextContent("confidence");
  });

  it("explains a graph with no edges", async () => {
    server.use(
      http.get("*/api/v1/incidents/:id/graph", () =>
        HttpResponse.json({ ...fixtures.graph, edges: [], topology_source: "none" }),
      ),
    );
    renderApp(<App />, "/incidents/INC-0002");
    expect(await screen.findByText("No call graph")).toBeInTheDocument();
  });

  it("shows a missing incident as an error", async () => {
    renderApp(<App />, "/incidents/INC-9999");
    expect(await screen.findByRole("alert")).toHaveTextContent("does not exist");
  });
});

describe("graph data", () => {
  it("styles by state, marks no telemetry and carries edge flags", () => {
    const elements = graphElements({
      nodes: [{ service: "a", state: "no_telemetry", score: null, has_telemetry: false }],
      edges: [{ caller: "a", callee: "b", inferred: true, on_anomalous_path: true }],
      topology_source: "inferred",
    });
    expect(elements[0]?.data).toMatchObject({ state: "no_telemetry", label: "a\nno data" });
    expect(elements[1]?.data).toMatchObject({ source: "a", inferred: true, anomalous: true });
  });
});

describe("service map", () => {
  it("draws a chosen incident", async () => {
    const asked: string[] = [];
    server.use(
      http.get("*/api/v1/incidents/:id/graph", ({ params }) => {
        asked.push(String(params.id));
        return HttpResponse.json(fixtures.graph);
      }),
    );
    renderApp(<App />, "/service-map");
    await userEvent.selectOptions(await screen.findByLabelText("Incident"), "INC-0001");
    await waitFor(() => expect(asked).toContain("INC-0001"));
  });

  it("says when there are no incidents", async () => {
    server.use(
      http.get("*/api/v1/incidents", () =>
        HttpResponse.json({ ...fixtures.incidentPage, items: [] }),
      ),
    );
    renderApp(<App />, "/service-map");
    expect(await screen.findByText("No incidents yet")).toBeInTheDocument();
  });
});
