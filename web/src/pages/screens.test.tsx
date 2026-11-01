import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { describe, expect, it, vi } from "vitest";
import { App } from "../App";
import type { HistoryReport, ModelReport } from "../api/client";
import { CurveChart } from "../components/CurveChart";
import { ReliabilityDiagram } from "../components/ReliabilityDiagram";
import { labelsToCsv } from "../csv";
import * as fixtures from "../mocks/fixtures";
import { renderApp } from "../test/render";
import { server } from "../test/setup";

vi.mock("cytoscape", () => ({ default: Object.assign(vi.fn(), { use: vi.fn() }) }));

const LAB = "/incidents/INC-0002/what-if";
const report: HistoryReport = {
  items: [fixtures.label],
  weeks: [{ week_start: 345_600, top_1: 1, top_3: 0, missed: 0, unknown: 0 }],
  top_1: { estimate: 1, lower: 0.5, upper: 1, labels: 1 },
  top_3: { estimate: 1, lower: 0.5, upper: 1, labels: 1 },
  includes_demo: true,
  disclaimer: fixtures.DISCLAIMER,
};
const fullModel: ModelReport = {
  ...fixtures.model,
  result_status: "exploratory",
  reliability: [{ confidence: 0.5, accuracy: 0.5, cases: 10 }],
  results: [
    { name: "Ranker Top-1 (clean)", estimate: 0.9, lower: 0.8, upper: 0.95, source: "a.json" },
    { name: "ECE after calibration", estimate: 0.03, lower: 0.02, upper: 0.05, source: "b.json" },
  ],
  checks: [
    { name: "Identity probe", passed: true, source: "a.json" },
    { name: "Mask-only probe", passed: false, source: "b.json" },
  ],
  risk_coverage: [
    { coverage: 0.5, estimate: 0.99, lower: 0.98, upper: 1 },
    { coverage: 1, estimate: 0.9, lower: 0.85, upper: 0.95 },
  ],
  robustness: [
    {
      method: "ranker",
      perturbation: "clean",
      point: "clean",
      realism: "clean",
      estimate: 0.9,
      lower: 0.85,
      upper: 0.95,
    },
    {
      method: "ranker",
      perturbation: "P2",
      point: "P2:rate=0.3",
      realism: "simulated",
      estimate: 0.7,
      lower: 0.6,
      upper: 0.8,
    },
  ],
};

describe("what-if lab", () => {
  it("runs a scenario and labels it simulated", async () => {
    let body: unknown;
    server.use(
      http.post("*/api/v1/incidents/:id/whatif", async ({ request }) => {
        body = await request.json();
        return HttpResponse.json(fixtures.whatIf);
      }),
    );
    renderApp(<App />, LAB);
    await userEvent.click(await screen.findByRole("button", { name: "Run" }));
    expect(await screen.findByText("Simulated")).toBeInTheDocument();
    expect(body).toEqual({ perturbation: "P1", params: { rate: 0.3 }, seed: 0 });
    expect(screen.getByText(/stored incident did not change/)).toBeInTheDocument();
    expect(screen.getByRole("table", { name: "Rank changes" })).toHaveTextContent("No change");
    expect(screen.getByRole("img", { name: /Confidence against intensity/ })).toBeInTheDocument();
  });

  it("applies a preset", async () => {
    let body: unknown;
    server.use(
      http.post("*/api/v1/incidents/:id/whatif", async ({ request }) => {
        body = await request.json();
        return HttpResponse.json(fixtures.whatIf);
      }),
    );
    renderApp(<App />, LAB);
    await userEvent.click(await screen.findByRole("button", { name: "Late window" }));
    expect(screen.getByRole("slider")).toHaveAttribute("aria-valuetext", "+120 s");
    await userEvent.click(screen.getByRole("button", { name: "Run" }));
    await waitFor(() =>
      expect(body).toEqual({ perturbation: "P4", params: { offset_seconds: 120 }, seed: 0 }),
    );
  });

  it("switches the slider with the change", async () => {
    renderApp(<App />, LAB);
    await userEvent.selectOptions(await screen.findByLabelText("Change to simulate"), "edges");
    expect(screen.getByRole("slider")).toHaveAccessibleName(/Edges removed/);
  });

  it("stops an incident with no graph from removing edges", async () => {
    server.use(
      http.get("*/api/v1/incidents/:id", () =>
        HttpResponse.json({
          ...fixtures.incidentDetail(fixtures.incidents[0]!),
          topology_source: "none",
        }),
      ),
    );
    renderApp(<App />, LAB);
    await userEvent.selectOptions(await screen.findByLabelText("Change to simulate"), "edges");
    expect(screen.getByRole("alert")).toHaveTextContent("no call graph");
    expect(screen.getByRole("button", { name: "Run" })).toBeDisabled();
  });

  it("cancels a running scenario", async () => {
    server.use(
      http.post("*/api/v1/incidents/:id/whatif", async () => {
        await new Promise((resolve) => setTimeout(resolve, 2000));
        return HttpResponse.json(fixtures.whatIf);
      }),
    );
    renderApp(<App />, LAB);
    await userEvent.click(await screen.findByRole("button", { name: "Run" }));
    expect(await screen.findByText("Running scenario…")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Cancel" }));
    expect(await screen.findByText("Cancelled.")).toBeInTheDocument();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Run" })).toBeEnabled();
  });

  it("shows a busy service", async () => {
    server.use(
      http.post("*/api/v1/incidents/:id/whatif", () =>
        HttpResponse.json(
          { error: { code: "whatif_busy", message: "Busy" }, disclaimer: "" },
          { status: 503 },
        ),
      ),
    );
    renderApp(<App />, LAB);
    await userEvent.click(await screen.findByRole("button", { name: "Run" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("What-if is busy");
  });

  it("explains a scenario with no intensity scale", async () => {
    server.use(
      http.post("*/api/v1/incidents/:id/whatif", () =>
        HttpResponse.json({ ...fixtures.whatIf, curve: [], intensity_parameter: null }),
      ),
    );
    renderApp(<App />, LAB);
    await userEvent.click(await screen.findByRole("button", { name: "Run" }));
    expect(await screen.findByText("No intensity scale")).toBeInTheDocument();
  });

  it("asks for an incident first", async () => {
    renderApp(<App />, "/what-if");
    expect(await screen.findByRole("link", { name: /INC-0001/ })).toHaveAttribute(
      "href",
      "/incidents/INC-0001/what-if",
    );
  });
});

describe("history", () => {
  it("shows empty states and pending rates", async () => {
    renderApp(<App />, "/history");
    expect(await screen.findByText("No labelled weeks yet")).toBeInTheDocument();
    expect(screen.getAllByText("[pending]").length).toBeGreaterThanOrEqual(2);
    expect(screen.getByRole("button", { name: "Export CSV" })).toBeDisabled();
  });

  it("shows rates with intervals and warns about sample data", async () => {
    server.use(http.get("*/api/v1/history", () => HttpResponse.json(report)));
    renderApp(<App />, "/history");
    expect(await screen.findAllByText("95% interval 50.0% to 100.0%, 1 labelled")).toHaveLength(2);
    expect(screen.getByText(/They include sample data/)).toBeInTheDocument();
    expect(screen.getByText(/not a fair accuracy estimate/)).toBeInTheDocument();
    expect(screen.getByRole("table", { name: "Labelled incidents" })).toHaveTextContent("Top 1");
    expect(screen.getByRole("img", { name: /Outcomes, week of 1970-01-05/ })).toBeInTheDocument();
  });

  it("exports CSV through a download", async () => {
    server.use(http.get("*/api/v1/history", () => HttpResponse.json(report)));
    const create = vi.fn(() => "blob:sample");
    URL.createObjectURL = create;
    URL.revokeObjectURL = vi.fn();
    const click = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {});
    renderApp(<App />, "/history");
    await userEvent.click(await screen.findByRole("button", { name: "Export CSV" }));
    expect(create).toHaveBeenCalledOnce();
    expect(click).toHaveBeenCalledOnce();
    click.mockRestore();
  });
});

describe("csv", () => {
  it("quotes, and neutralises formulas", () => {
    const text = labelsToCsv([{ ...fixtures.label, title: "=cmd()", note: 'say "hi", ok' }]);
    const [header, row] = text.split("\r\n");
    expect(header).toContain("incident_id");
    expect(row).toContain("'=cmd()");
    expect(row).toContain('"say ""hi"", ok"');
  });

  it("leaves numbers alone and blanks nulls", () => {
    const text = labelsToCsv([{ ...fixtures.label, actual_service: null, confidence: -0.5 }]);
    expect(text).toContain(",,");
    expect(text).toContain(",-0.5,");
    expect(text).not.toContain("null");
  });
});

describe("model", () => {
  it("keeps missing values pending and says why", async () => {
    renderApp(<App />, "/model");
    expect(await screen.findByRole("note")).toHaveTextContent("no committed result holds them");
    expect(screen.getByRole("note")).toHaveTextContent("No config is frozen");
    const table = screen.getByRole("table", { name: /Results/ });
    expect(within(table).getAllByText("[pending]")).toHaveLength(2);
    expect(screen.getByText(/not the bins behind them/)).toBeInTheDocument();
  });

  it("shows committed values with their sources", async () => {
    server.use(http.get("*/api/v1/model", () => HttpResponse.json(fullModel)));
    renderApp(<App />, "/model");
    const table = await screen.findByRole("table", { name: /Results/ });
    expect(within(table).getByText("90.0%")).toBeInTheDocument();
    expect(within(table).getByText("80.0% to 95.0%")).toBeInTheDocument();
    expect(within(table).getByText("0.030")).toBeInTheDocument();
    expect(within(table).getByText("a.json")).toBeInTheDocument();
  });

  it("labels robustness as simulated and gives a table", async () => {
    server.use(http.get("*/api/v1/model", () => HttpResponse.json(fullModel)));
    renderApp(<App />, "/model");
    expect(await screen.findByText("Simulated")).toBeInTheDocument();
    expect(screen.getByRole("table", { name: "Top-1 under P2" })).toHaveTextContent(
      "70.0% [60.0%, 80.0%]",
    );
  });

  it("shows a flagged probe in words", async () => {
    server.use(http.get("*/api/v1/model", () => HttpResponse.json(fullModel)));
    renderApp(<App />, "/model");
    expect(await screen.findByText("Flagged")).toBeInTheDocument();
    expect(screen.getByText("Passed")).toBeInTheDocument();
  });

  it("states the limits", async () => {
    renderApp(<App />, "/model");
    expect(await screen.findByText(/not production incidents/)).toBeInTheDocument();
    expect(screen.getByText(/not from all the data/)).toBeInTheDocument();
  });
});

describe("settings", () => {
  it("marks every data source as configuration only", async () => {
    renderApp(<App />, "/settings/integrations");
    expect(await screen.findByText("Configuration only")).toBeInTheDocument();
    expect(screen.getByText(/not connected/)).toBeInTheDocument();
  });

  it("saves blank fields as null", async () => {
    let body: unknown;
    server.use(
      http.put("*/api/v1/settings", async ({ request }) => {
        body = await request.json();
        return HttpResponse.json(fixtures.settings);
      }),
    );
    renderApp(<App />, "/settings/integrations");
    await userEvent.click(await screen.findByRole("button", { name: "Save settings" }));
    expect(await screen.findByText("Settings saved.")).toBeInTheDocument();
    expect(body).toEqual({
      abstain_threshold: null,
      window_seconds: null,
      use_provided_topology: true,
    });
  });

  it("sends what was typed", async () => {
    let body: unknown;
    server.use(
      http.put("*/api/v1/settings", async ({ request }) => {
        body = await request.json();
        return HttpResponse.json(fixtures.settings);
      }),
    );
    renderApp(<App />, "/settings/integrations");
    await userEvent.type(await screen.findByLabelText(/Abstain threshold/), "0.4");
    await userEvent.type(screen.getByLabelText(/Window length/), "300");
    await userEvent.click(screen.getByRole("switch"));
    await userEvent.click(screen.getByRole("button", { name: "Save settings" }));
    await waitFor(() =>
      expect(body).toEqual({
        abstain_threshold: 0.4,
        window_seconds: 300,
        use_provided_topology: false,
      }),
    );
  });

  it("shows a refusal", async () => {
    server.use(
      http.put("*/api/v1/settings", () =>
        HttpResponse.json(
          { error: { code: "validation_error", message: "Out of range" }, disclaimer: "" },
          { status: 422 },
        ),
      ),
    );
    renderApp(<App />, "/settings/integrations");
    await userEvent.click(await screen.findByRole("button", { name: "Save settings" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Out of range");
  });

  it("states the security position", async () => {
    renderApp(<App />, "/settings/integrations");
    expect(await screen.findByText(/never expose it publicly/)).toBeInTheDocument();
  });
});

describe("charts", () => {
  it("gives a curve a text alternative and a threshold", () => {
    render(
      <CurveChart
        label="Example"
        xLabel="x"
        yLabel="y"
        threshold={0.5}
        series={[{ label: "A", points: [{ x: 0, y: 0.25 }] }]}
      />,
    );
    expect(screen.getByRole("img")).toHaveAccessibleName("Example. A: 0 0.250");
    expect(screen.getByText("threshold")).toBeInTheDocument();
  });

  it("names several lines in a legend", () => {
    render(
      <CurveChart
        label="Example"
        xLabel="x"
        yLabel="y"
        series={[
          { label: "A", points: [{ x: 0, y: 0.25 }] },
          { label: "B", points: [{ x: 0, y: 0.5 }] },
        ]}
      />,
    );
    expect(screen.getByText("A")).toBeInTheDocument();
    expect(screen.getByText("B")).toBeInTheDocument();
  });

  it("draws reliability bins when they exist", () => {
    render(<ReliabilityDiagram bins={[{ confidence: 0.5, accuracy: 0.4, cases: 3 }]} />);
    expect(screen.getByRole("img")).toHaveAccessibleName(
      "Reliability diagram: confidence 0.50, accuracy 0.40, 3 cases",
    );
  });
});
