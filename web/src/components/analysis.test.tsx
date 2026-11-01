import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { describe, expect, it, vi } from "vitest";
import * as fixtures from "../mocks/fixtures";
import { renderApp } from "../test/render";
import { server } from "../test/setup";
import { formatPercent, formatSigned, formatTime, formatUnit } from "../format";
import { ConfidenceGauge } from "./ConfidenceGauge";
import { ContributionBars } from "./ContributionBars";
import { CoverageHeatmap } from "./CoverageHeatmap";
import { abstainReasons, DecisionCard, reasonFor } from "./DecisionCard";
import { FeedbackForm } from "./FeedbackForm";
import { LineChart } from "./LineChart";
import { OnsetTimeline } from "./OnsetTimeline";
import { StackedBars } from "./StackedBars";
import { SuspectRow } from "./SuspectRow";

const confident = fixtures.incidentDetail(fixtures.incidents[0]!);
const abstained = fixtures.incidentDetail(fixtures.incidents[1]!);

describe("format", () => {
  it("never rounds a value near 1 up to 1", () => {
    expect(formatUnit(0.99998)).toBe("0.99998");
    expect(formatUnit(0.9999999)).toBe(">0.99999");
    expect(formatUnit(1)).toBe("1.000");
    expect(formatUnit(0.295)).toBe("0.295");
  });

  it("formats times, shares and signs", () => {
    expect(formatTime(0)).toBe("1970-01-01 00:00 UTC");
    expect(formatPercent(0.5)).toBe("50.0%");
    expect(formatSigned(1)).toBe("+1.00");
    expect(formatSigned(-1)).toBe("-1.00");
  });
});

describe("ConfidenceGauge", () => {
  it("reports its value to assistive technology", () => {
    render(<ConfidenceGauge confidence={0.3} threshold={0.5} />);
    expect(screen.getByRole("meter", { name: "Confidence" })).toHaveAttribute(
      "aria-valuetext",
      "0.300",
    );
    expect(screen.getByText("Threshold 0.500")).toBeInTheDocument();
  });

  it("says when there is no threshold", () => {
    render(<ConfidenceGauge confidence={0.3} threshold={null} />);
    expect(screen.getByText("No threshold")).toBeInTheDocument();
  });
});

describe("DecisionCard", () => {
  it("names the origin when confident", () => {
    render(<DecisionCard incident={confident} telemetryCoverage={1} benchmarkTop3={0.97} />);
    const card = screen.getByRole("region", { name: "Decision" });
    expect(card).toHaveTextContent("Most likely origin");
    expect(card).toHaveTextContent("alpha");
    expect(card).toHaveTextContent("97.0%");
    expect(card).not.toHaveTextContent("Investigate manually");
  });

  it("gives no origin when abstaining", () => {
    render(<DecisionCard incident={abstained} telemetryCoverage={0.8} benchmarkTop3={null} />);
    const card = screen.getByRole("region", { name: "Decision" });
    expect(card).toHaveTextContent("No confident answer. Investigate manually.");
    expect(card).not.toHaveTextContent("Most likely origin");
    expect(card).toHaveTextContent("80.0% of samples are present");
    expect(card).toHaveTextContent("What would help: complete telemetry");
    expect(card).toHaveTextContent("[pending]");
  });

  it("warns that an uncalibrated confidence is not a probability", () => {
    const incident = { ...confident, decision: "uncalibrated" as const };
    render(<DecisionCard incident={incident} telemetryCoverage={1} benchmarkTop3={null} />);
    expect(screen.getByText(/not a\s+probability/)).toBeInTheDocument();
  });

  it("explains inferred and missing graphs", () => {
    const inferred = abstainReasons({ ...abstained, topology_source: "inferred" }, 1);
    expect(inferred.join(" ")).toContain("inferred from metrics");
    const none = abstainReasons({ ...abstained, topology_source: "none" }, 1);
    expect(none.join(" ")).toContain("No call graph");
  });

  it("describes the ranking without claiming a cause", () => {
    expect(reasonFor(confident)).toContain("signal features");
    expect(reasonFor(confident).toLowerCase()).not.toContain("caused");
  });
});

describe("SuspectRow", () => {
  const item = confident.diagnosis.ranking[0]!;

  it("selects a service by keyboard", async () => {
    const select = vi.fn();
    render(<SuspectRow item={item} rank={1} selected={false} onSelect={select} />);
    await userEvent.tab();
    await userEvent.keyboard("{Enter}");
    expect(select).toHaveBeenCalledWith("alpha");
  });

  it("marks the selected row", () => {
    render(<SuspectRow item={item} rank={1} selected onSelect={() => {}} />);
    expect(screen.getByRole("button")).toHaveAttribute("aria-pressed", "true");
  });

  it("states missing telemetry", () => {
    const gaps = { ...item, evidence: { ...item.evidence, onset_rank: null, missing_frac: 0.5 } };
    render(<SuspectRow item={gaps} rank={2} selected={false} onSelect={() => {}} />);
    expect(screen.getByText("no sustained anomaly, 50.0% missing")).toBeInTheDocument();
  });
});

describe("charts", () => {
  it("shows contributions with their sign in text", () => {
    render(
      <ContributionBars
        label="Example"
        items={[
          { label: "Up", contribution: 2 },
          { label: "Down", contribution: -1 },
        ]}
      />,
    );
    expect(screen.getByText("+2.00")).toBeInTheDocument();
    expect(screen.getByText("-1.00")).toBeInTheDocument();
  });

  it("breaks a line at a gap and draws nothing across it", () => {
    const series = {
      metric: "cpu",
      family: "cpu",
      points: [
        { ts: 0, value: 1 },
        { ts: 5, value: 2 },
        { ts: 10, value: null },
        { ts: 15, value: 3 },
        { ts: 20, value: 4 },
      ],
    };
    const { container } = render(<LineChart series={series} />);
    expect(container.querySelectorAll("path")).toHaveLength(2);
    expect(screen.getByRole("img")).toHaveAccessibleName("cpu: 5 points, 1 missing");
  });

  it("says when a metric has no samples", () => {
    render(
      <LineChart series={{ metric: "cpu", family: "cpu", points: [{ ts: 0, value: null }] }} />,
    );
    expect(screen.getByText("No samples in this window.")).toBeInTheDocument();
  });

  it("gives the timeline a text alternative", () => {
    const start = fixtures.timeline.window_start;
    const timeline = {
      ...fixtures.timeline,
      services: [
        ...fixtures.timeline.services,
        {
          service: "silent",
          onset: null,
          anomalous_until: null,
          telemetry_stopped: start + 225,
          has_telemetry: true,
        },
        {
          service: "absent",
          onset: null,
          anomalous_until: null,
          telemetry_stopped: null,
          has_telemetry: false,
        },
      ],
    };
    render(<OnsetTimeline timeline={timeline} />);
    const label = screen.getByRole("img").getAttribute("aria-label");
    expect(label).toContain("alpha: anomalous from +245s");
    expect(label).toContain("silent: no anomaly, telemetry stopped at +225s");
    expect(label).toContain("absent: no telemetry");
  });

  it("writes every coverage share as text", () => {
    render(<CoverageHeatmap coverage={fixtures.coverage} />);
    expect(screen.getByRole("cell", { name: "50.0%" })).toBeInTheDocument();
    expect(screen.getByRole("cell", { name: "100.0%" })).toBeInTheDocument();
  });

  it("lists counts beside a stacked bar", () => {
    render(
      <StackedBars
        label="Decisions"
        segments={[
          { label: "Confident", value: 3, color: "red" },
          { label: "Abstained", value: 0, color: "blue" },
        ]}
      />,
    );
    expect(screen.getByRole("img")).toHaveAccessibleName("Decisions: Confident 3, Abstained 0");
    expect(screen.getByText("Abstained: 0")).toBeInTheDocument();
  });
});

describe("FeedbackForm", () => {
  it("needs an answer first", () => {
    renderApp(<FeedbackForm incident={confident} />);
    expect(screen.getByRole("button", { name: "Save label" })).toBeDisabled();
  });

  it("sends a confirmed origin", async () => {
    let body: unknown;
    server.use(
      http.post("*/api/v1/incidents/:id/feedback", async ({ request }) => {
        body = await request.json();
        return HttpResponse.json(fixtures.label, { status: 201 });
      }),
    );
    renderApp(<FeedbackForm incident={confident} />);
    await userEvent.click(screen.getByRole("radio", { name: "No" }));
    await userEvent.selectOptions(screen.getByLabelText("Confirmed origin"), "gamma");
    await userEvent.type(screen.getByLabelText("Note"), "Checked.");
    await userEvent.click(screen.getByRole("button", { name: "Save label" }));
    expect(await screen.findByText("Label saved.")).toBeInTheDocument();
    expect(body).toEqual({ correct: false, actual_service: "gamma", note: "Checked." });
  });

  it("offers only other services as the confirmed origin", async () => {
    renderApp(<FeedbackForm incident={confident} />);
    await userEvent.click(screen.getByRole("radio", { name: "No" }));
    const options = within(screen.getByLabelText("Confirmed origin")).getAllByRole("option");
    expect(options.map((option) => option.textContent)).toEqual(["Not known", "gamma"]);
  });

  it("shows a server refusal", async () => {
    server.use(
      http.post("*/api/v1/incidents/:id/feedback", () =>
        HttpResponse.json(
          { error: { code: "conflicting_feedback", message: "That contradicts itself" } },
          { status: 422 },
        ),
      ),
    );
    renderApp(<FeedbackForm incident={confident} />);
    await userEvent.click(screen.getByRole("radio", { name: "Yes" }));
    await userEvent.click(screen.getByRole("button", { name: "Save label" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("That contradicts itself");
  });

  it("says a closed incident already has a label", () => {
    renderApp(<FeedbackForm incident={{ ...confident, status: "closed" }} />);
    expect(screen.getByText(/A label is saved/)).toBeInTheDocument();
  });
});
