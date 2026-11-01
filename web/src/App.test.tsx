import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { describe, expect, it } from "vitest";
import { App } from "./App";
import { DISCLAIMER, READ_ONLY } from "./components/Disclaimer";
import { incidentPage, model } from "./mocks/fixtures";
import { renderApp } from "./test/render";
import { server } from "./test/setup";

describe("app shell", () => {
  it("sends the root to incidents", async () => {
    renderApp(<App />);
    expect(await screen.findByRole("link", { name: "Incidents" })).toHaveAttribute(
      "aria-current",
      "page",
    );
  });

  it("shows the disclaimer and the read-only statement", () => {
    renderApp(<App />, "/model");
    const footer = screen.getByRole("contentinfo");
    expect(footer).toHaveTextContent(DISCLAIMER);
    expect(footer).toHaveTextContent(READ_ONLY);
  });

  it("names the page in the breadcrumbs", () => {
    renderApp(<App />, "/settings/integrations");
    const crumbs = screen.getByRole("navigation", { name: "Breadcrumb" });
    expect(within(crumbs).getByText("Integrations")).toHaveAttribute("aria-current", "page");
  });

  it("shows the model's health from the API", async () => {
    renderApp(<App />);
    const health = screen.getByRole("region", { name: "Model health" });
    expect(await within(health).findByText("sample")).toBeInTheDocument();
    expect(within(health).getByText("yes")).toBeInTheDocument();
    expect(within(health).getByText("not frozen")).toBeInTheDocument();
  });

  it("shows pending when a health value is missing", async () => {
    server.use(
      http.get("*/api/v1/model", () => HttpResponse.json({ ...model, abstain_threshold: null })),
    );
    renderApp(<App />);
    const health = screen.getByRole("region", { name: "Model health" });
    await within(health).findByText("sample");
    expect(within(health).getByText("[pending]")).toBeInTheDocument();
  });

  it("says when the model cannot be reached", async () => {
    server.use(http.get("*/api/v1/model", () => HttpResponse.error()));
    renderApp(<App />);
    const health = screen.getByRole("region", { name: "Model health" });
    expect(await within(health).findByText("Not reachable")).toBeInTheDocument();
  });

  it("shows the sample badge for demo incidents", async () => {
    renderApp(<App />);
    expect(await screen.findByText("Sample data")).toBeInTheDocument();
  });

  it("hides the sample badge for real incidents", async () => {
    const items = incidentPage.items.map((item) => ({ ...item, is_demo: false }));
    server.use(http.get("*/api/v1/incidents", () => HttpResponse.json({ ...incidentPage, items })));
    renderApp(<App />);
    await within(screen.getByRole("region", { name: "Model health" })).findByText("sample");
    expect(screen.queryByText("Sample data")).not.toBeInTheDocument();
  });

  it("focuses the search on / unless someone is typing", async () => {
    renderApp(<App />);
    const search = screen.getByRole("searchbox", { name: "Search" });
    await userEvent.keyboard("/");
    expect(search).toHaveFocus();
    expect(search).toHaveValue("");
    await userEvent.keyboard("/");
    expect(search).toHaveValue("/");
  });

  it("hides the component sheet unless it is turned on", () => {
    renderApp(<App />, "/components");
    expect(screen.queryByRole("link", { name: "Components" })).not.toBeInTheDocument();
    expect(screen.getByText("This page does not exist")).toBeInTheDocument();
  });

  it("opens the notifications", async () => {
    renderApp(<App />);
    await userEvent.click(screen.getByRole("button", { name: "Notifications" }));
    expect(screen.getByText("No notifications.")).toBeInTheDocument();
  });
});
