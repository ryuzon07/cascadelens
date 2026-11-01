import { expect, test, type APIRequestContext, type Page } from "@playwright/test";

const API = "/api/v1";
const DISCLAIMER = "Ranking of likely origin services, not proof of causation.";

interface Row {
  id: string;
  decision: string;
  confidence: number;
  top_service: string;
  title: string;
}

async function rows(request: APIRequestContext): Promise<Row[]> {
  const page = await (await request.get(`${API}/incidents`)).json();
  return page.items as Row[];
}

/** Set a threshold, re-run an incident and return it, so a test can rely on its decision. */
async function withThreshold(request: APIRequestContext, id: string, threshold: number | null) {
  await request.put(`${API}/settings`, { data: { abstain_threshold: threshold } });
  await request.post(`${API}/incidents/${id}/rerun`);
}

/** The least confident incident, the one a threshold of 1 is certain to turn down. */
async function leastConfident(request: APIRequestContext): Promise<string> {
  const sorted = (await rows(request)).sort((a, b) => a.confidence - b.confidence);
  return sorted[0]!.id;
}

async function firstId(request: APIRequestContext): Promise<string> {
  return (await rows(request))[0]!.id;
}

function networkGuard(page: Page) {
  const outside: string[] = [];
  page.on("request", (request) => {
    if (!request.url().startsWith("http://127.0.0.1:") && !request.url().startsWith("data:")) {
      outside.push(request.url());
    }
  });
  return outside;
}

test.afterEach(async ({ request }) => {
  await request.put(`${API}/settings`, { data: {} });
});

test("lists incidents with a sample badge and the disclaimer", async ({ page }) => {
  const outside = networkGuard(page);
  await page.goto("/");
  await expect(page).toHaveURL(/\/incidents$/);
  await expect(page.getByRole("table", { name: "Incidents" })).toBeVisible();
  await expect(page.getByText("Sample data")).toBeVisible();
  await expect(page.getByRole("contentinfo")).toContainText(DISCLAIMER);
  expect(outside).toEqual([]);
});

test("a deep link loads from the server", async ({ page, request }) => {
  const id = await firstId(request);
  await page.goto(`/incidents/${id}`);
  await expect(page.getByRole("heading", { level: 1 })).toBeVisible();
});

test("a confident incident, a suspect, feedback and history", async ({ page, request }) => {
  const id = await firstId(request);
  await withThreshold(request, id, 0);
  await page.goto("/incidents");
  await page.getByRole("link", { name: id }).click();
  await expect(page.getByRole("region", { name: "Decision" })).toContainText("Most likely origin");
  await expect(page.locator("canvas").first()).toBeVisible();

  const suspects = page
    .getByRole("list")
    .filter({ has: page.getByRole("button", { pressed: true }) });
  const other = suspects.getByRole("button", { pressed: false }).first();
  const name = (await other.locator("span span").first().textContent()) ?? "";
  await other.click();
  await expect(page.getByText(`Feature contributions: ${name}`)).toBeVisible();

  await page.getByRole("radio", { name: "Yes" }).check();
  await page.getByLabel("Note").fill("Confirmed in the end-to-end test.");
  await page.getByRole("button", { name: "Save label" }).click();
  await expect(page.getByText("Label saved.")).toBeVisible();

  await page.getByRole("link", { name: "History & feedback" }).click();
  const table = page.getByRole("table", { name: "Labelled incidents" });
  await expect(table.getByRole("link", { name: id })).toBeVisible();
  await expect(page.getByText(/include sample data/)).toBeVisible();
  const download = page.waitForEvent("download");
  await page.getByRole("button", { name: "Export CSV" }).click();
  expect((await download).suggestedFilename()).toBe("cascadelens-labels.csv");
});

test("an abstained incident shows leads, not an answer", async ({ page, request }) => {
  const id = await leastConfident(request);
  await withThreshold(request, id, 1);
  await page.goto(`/incidents/${id}`);
  await expect(page.getByRole("region", { name: "Decision" })).toContainText("No confident answer");
  await expect(page.getByRole("heading", { name: "Leads to check" })).toBeVisible();
  await expect(page.getByText("Most likely origin")).toHaveCount(0);
});

test("a what-if run changes nothing stored", async ({ page, request }) => {
  const id = await firstId(request);
  const before = await (await request.get(`${API}/incidents/${id}`)).json();
  await page.goto(`/incidents/${id}/what-if`);
  await page.getByRole("button", { name: "Heavy telemetry loss" }).click();
  await page.getByRole("button", { name: "Run", exact: true }).click();
  await expect(page.getByText("Simulated", { exact: true })).toBeVisible();
  await expect(page.getByRole("table", { name: "Rank changes" })).toBeVisible();
  await expect(page.getByRole("img", { name: /Confidence against intensity/ })).toBeVisible();
  expect(await (await request.get(`${API}/incidents/${id}`)).json()).toEqual(before);
});

test("a new threshold reaches an incident only after a re-run", async ({ page, request }) => {
  const id = await leastConfident(request);
  await withThreshold(request, id, 0);
  await request.put(`${API}/settings`, { data: {} });
  const before = await (await request.get(`${API}/incidents/${id}`)).json();
  await page.goto("/settings/integrations");
  await page.getByLabel(/Abstain threshold/).fill("1");
  await page.getByRole("button", { name: "Save settings" }).click();
  await expect(page.getByText("Settings saved.")).toBeVisible();
  await page.goto(`/incidents/${id}`);
  await expect(page.getByRole("heading", { level: 1 })).toBeVisible();
  const stored = await (await request.get(`${API}/incidents/${id}`)).json();
  expect(stored.abstain_threshold).toBe(before.abstain_threshold);
  await page.getByRole("button", { name: "Re-run diagnosis" }).click();
  await expect(page.getByRole("region", { name: "Decision" })).toContainText("No confident answer");
});

test("the model screen reads committed results", async ({ page }) => {
  await page.goto("/model");
  await expect(page.getByRole("table", { name: /Results from committed/ })).toContainText(
    "experiments/robustness/results.json",
  );
  await expect(page.getByText("Flagged")).toBeVisible();
  await page.getByLabel("Perturbation").selectOption("P1");
  await expect(page.getByRole("table", { name: "Top-1 under P1" })).toBeVisible();
});

test("search filters the incident list", async ({ page, request }) => {
  const [first] = await rows(request);
  await page.goto("/incidents");
  await page.keyboard.press("/");
  await page.keyboard.type(`${first!.id}\n`);
  await expect(page).toHaveURL(/q=/);
  await expect(page.getByRole("table", { name: "Incidents" }).getByRole("row")).toHaveCount(2);
});

test("a missing incident says so", async ({ page }) => {
  await page.goto("/incidents/INC-9999");
  await expect(page.getByRole("alert")).toContainText("does not exist");
});

test("no ground truth reaches the page", async ({ page, request }) => {
  const id = await firstId(request);
  const bodies: string[] = [];
  page.on("response", async (response) => {
    if (response.url().includes(API)) bodies.push(await response.text());
  });
  await page.goto(`/incidents/${id}`);
  await expect(page.getByRole("heading", { level: 1 })).toBeVisible();
  await page.waitForLoadState("networkidle");
  const text = bodies.join("") + (await page.content());
  expect(bodies.length).toBeGreaterThan(0);
  expect(text).not.toContain("root_cause_service");
  expect(text).not.toContain("inject_time");
});
