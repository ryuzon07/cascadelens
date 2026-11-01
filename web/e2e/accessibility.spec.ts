import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page } from "@playwright/test";

const ROUTES = [
  "/incidents",
  "/service-map",
  "/what-if",
  "/history",
  "/model",
  "/settings/integrations",
  "/components",
  "/nothing-here",
];

async function serious(page: Page) {
  const results = await new AxeBuilder({ page })
    .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"])
    .analyze();
  expect(results.passes.length, "axe must have checked something").toBeGreaterThan(5);
  return results.violations
    .filter((item) => item.impact === "serious" || item.impact === "critical")
    .map((item) => `${item.id}: ${item.nodes.map((node) => node.target.join(" ")).join(", ")}`);
}

async function settled(page: Page) {
  await expect(page.getByRole("heading").first()).toBeVisible();
  await page.waitForLoadState("networkidle");
}

for (const route of ROUTES) {
  test(`axe finds nothing serious on ${route}`, async ({ page }) => {
    await page.goto(route);
    await settled(page);
    expect(await serious(page)).toEqual([]);
  });
}

test("axe finds nothing serious on every incident view", async ({ page, request }) => {
  const list = await (await request.get("/api/v1/incidents")).json();
  const id = list.items[0].id as string;
  await page.goto(`/incidents/${id}`);
  await settled(page);
  expect(await serious(page)).toEqual([]);
  for (const name of ["Evidence", "Timeline", "What-if", "Raw data"]) {
    await page.getByRole("tab", { name }).click();
    await page.waitForLoadState("networkidle");
    expect(await serious(page), name).toEqual([]);
  }
});

test("axe finds nothing serious on the what-if lab after a run", async ({ page, request }) => {
  const list = await (await request.get("/api/v1/incidents")).json();
  await page.goto(`/incidents/${list.items[0].id}/what-if`);
  await page.getByRole("button", { name: "Run", exact: true }).click();
  await expect(page.getByRole("table", { name: "Rank changes" })).toBeVisible();
  expect(await serious(page)).toEqual([]);
});

test("axe finds nothing serious at phone width", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/incidents");
  await settled(page);
  expect(await serious(page)).toEqual([]);
  const scroll = await page.evaluate(() => document.documentElement.scrollWidth);
  expect(scroll).toBeLessThanOrEqual(390);
});

test("the keyboard reaches the main controls", async ({ page }) => {
  await page.goto("/incidents");
  await page.keyboard.press("Tab");
  await expect(page.getByRole("link", { name: "Skip to content" })).toBeFocused();
  await page.keyboard.press("Enter");
  await expect(page.locator("#main")).toBeFocused();
});
