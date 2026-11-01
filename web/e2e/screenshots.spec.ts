import { expect, test } from "@playwright/test";

const OUT = "../docs/demo";

// Writes the README images. Run with SCREENSHOTS=1 after the UI changes.
test.skip(!process.env.SCREENSHOTS, "Only runs when refreshing the images");
test.use({ viewport: { width: 1280, height: 860 } });

test("capture the screens", async ({ page, request }) => {
  const list = await (await request.get("/api/v1/incidents")).json();
  const id = list.items[0].id as string;

  await page.goto("/incidents");
  await expect(page.getByRole("table", { name: "Incidents" })).toBeVisible();
  await page.waitForLoadState("networkidle");
  await page.screenshot({ path: `${OUT}/ui-incidents.png` });

  await page.goto(`/incidents/${id}`);
  await expect(page.locator("canvas").first()).toBeVisible();
  await page.waitForLoadState("networkidle");
  await page.screenshot({ path: `${OUT}/ui-incident.png` });

  await page.goto(`/incidents/${id}/what-if`);
  await page.getByRole("button", { name: "Heavy telemetry loss" }).click();
  await page.getByRole("button", { name: "Run", exact: true }).click();
  await expect(page.getByRole("table", { name: "Rank changes" })).toBeVisible();
  await page.screenshot({ path: `${OUT}/ui-what-if.png` });
});
