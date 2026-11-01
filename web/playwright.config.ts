import { defineConfig, devices } from "@playwright/test";

const PORT = 8765;
const PYTHON = process.env.E2E_PYTHON ?? "uv run --frozen python";

// The tests change the database in turn, so they run one at a time against one server.
export default defineConfig({
  testDir: "./e2e",
  workers: 1,
  fullyParallel: false,
  forbidOnly: Boolean(process.env.CI),
  reporter: "list",
  timeout: 30_000,
  use: { baseURL: `http://127.0.0.1:${PORT}`, trace: "retain-on-failure" },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: {
    command: `${PYTHON} ../scripts/e2e_server.py ${PORT} dist-e2e`,
    url: `http://127.0.0.1:${PORT}/health`,
    timeout: 120_000,
    reuseExistingServer: false,
    stdout: "pipe",
  },
});
