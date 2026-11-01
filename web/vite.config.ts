import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

const API = "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    strictPort: true,
    proxy: { "/api": API, "/diagnose": API },
  },
  test: {
    environment: "jsdom",
    globals: true,
    // Tests must not depend on a developer's .env.local.
    env: { VITE_SHOW_COMPONENTS: "false", VITE_USE_MOCKS: "false" },
    setupFiles: "./src/test/setup.ts",
    css: { modules: { classNameStrategy: "non-scoped" } },
  },
});
