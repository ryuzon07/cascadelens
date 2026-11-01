import "@fontsource/ibm-plex-mono/400.css";
import "@fontsource/ibm-plex-mono/500.css";
import "@fontsource/ibm-plex-sans/400.css";
import "@fontsource/ibm-plex-sans/500.css";
import "@fontsource/ibm-plex-sans/600.css";
import "./styles/tokens.css";
import "./styles/global.css";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import { App } from "./App";
import { useMocks } from "./config";

const client = new QueryClient({
  defaultOptions: { queries: { retry: 1, refetchOnWindowFocus: false } },
});

async function start() {
  if (useMocks) {
    const { worker } = await import("./mocks/browser");
    await worker.start({ onUnhandledFrame: "bypass" });
  }
  const root = document.getElementById("root");
  if (!root) throw new Error("The page has no root element");
  createRoot(root).render(
    <StrictMode>
      <QueryClientProvider client={client}>
        <BrowserRouter>
          <App />
        </BrowserRouter>
      </QueryClientProvider>
    </StrictMode>,
  );
}

void start();
