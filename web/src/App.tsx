import { lazy, Suspense } from "react";
import { Navigate, Route, Routes } from "react-router-dom";
import { AppShell } from "./components/AppShell";
import { showComponents } from "./config";
import { IncidentDetail } from "./pages/IncidentDetail";
import { Incidents } from "./pages/Incidents";
import { NotFound, Placeholder } from "./pages/Placeholder";
import { ServiceMap } from "./pages/ServiceMap";

// Split out, so sample fixtures never load in a normal session.
const Components = lazy(() =>
  import("./pages/Components").then((module) => ({ default: module.Components })),
);

export function App() {
  return (
    <Routes>
      <Route element={<AppShell />}>
        <Route index element={<Navigate to="/incidents" replace />} />
        <Route path="incidents" element={<Incidents />} />
        <Route path="incidents/:id" element={<IncidentDetail />} />
        <Route path="incidents/:id/what-if" element={<Placeholder title="What-if" phase={5} />} />
        <Route path="service-map" element={<ServiceMap />} />
        <Route path="what-if" element={<Placeholder title="What-if lab" phase={5} />} />
        <Route path="history" element={<Placeholder title="History & feedback" phase={5} />} />
        <Route path="model" element={<Placeholder title="Model" phase={5} />} />
        <Route
          path="settings/integrations"
          element={<Placeholder title="Integrations & settings" phase={5} />}
        />
        {showComponents && (
          <Route
            path="components"
            element={
              <Suspense fallback={null}>
                <Components />
              </Suspense>
            }
          />
        )}
        <Route path="*" element={<NotFound />} />
      </Route>
    </Routes>
  );
}
