import { lazy, Suspense } from "react";
import { Navigate, Route, Routes } from "react-router-dom";
import { AppShell } from "./components/AppShell";
import { showComponents } from "./config";
import { History } from "./pages/History";
import { IncidentDetail } from "./pages/IncidentDetail";
import { Incidents } from "./pages/Incidents";
import { Model } from "./pages/Model";
import { NotFound } from "./pages/Placeholder";
import { ServiceMap } from "./pages/ServiceMap";
import { Settings } from "./pages/Settings";
import { WhatIf, WhatIfChooser } from "./pages/WhatIf";

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
        <Route path="incidents/:id/what-if" element={<WhatIf />} />
        <Route path="service-map" element={<ServiceMap />} />
        <Route path="what-if" element={<WhatIfChooser />} />
        <Route path="history" element={<History />} />
        <Route path="model" element={<Model />} />
        <Route path="settings/integrations" element={<Settings />} />
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
