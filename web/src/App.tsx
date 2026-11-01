import { Navigate, Route, Routes } from "react-router-dom";
import { AppShell } from "./components/AppShell";
import { showComponents } from "./config";
import { Components } from "./pages/Components";
import { NotFound, Placeholder } from "./pages/Placeholder";

export function App() {
  return (
    <Routes>
      <Route element={<AppShell />}>
        <Route index element={<Navigate to="/incidents" replace />} />
        <Route path="incidents" element={<Placeholder title="Incidents" phase={4} />} />
        <Route path="incidents/:id" element={<Placeholder title="Incident" phase={4} />} />
        <Route path="incidents/:id/what-if" element={<Placeholder title="What-if" phase={5} />} />
        <Route path="service-map" element={<Placeholder title="Service map" phase={4} />} />
        <Route path="what-if" element={<Placeholder title="What-if lab" phase={5} />} />
        <Route path="history" element={<Placeholder title="History & feedback" phase={5} />} />
        <Route path="model" element={<Placeholder title="Model" phase={5} />} />
        <Route
          path="settings/integrations"
          element={<Placeholder title="Integrations & settings" phase={5} />}
        />
        {showComponents && <Route path="components" element={<Components />} />}
        <Route path="*" element={<NotFound />} />
      </Route>
    </Routes>
  );
}
