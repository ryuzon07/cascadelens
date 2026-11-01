import { http, HttpResponse } from "msw";
import * as fixtures from "./fixtures";

const API = "*/api/v1";

function detail(id: string | readonly string[] | undefined) {
  const found = fixtures.incidents.find((item) => item.id === id);
  if (found) return HttpResponse.json(fixtures.incidentDetail(found));
  return HttpResponse.json(
    {
      error: { code: "incident_not_found", message: "No such incident" },
      disclaimer: fixtures.DISCLAIMER,
    },
    { status: 404 },
  );
}

export const handlers = [
  http.get(`${API}/incidents`, () => HttpResponse.json(fixtures.incidentPage)),
  http.get(`${API}/incidents/:id`, ({ params }) => detail(params.id)),
  http.post(`${API}/incidents/:id/rerun`, ({ params }) => detail(params.id)),
  http.get(`${API}/incidents/:id/graph`, () => HttpResponse.json(fixtures.graph)),
  http.get(`${API}/incidents/:id/timeline`, () => HttpResponse.json(fixtures.timeline)),
  http.get(`${API}/incidents/:id/coverage`, () => HttpResponse.json(fixtures.coverage)),
  http.get(`${API}/incidents/:id/evidence/:service`, ({ params }) =>
    HttpResponse.json(fixtures.serviceEvidence(String(params.service))),
  ),
  http.post(`${API}/incidents/:id/whatif`, () => HttpResponse.json(fixtures.whatIf)),
  http.post(`${API}/incidents/:id/feedback`, () =>
    HttpResponse.json(fixtures.label, { status: 201 }),
  ),
  http.get(`${API}/history`, () => HttpResponse.json(fixtures.history)),
  http.get(`${API}/model`, () => HttpResponse.json(fixtures.model)),
  http.get(`${API}/settings`, () => HttpResponse.json(fixtures.settings)),
  http.put(`${API}/settings`, () => HttpResponse.json(fixtures.settings)),
  http.get(`${API}/integrations`, () => HttpResponse.json(fixtures.integrations)),
];
