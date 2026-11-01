import { http, HttpResponse } from "msw";
import { describe, expect, it } from "vitest";
import { server } from "../test/setup";
import { api, ApiError } from "./client";

describe("api client", () => {
  it("returns parsed bodies", async () => {
    const page = await api.incidents();
    expect(page.items[0]?.id).toBe("INC-0002");
  });

  it("sends only the filters that are set", async () => {
    let seen = "";
    server.use(
      http.get("*/api/v1/incidents", ({ request }) => {
        seen = new URL(request.url).search;
        return HttpResponse.json({ items: [], next_cursor: null, disclaimer: "" });
      }),
    );
    await api.incidents({ status: "open", severity: "", service: undefined, limit: 5 });
    expect(seen).toBe("?status=open&limit=5");
  });

  it("escapes IDs in paths", async () => {
    let seen = "";
    server.use(
      http.get("*/api/v1/incidents/:id/evidence/:service", ({ request }) => {
        seen = new URL(request.url).pathname;
        return HttpResponse.json({});
      }),
    );
    await api.evidence("INC-1", "a/b");
    expect(seen).toContain("a%2Fb");
  });

  it("maps a known error code to a safe message", async () => {
    const failure = await api.incident("INC-9").catch((error: unknown) => error);
    expect(failure).toBeInstanceOf(ApiError);
    expect(failure).toMatchObject({
      status: 404,
      code: "incident_not_found",
      message: "This incident does not exist.",
    });
  });

  it("keeps the server's message for an unmapped code", async () => {
    server.use(
      http.post("*/api/v1/incidents/:id/feedback", () =>
        HttpResponse.json(
          { error: { code: "conflicting_feedback", message: "That contradicts itself" } },
          { status: 422 },
        ),
      ),
    );
    await expect(api.feedback("INC-1", { correct: true })).rejects.toThrow(
      "That contradicts itself",
    );
  });

  it("handles a body that is not an error document", async () => {
    server.use(http.get("*/api/v1/model", () => new HttpResponse("oops", { status: 502 })));
    await expect(api.model()).rejects.toMatchObject({ status: 502, code: "internal_error" });
  });

  it("reports a network failure", async () => {
    server.use(http.get("*/api/v1/model", () => HttpResponse.error()));
    await expect(api.model()).rejects.toMatchObject({ status: 0, code: "network_error" });
  });

  it("lets an aborted request stay an abort", async () => {
    const controller = new AbortController();
    controller.abort();
    await expect(
      api.whatIf("INC-1", { perturbation: "P1" }, controller.signal),
    ).rejects.toMatchObject({ name: "AbortError" });
  });
});
