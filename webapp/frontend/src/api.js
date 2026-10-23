// All calls to the backend go through here. The Vite dev server forwards /api to http://127.0.0.1:8000.

async function request(path, options = {}) {
  let res;
  try {
    res = await fetch(path, options);
  } catch {
    throw new Error("Cannot reach the backend. Is it running on port 8000?");
  }
  const text = await res.text();
  let body = null;
  try { body = text ? JSON.parse(text) : null; } catch { body = text; }
  // a saved run's id comes back in a header (the body stays the exact component output);
  // it is attached as a hidden property, so JSON.stringify(body) is unchanged
  const runId = res.headers.get("X-Run-Id");
  if (runId && body && typeof body === "object") Object.defineProperty(body, "runId", { value: runId, enumerable: false });
  if (!res.ok) {
    const d = body && body.detail !== undefined ? body.detail : body;
    const msg = typeof d === "string" ? d : JSON.stringify(d, null, 2);
    throw new Error(`${res.status}: ${msg}`);
  }
  return body;
}

const get = (path) => request(path);
const postJson = (path, data) =>
  request(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data) });
const postForm = (path, fields) => {
  const fd = new FormData();
  Object.entries(fields).forEach(([k, v]) => fd.append(k, v));
  return request(path, { method: "POST", body: fd });
};

export const api = {
  // platform
  health: () => get("/api/health"),
  datasets: () => get("/api/datasets"),
  latestRun: () => get("/api/pipeline/latest"),
  runPipeline: (dataset, file) => postForm("/api/pipeline/run", { dataset, file }),
  // Drift Lab
  labOptions: (dataset) => get(`/api/lab/options?dataset=${encodeURIComponent(dataset)}`),
  labRun: (params) => postJson("/api/lab/run", params),
  // saved runs (Run history)
  runs: () => get("/api/runs"),
  run: (id) => get(`/api/runs/${encodeURIComponent(id)}`),
  runFileUrl: (id) => `/api/runs/${encodeURIComponent(id)}`,
  runBatchUrl: (id) => `/api/runs/${encodeURIComponent(id)}/batch`,
  runsExportUrl: () => "/api/runs/export.csv",
  // C1
  c1EngineInfo: (dataset) => get(`/api/c1/engine-info?dataset=${encodeURIComponent(dataset)}`),
  c1Detect: (dataset, file) => postForm("/api/c1/detect", { dataset, file }),
  // C3 (Mathusan K)
  c3Info: () => get("/api/c3/info"),
  c3Runs: () => get("/api/c3/runs"),
  c3Models: () => get("/api/c3/models"),
  c3LatestUpstream: () => get("/api/c3/upstream/latest"),
  c3IntegratedRepair: (payload) => postJson("/api/c3/repair", payload),
  c3UploadArtifact: (role, file) => postForm("/api/c3/artifacts", { role, file }),
  c3Scenarios: (dataset) => get(`/api/c3/scenarios${dataset ? `?dataset=${encodeURIComponent(dataset)}` : ""}`),
  c3ScenarioEvidence: (scenarioId, dataset) => get(`/api/c3/scenario/${encodeURIComponent(scenarioId)}/evidence${dataset ? `?dataset=${encodeURIComponent(dataset)}` : ""}`),
  c3Benchmarks: (dataset) => get(`/api/c3/benchmarks${dataset ? `?dataset=${encodeURIComponent(dataset)}` : ""}`),
  c3Repair: (payload) => postJson("/api/c3/repair", payload),
  c3DemoRepair: (payloadOrDataset, maybeScenario) => {
    const body =
      typeof payloadOrDataset === "object" && payloadOrDataset !== null
        ? payloadOrDataset
        : {
            dataset: payloadOrDataset || "uci",
            scenario: maybeScenario || "education_num_marginal",
          };
    return postJson("/api/c3/demo/repair", body);
  },
  c3DownloadUrl: (runId) => `/api/c3/packages/${encodeURIComponent(runId)}/dataset`,
  // C2 (Tharmika): diagnosis runs, run history with temporal SHAP, business report
  c2Info: () => get("/api/c2/info"),
  c2RunNew: (params) => postJson("/api/c2/run", params),
  c2Runs: () => get("/api/c2/runs"),
  c2Run: (n) => get(`/api/c2/runs/${encodeURIComponent(n)}`),
  c2Report: (n, refresh = false) => get(`/api/c2/runs/${encodeURIComponent(n)}/report${refresh ? "?refresh=true" : ""}`),
  c2ReportMdUrl: (n) => `/api/c2/runs/${encodeURIComponent(n)}/report.md`,
  c2Reset: () => postJson("/api/c2/reset", {}),
  // C4: add your own calls here
};
