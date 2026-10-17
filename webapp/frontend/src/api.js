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
  // C1
  c1EngineInfo: (dataset) => get(`/api/c1/engine-info?dataset=${encodeURIComponent(dataset)}`),
  c1Detect: (dataset, file) => postForm("/api/c1/detect", { dataset, file }),
  // C2, C3, C4: add your own calls here, e.g.  c2Diagnose: (...) => postForm("/api/c2/diagnose", {...}),
};
