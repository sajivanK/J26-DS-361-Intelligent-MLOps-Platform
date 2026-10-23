import { useState } from "react";
import { api } from "../../api.js";
import { useApi } from "../../hooks.js";
import { useSession } from "../../session.jsx";
import { ErrorBox, JsonBox, KV, PageHeader, Verdict } from "../../ui/parts.jsx";
import "./C3Page.css";

const example = {
  c1_evidence: {
    model_id: "your-model-v1", predicted_harm: 0.052, harm_interval: [0.02, 0.08],
    harm_severity: "moderate", drift_detected: true, marginal_extent: "single",
    interaction_suspected: 0.2, affected_features: ["age"], harm_metric: "accuracy_drop",
  },
  c2_diagnosis: {
    root_cause_feature: "age", root_cause_category: "marginal",
    affected_slice_rule: "age > 40", affected_features_ranked: ["age"],
    affected_value_ranges: {}, slice_support: 0.5, diagnosis_stability: 0.9,
    drift_impact_score: 0.052, temporal_window: 1,
  },
  data_context: {
    model_id: "your-model-v1", model_uri: "", dataset_id: "your-dataset-v1",
    reference_dataset_uri: "", current_dataset_uri: "", target_column: "target",
    feature_columns: ["age", "income"], categorical_features: [],
    label_status: "AVAILABLE", input_representation: "model_ready",
    repair_metric: "f1", generator_epochs: 15,
  },
};

const artifactFields = {
  model: "model_uri", preprocessor: "preprocessing_uri",
  reference_dataset: "reference_dataset_uri", current_dataset: "current_dataset_uri",
  evaluation_dataset: "evaluation_dataset_uri", source_evaluation: "source_evaluation_uri",
};

export default function C3Page() {
  const info = useApi(() => api.c3Info(), []);
  const [mode, setMode] = useState("integrated");
  const [contract, setContract] = useSession("c3:contract:v2", JSON.stringify(example, null, 2));
  const [history, setHistory] = useSession("c3:history:v2", []);
  const [selected, setSelected] = useState(-1);
  const [busy, setBusy] = useState("");
  const [error, setError] = useState(null);
  const [scenario, setScenario] = useState("education_num_marginal");
  const scenarios = useApi(() => mode === "simulation" ? api.c3Scenarios("uci") : Promise.resolve([]), [mode]);
  const output = selected >= 0 ? history[selected] : history[history.length - 1];
  let parsed;
  try { parsed = JSON.parse(contract); } catch { parsed = null; }

  async function act(label, operation) {
    setBusy(label); setError(null);
    try { return await operation(); }
    catch (e) { setError(e.message || String(e)); }
    finally { setBusy(""); }
  }

  async function upload(role, file) {
    if (!file) return;
    await act(`Uploading ${role}`, async () => {
      if (!parsed) throw new Error("Correct the contract JSON before uploading artifacts.");
      const artifact = await api.c3UploadArtifact(role, file);
      const next = { ...parsed, data_context: { ...parsed.data_context,
        [artifactFields[role]]: artifact.uri,
        artifact_sha256: { ...parsed.data_context?.artifact_sha256, [role]: artifact.sha256 },
      } };
      if (role === "preprocessor") delete next.data_context.input_representation;
      if (next.c1_evidence && ["current_dataset", "reference_dataset"].includes(role)) {
        next.c1_evidence[artifactFields[role]] = artifact.uri;
      }
      setContract(JSON.stringify(next, null, 2));
    });
  }

  async function run() {
    await act("Repair running", async () => {
      const result = mode === "simulation"
        ? await api.c3DemoRepair({ dataset: "uci", scenario })
        : await api.c3IntegratedRepair(JSON.parse(contract));
      setHistory((previous) => [...previous, result]);
      setSelected(-1);
    });
  }

  const approved = output?.repair_package_status === "APPROVED";
  const utility = output?.utility_metrics;
  const validation = output?.validation_report;
  return <>
    <PageHeader eyebrow="Component 3 · Mathusan K (IT23174580)" title="C3 Synthetic Data Repair"
      subtitle="Repair the supplied binary classifier using the actual diagnosed data slice. Approve candidates only after independent validation."
      chip={`c3 v${info.data?.engine_version || "2.0.0"}`} />
    <section className="card stack">
      <div className="row" style={{ gap: 12 }}>
        <button className={`btn ${mode === "integrated" ? "primary" : ""}`} disabled={!!busy} onClick={() => setMode("integrated")}>Real C1/C2 integration</button>
        <button className={`btn ${mode === "simulation" ? "primary" : ""}`} disabled={!!busy} onClick={() => setMode("simulation")}>Research simulation</button>
      </div>
      {mode === "integrated" ? <>
        <p>Import the team's three-section contract or load a real upstream run. Supply the exact model, fitted preprocessing and labeled datasets described by that evidence.</p>
        <div className="row" style={{ gap: 12, flexWrap: "wrap" }}>
          <button className="btn" disabled={!!busy} onClick={() => act("Loading upstream", async () => {
            const data = await api.c3LatestUpstream();
            setContract(JSON.stringify(data, null, 2));
          })}>Load latest real C1/C2 inputs</button>
          <label className="btn">Import contract JSON
            <input aria-label="Import C1 C2 contract JSON" type="file" accept=".json" disabled={!!busy} style={{ display: "none" }}
              onChange={(event) => { const file = event.target.files?.[0]; if (file) act("Reading contract", async () => {
                const data = JSON.parse(await file.text()); setContract(JSON.stringify(data, null, 2));
              }); event.target.value = ""; }} />
          </label>
        </div>
        <div className="c3-gates-grid">
          {Object.entries(artifactFields).map(([role, field]) => <label key={role} className="field" style={{ minWidth: 0 }}>
            <span>{role.replaceAll("_", " ")}{["model", "reference_dataset", "current_dataset"].includes(role) ? "" : " (optional)"}</span>
            <input type="file" accept={["model", "preprocessor"].includes(role) ? ".joblib,.pkl,.pickle" : ".csv"}
              aria-label={`Upload ${role.replaceAll("_", " ")}`} disabled={!!busy}
              onChange={(event) => { upload(role, event.target.files?.[0]); event.target.value = ""; }} />
            <small className="muted" style={{ overflowWrap: "anywhere" }}>{parsed?.data_context?.[field] || "Use an uploaded artifact or an authorized local path in the contract."}</small>
          </label>)}
        </div>
        <p className="small muted">Upload trusted model artifacts from your team. Preprocessing is optional for a complete fitted Pipeline or explicitly model-ready features. Separate evaluation CSVs are recommended; otherwise C3 creates stratified holdouts. Slice rules use the supplied dataset's feature values.</p>
        <label className="field" htmlFor="c3-contract">C1 evidence, C2 diagnosis and model/data context</label>
        <textarea id="c3-contract" className="mono" rows={22} spellCheck={false} value={contract}
          style={{ width: "100%", resize: "vertical", padding: 12, boxSizing: "border-box" }}
          disabled={!!busy} onChange={(event) => setContract(event.target.value)} />
        {!parsed && <p role="alert">Contract JSON is invalid.</p>}
        {parsed && <KV rows={[
          ["Model", parsed.data_context?.model_id || "Missing"],
          ["Dataset", parsed.data_context?.dataset_id || "Missing"],
          ["Diagnosed slice", parsed.c2_diagnosis?.affected_slice_rule || parsed.c2_to_c3?.affected_slice_rule || "Missing"],
          ["Labels", parsed.data_context?.label_status || "Unknown: repair will abstain"],
        ]} />}
      </> : <>
        <p>This mode constructs simulated C1/C2 evidence for a controlled Adult scenario. Its results are labeled as simulation.</p>
        <select aria-label="Research scenario" value={scenario} disabled={!!busy} onChange={(event) => setScenario(event.target.value)}>
          {(scenarios.data || []).map((item) => <option key={item.id} value={item.id}>{item.name}</option>)}
        </select>
      </>}
      <button className="btn primary" type="button" disabled={!!busy || (mode === "integrated" && !parsed)} onClick={run}>
        {busy || (mode === "integrated" ? "Start repair with supplied inputs" : "Run research simulation")}
      </button>
      <ErrorBox error={error || info.error} />
    </section>
    {output && <section className="card stack" aria-live="polite">
      <Verdict title={output.execution_status || output.repair_package_status} tone={approved ? "ok" : "warn"}
        text={approved ? "Candidate passed validation and is ready for C4 review." : output.reason || "Candidate was not approved."}
        aside={output.mode === "simulation" ? "SIMULATION" : "Supplied artifacts"} />
      <KV rows={[
        ["Run", output.run_id], ["Generator executed", output.generator], ["Approved synthetic rows", output.n_synthetic_rows],
        ["Affected-slice metric", utility?.metric || "Not evaluated"],
        ["Affected-slice gain", utility?.utility_gain == null ? "Not evaluated" : `${(utility.utility_gain * 100).toFixed(2)} pp`],
        ["Real-only control", utility?.authentic_control_score == null ? "Not evaluated" : utility.authentic_control_score.toFixed(4)],
        ["Privacy", output.privacy_status], ["Target coverage", validation?.target_coverage == null ? "Not evaluated" : `${(validation.target_coverage * 100).toFixed(1)}%`],
      ]} />
      {approved && output.download_url && <a className="btn primary" href={output.download_url} download>Download approved repair CSV</a>}
      <JsonBox title="Validation evidence" data={validation} />
      <JsonBox title="Input artifacts and provenance" data={output.provenance} />
      <JsonBox title="C3 result" data={output} />
    </section>}
    {history.length > 0 && <section className="card stack">
      <h3>Repair history</h3>
      {history.map((item, index) => <button className="btn" key={item.run_id || index} onClick={() => setSelected(index)}>
        {item.run_id} · {item.execution_status || item.repair_package_status}
      </button>)}
    </section>}
    {info.data && <section className="card stack">
      <KV rows={[["Supported models", info.data.model_requirements], ["Supported data", info.data.dataset_requirements],
        ["Prescription", "Rule based, using diagnosis and feature types"],
        ["C2 platform implementation", info.data.upstream_sources?.c2], ["C4 platform implementation", info.data.upstream_sources?.c4]]} />
    </section>}
  </>;
}
