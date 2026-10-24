import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../../api.js";
import { ErrorBox } from "../../ui/parts.jsx";
import "./C3Page.css";

const STAGES = ["Input Validation", "Repair Readiness", "Slice Extraction", "Generator Prescription", "Repair Quantity Calculation", "Generator Training", "Synthetic Data Generation", "Structural Validation", "Target Coverage Validation", "Fidelity Validation", "Privacy Validation", "Utility Evaluation", "Final Decision", "C4 Package Preparation"];
const fmt = (v, digits = 3) => typeof v === "number" && Number.isFinite(v) ? v.toFixed(digits) : "—";
const text = (v) => v === undefined || v === null || v === "" ? "—" : typeof v === "object" ? JSON.stringify(v) : String(v);
const pp = (v) => typeof v === "number" ? `${v >= 0 ? "+" : ""}${(v * 100).toFixed(2)} pp` : "—";
function Badge({ value }) {
  const good = ["PASS", "Completed", "APPROVED", "APPROVED_FOR_REVIEW"].includes(value);
  return <span className={`c3-badge-pill ${good ? "green" : "amber"}`}>{value || "NOT EVALUATED"}</span>;
}
function Fields({ values }) {
  return <dl className="c3-facts">{values.map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{text(value)}</dd></div>)}</dl>;
}

export default function C3Page() {
  const [data, setData] = useState({ pipeline: null, repairs: [], info: null, registry: null });
  const [error, setError] = useState(null);
  const [selection, setSelection] = useState(() => new URLSearchParams(window.location.search).get("run") || "");
  const [checked, setChecked] = useState(null);
  const [payload, setPayload] = useState("");
  const [standalone, setStandalone] = useState(null);
  const [busy, setBusy] = useState(false);
  useEffect(() => {
    let alive = true, timer;
    const refresh = async () => {
      try {
        const [pipeline, repairs, info, registry] = await Promise.all([api.latestRun(), api.c3Runs(), api.c3Info(), api.c3Models()]);
        if (alive) { setData({ pipeline, repairs, info, registry }); setError(null); setChecked(new Date().toISOString()); }
      } catch (e) { if (alive) setError(e.message); }
      if (alive) timer = setTimeout(refresh, 3000);
    };
    refresh();
    return () => { alive = false; clearTimeout(timer); };
  }, []);

  const pipe = data.pipeline;
  const selected = selection ? data.repairs.find(r => r.run_id === selection) : null;
  const active = pipe?.run_id ? data.repairs.find(r => r.progress?.pipeline_run_id === pipe.run_id || r.result?.pipeline_run_id === pipe.run_id) : null;
  const record = selected || (!selection ? active : null);
  // Always bind evidence, output and preview to one run. A newer clean run has no older C3 output.
  const output = selection ? selected?.result : standalone || pipe?.outputs?.c3 || record?.result;
  const progress = selection ? selected?.progress : standalone ? data.repairs.find(r => r.run_id === standalone.run_id)?.progress : record?.progress;
  const provenance = output?.provenance || {};
  const context = output ? provenance.data_context || (selection ? {} : pipe?.context) || {} : pipe?.context || {};
  const c1 = output?.upstream_evidence?.c1 || (!selection && !standalone ? pipe?.outputs?.c1 : null);
  const c2 = output?.upstream_evidence?.c2 || (!selection && !standalone ? pipe?.outputs?.c2_to_c3 : null);
  const validation = output?.validation_report || {};
  const fidelity = validation.fidelity_report;
  const privacy = validation.privacy_report;
  const utility = output?.utility_metrics || {};
  const thresholds = provenance.quality_thresholds || {};
  const config = provenance.generator_configuration || {};
  const approved = output?.repair_package_status === "APPROVED";
  const authentic = output?.repair_strategy === "AUTHENTIC_WEIGHTED_ADAPTATION";
  const status = output?.execution_status || output?.status || output?.repair_package_status || progress?.status || (!selection ? pipe?.status : null) || "NO RUNS AVAILABLE";
  const columns = output?.preview_columns || [];
  const rows = output?.preview_rows || [];
  const c4Source = !selection && !standalone ? pipe?.component_sources?.c4 : data.info?.upstream_sources?.c4;
  const sourceNote = !selection && !standalone ? `C1: ${pipe?.component_sources?.c1 || "not run"}; C2: ${pipe?.component_sources?.c2 || "not run"}` : "Evidence saved with this repair";

  async function runStandalone(e) {
    e.preventDefault(); setBusy(true); setError(null);
    try { const result = await api.c3IntegratedRepair(JSON.parse(payload)); setStandalone(result); setSelection(""); }
    catch (e) { setError(e.message); }
    finally { setBusy(false); }
  }

  return <div className="c3-dashboard">
    <div className="c3-header-row"><div className="c3-header-titles"><h1>C3 Adaptive Data Repair</h1><p>Validated synthetic or authentic adaptation for the exact monitored model, batch and diagnosed slice.</p></div><Badge value={error ? "BACKEND ERROR" : status} /></div>
    {error && <ErrorBox error={error} />}
    <div className="c3-header-meta">
      <label>View run <select aria-label="Repair run" value={selection} onChange={e => { setSelection(e.target.value); setStandalone(null); }}><option value="">Latest pipeline</option>{data.repairs.map(r => <option key={r.run_id} value={r.run_id}>{r.run_id} · {r.progress?.status || r.result?.execution_status}</option>)}</select></label>
      <Link className="btn" to="/lab">Open Drift Lab</Link>
      <button className="btn" onClick={() => { setSelection(""); setStandalone(null); }}>Follow latest pipeline</button>
      <span className="small muted">Backend checked: {checked || "Connecting…"}</span>
    </div>
    <div className="c3-card"><Fields values={[["Model", output?.model_id || context.model_id || progress?.model_id], ["Dataset", output?.dataset_id || context.dataset_id || progress?.dataset_id], ["Batch", output?.batch_id || context.batch_id || progress?.batch_id], ["Pipeline run", output?.pipeline_run_id || progress?.pipeline_run_id || (!selection ? pipe?.run_id : null)], ["Repair run", output?.run_id || progress?.run_id], ["Result timestamp", output?.repair_timestamp || progress?.updated_at]]} /></div>
    {!pipe && !output && !progress && <div className="empty">No runs available. Run the full pipeline in Drift Lab or supply a standalone contract below.</div>}
    {pipe?.stopped_reason && !selection && !standalone && <div className="c3-card">Pipeline: {pipe.stopped_reason}</div>}
    <div className="c3-upstream-grid">
      <section className="c3-card"><h3>C1 drift evidence</h3><p className="small muted">{sourceNote}</p><Fields values={[["Evidence ID", c1?.evidence_version], ["Drift detected", c1?.drift_detected], ["Predicted harm", pp(c1?.predicted_harm)], ["Severity", c1?.harm_severity], ["Affected features", c1?.affected_features], ["Interaction signal", c1?.interaction_suspected], ["Harm interval", c1?.harm_interval], ["Execution seconds", c1?.runtime_sec]]} /><details><summary>Detector results</summary><pre>{JSON.stringify(c1?.detector_scores || {}, null, 2)}</pre></details></section>
      <section className="c3-card"><h3>C2 diagnosis</h3><Fields values={[["Root cause", c2?.root_cause_feature], ["Category", c2?.root_cause_category], ["Exact slice rule", c2?.affected_slice_rule], ["Slice support", c2?.slice_support], ["Observed support", provenance.observed_slice_support], ["Stability", c2?.diagnosis_stability], ["Ranked features", c2?.affected_features_ranked], ["Drift impact", c2?.drift_impact_score]]} /></section>
    </div>
    <section className="c3-card"><h3>Actual execution stages</h3><p className="small muted">Stages update from backend events. Waiting means the stage has not executed. Training loss and epoch progress appear only when the generator reports them.</p><ol className="c3-stage-list">{STAGES.map((name, i) => {
      const event = progress?.stages?.[i];
      return <li key={name}><span>{name}</span><Badge value={event?.status || "Waiting"} />{event?.detail != null && <details><summary>Measured details</summary><pre>{JSON.stringify(event.detail, null, 2)}</pre></details>}</li>;
    })}</ol></section>
    <div className="c3-upstream-grid">
      <section className="c3-card"><h3>Generator and repair quantity</h3><Fields values={[["Approved repair strategy", output?.repair_strategy], ["Executed generator", output?.generator], ["Selection", config.selection_mode], ["Policy", provenance.prescription_policy], ["Prescription confidence", output ? provenance.prescription_confidence : null], ["Epochs", config.epochs], ["Training batch size", config.batch_size], ["Numeric features generated as discrete values", config.discrete_numeric_features], ["Training slice rows", provenance.partition_rows?.generator_training_slice], ["Requested ratio", provenance.requested_augmentation_ratio], ["Requested rows", config.requested_rows], ["Generated pool rows", provenance.generated_rows], ["Candidate rows after privacy screening", provenance.candidate_rows], ["Discarded too-close rows", provenance.privacy_sampling?.discarded_too_close_rows], ["Approved synthetic rows", output?.n_synthetic_rows]]} /><p className="small muted">Labels: {context.label_status || "not specified"}. Authentic adaptation labels and independent evaluation labels are required. Drift Lab labels are withheld from C1/C2.</p><details><summary>Partitions, provenance and artifact hashes</summary><pre>{JSON.stringify(provenance, null, 2)}</pre></details></section>
      <section className="c3-card"><h3>Validation scorecard</h3><div className="c3-preview-scroll"><table className="c3-table"><thead><tr><th>Gate</th><th>Measured</th><th>Required</th><th>Result</th></tr></thead><tbody>
        <tr><td>Structure</td><td>{validation.structure ? "Schema checked" : "—"}</td><td>Exact schema, finite values, valid categories</td><td><Badge value={validation.structure ? validation.structure.passed ? "PASS" : "FAIL" : null} /></td></tr>
        <tr><td>Slice coverage</td><td>{fmt(validation.target_coverage)}</td><td>{text(thresholds.target_coverage_min)}</td><td><Badge value={validation.target_coverage == null ? null : validation.target_coverage >= thresholds.target_coverage_min ? "PASS" : "FAIL"} /></td></tr>
        <tr><td>Fidelity (mean KS)</td><td>{authentic ? "Not applicable (real data)" : fmt(fidelity?.metrics?.continuous_metrics_summary?.mean_ks_statistic)}</td><td>≤ {text(thresholds.fidelity_ks_max)}</td><td><Badge value={authentic ? "NOT APPLICABLE" : fidelity?.status} /></td></tr>
        <tr><td>Privacy (DCR p05)</td><td>{authentic ? "Not applicable (real data)" : fmt(privacy?.metrics?.dcr_metrics?.dcr_p05)}</td><td>≥ {text(thresholds.privacy_dcr_min)}; no exact copies</td><td><Badge value={authentic ? "PASS" : privacy?.status} /></td></tr>
        <tr><td>Utility ({utility.metric || "not evaluated"})</td><td>{pp(utility.utility_gain)}</td><td>≥ {text(thresholds.minimum_utility_gain)} gain, bootstrap and retention checks</td><td><Badge value={validation.utility == null ? null : validation.utility.passed ? "PASS" : "FAIL"} /></td></tr>
      </tbody></table></div><details><summary>Full validation measurements and failure reasons</summary><pre>{JSON.stringify(validation, null, 2)}</pre></details></section>
    </div>
    <section className="c3-card"><h3>Model utility comparison</h3><Fields values={[["Metric", utility.metric], ["Original affected slice", fmt(utility.baseline_score)], ["Authentic weighted candidate", fmt(utility.authentic_control_score)], ["Approved candidate", fmt(utility.candidate_score)], ["Gain over original", pp(utility.utility_gain)], ["Gain over authentic candidate", !authentic && typeof utility.candidate_score === "number" ? pp(utility.candidate_score - utility.authentic_control_score) : null], ["Paired 95% interval", utility.bootstrap_ci_95], ["Sample weights", utility.sample_weight_policy], ["Source retention", output?.source_retention_metrics], ["Overall current before", utility.overall_before], ["Overall current after", utility.overall_after]]} /></section>
    <section className="c3-card"><h3>Final decision and C4 handoff</h3><Badge value={status} /><p>{output?.reason || progress?.reason || (output ? "All configured gates passed; candidate ready for review." : "Waiting for actual repair results.")}</p><p>C4 adapter: <b>{c4Source || "not available"}</b>. {c4Source === "fake" ? "Stub handoff only. Real C4 retraining and production release are not connected." : "C3 approval is a candidate review decision; deployment belongs to C4."}</p>{approved && output?.download_url && <a className="btn primary" href={output.download_url} download>{authentic ? "Download approved labelled adaptation CSV" : `Download approved synthetic CSV (${output.n_synthetic_rows} rows)`}</a>}<details><summary>C3-to-C4 contract</summary><pre>{output ? JSON.stringify(Object.fromEntries(["repair_dataset_uri", "repair_package_status", "repair_strategy", "generator", "n_synthetic_rows", "augmentation_ratio", "fidelity_score", "utility_score", "privacy_status", "target_coverage", "prescription_confidence", "repair_version", "repair_timestamp"].map(k => [k, output[k]])), null, 2) : "No contract available."}</pre></details></section>
    <section className="c3-card"><h3>{authentic ? "Authentic adaptation artifact preview" : "Synthetic artifact preview"}</h3>{rows.length ? <div className="c3-preview-scroll"><table className="c3-table"><thead><tr>{columns.map(c => <th key={c}>{c}</th>)}</tr></thead><tbody>{rows.map((row, i) => <tr key={i}>{columns.map(c => <td key={c}>{text(row[c])}</td>)}</tr>)}</tbody></table></div> : <p className="muted">No approved artifact preview for this run.</p>}</section>
    <section className="c3-card"><h3>Registered model artifacts</h3>{data.registry?.error && <p>{data.registry.error}</p>}{data.registry?.models?.map(m => <div key={m.model_id}><b>{m.model_id}</b> · {m.dataset_id} · {m.algorithm || "Algorithm not loaded"} · <Badge value={m.ready ? "ARTIFACTS AVAILABLE" : "MISSING ARTIFACTS"} /><Fields values={Object.entries(m.artifacts).map(([k, a]) => [k, `${a.path || "not configured"} (${a.available ? "available" : "missing"})`])} /></div>)}</section>
    <section className="c3-card"><h3>Persisted repair history</h3><p className="small muted">Select a saved run above to inspect its own evidence, stages and artifact. Historical runs retain the policy used at the time.</p>{data.repairs.length ? <div className="c3-preview-scroll"><table className="c3-table"><thead><tr><th>Run</th><th>Model</th><th>Generator</th><th>Generated rows</th><th>Utility gain</th><th>Status</th></tr></thead><tbody>{data.repairs.map(r => <tr key={r.run_id}><td><button className="btn small" onClick={() => { setSelection(r.run_id); setStandalone(null); }}>{r.run_id}</button></td><td>{text(r.result?.model_id || r.progress?.model_id)}</td><td>{text(r.result?.generator)}</td><td>{text(r.result?.provenance?.generated_rows)}</td><td>{pp(r.result?.utility_metrics?.utility_gain)}</td><td><Badge value={r.progress?.status || r.result?.execution_status} /></td></tr>)}</tbody></table></div> : <p>No saved repair runs available.</p>}</section>
    <details className="c3-card"><summary>Standalone research mode — same real repair engine</summary><p>Supply C1 evidence, C2 diagnosis and exact model/data context. This runs real generation and validation.</p><form onSubmit={runStandalone}><label htmlFor="c3-contract">Three-section contract JSON</label><textarea id="c3-contract" rows={10} value={payload} onChange={e => setPayload(e.target.value)} style={{ width: "100%", fontFamily: "var(--mono)" }} required /><button className="btn primary" disabled={busy}>{busy ? "Repair running…" : "Run supplied contract"}</button></form></details>
  </div>;
}
