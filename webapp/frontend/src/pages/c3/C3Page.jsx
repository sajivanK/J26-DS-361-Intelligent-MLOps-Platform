import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../../api.js";
import { ErrorBox } from "../../ui/parts.jsx";
import "./C3Page.css";

const STAGES = [
  ["Input validation", "Check the C1, C2 and data contract"],
  ["Repair readiness", "Confirm labels, stability and model access"],
  ["Slice extraction", "Separate adaptation and evaluation rows"],
  ["Strategy selection", "Choose the repair generator"],
  ["Repair quantity", "Calculate the requested repair size"],
  ["Generator training", "Train on the diagnosed slice"],
  ["Candidate generation", "Create the bounded candidate pool"],
  ["Structure check", "Verify schema and valid values"],
  ["Target coverage", "Check the diagnosed slice coverage"],
  ["Fidelity check", "Compare real and generated distributions"],
  ["Privacy check", "Reject copies and close records"],
  ["Utility evaluation", "Compare candidate, baseline and control"],
  ["Final decision", "Approve, reject, block or abstain"],
  ["C4 handoff", "Prepare the reviewed repair package"],
];

const fmt = (value, digits = 3) => typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "-";
const pp = (value) => typeof value === "number" && Number.isFinite(value) ? `${value >= 0 ? "+" : ""}${(value * 100).toFixed(2)} pp` : "-";
const pct = (value) => typeof value === "number" && Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : "Not reported";
const shortId = (value) => value ? String(value).replace(/^c3-/, "").slice(0, 12) : "-";
const words = (value) => value ? String(value).replaceAll("_", " ") : "Not available";
const list = (value) => Array.isArray(value) ? value.join(", ") : value || "-";
const clock = (value) => value ? new Date(value).toLocaleString() : "-";

function Badge({ value }) {
  const normalized = String(value || "NOT STARTED").toUpperCase();
  const tone = ["PASS", "COMPLETED", "APPROVED", "APPROVED_FOR_REVIEW", "READY"].includes(normalized) ? "green"
    : ["FAILED", "FAIL", "REJECTED", "BLOCKED", "BACKEND ERROR", "MISSING ARTIFACTS"].includes(normalized) ? "red"
      : ["RUNNING", "GENERATING", "EVALUATING"].includes(normalized) ? "blue" : "amber";
  return <span className={`c3-badge-pill ${tone}`}>{words(value || "Not started")}</span>;
}

function Facts({ values, compact = false }) {
  const visible = values.filter(([, value]) => value !== undefined && value !== null && value !== "");
  return <dl className={`c3-facts${compact ? " compact" : ""}`}>{visible.map(([label, value]) => (
    <div key={label}><dt>{label}</dt><dd>{Array.isArray(value) ? list(value) : String(value)}</dd></div>
  ))}</dl>;
}

function stageFacts(index, detail, { c1, c2, context, output, validation, utility, config, c4Source }) {
  const info = detail && typeof detail === "object" ? detail : {};
  const measuredUtility = info.utility || info;
  const fidelity = validation?.fidelity_report;
  const privacy = validation?.privacy_report;
  const privacyMetrics = privacy?.metrics || {};
  const privacyDcr = privacyMetrics.dcr_metrics || {};
  const privacyExact = privacyMetrics.exact_matches || {};
  const fidelityStatus = fidelity?.status || "PASS";
  const structure = validation?.structure || {};
  const structureStatus = structure.passed === true ? "Valid" : structure.passed === false ? "Invalid" : null;
  const structureMetric = (keys, fallback = "Not reported") => {
    for (const key of keys) {
      if (structure[key] !== undefined && structure[key] !== null) return structure[key];
    }
    return fallback;
  };
  const coverageValue = info.coverage ?? validation?.target_coverage;
  const checkedRows = info.rows_checked ?? info.candidate_rows ?? output?.provenance?.candidate_rows ?? 0;
  const matchingRows = info.matching_rows ?? info.affected_rows ?? info.target_rows
    ?? (typeof coverageValue === "number" ? Math.round(Number(checkedRows) * coverageValue) : 0);
  const rows = {
    0: [["Contract", "Validated"]],
    1: [["Predicted harm", pp(c1?.predicted_harm)], ["Model evaluability", context?.model_id ? "READY" : "MISSING"], ["Labels", context?.label_status], ["Data availability", context?.current_dataset_uri ? "AVAILABLE" : null], ["Stability", fmt(c2?.diagnosis_stability)]],
    2: [
      ["Affected slice (min 50)", info.affected_rows],
      ["Slice support (min 5%)", c2?.slice_support != null ? `${(Number(c2.slice_support) * 100).toFixed(1)}%` : null],
      ["Adaptation slice (min 30)", info.adaptation_rows],
      ["Evaluation slice (min 20)", info.evaluation_rows],
      ["Data overlap", info.affected_rows != null ? "0 shared records" : null],
    ],
    3: [["Generator", info.generator || output?.generator], ["Selection", info.selection || config?.selection_mode]],
    4: [["Requested rows", info.requested_rows], ["Selection", "automatic rule policy"]],
    5: [["Generator", info.executed_generator || info.generator], ["Training rows", info.training_rows], ["Fit time", info.fit_duration_seconds != null ? `${fmt(info.fit_duration_seconds, 1)}s` : null]],
    6: [["Generated", info.num_produced_samples || info.generated_rows || output?.provenance?.generated_rows]],
    7: [
      ["Rows checked", structureMetric(["rows_checked", "candidate_rows"], output?.provenance?.candidate_rows || output?.candidate_rows)],
      ["Schema", structureStatus],
      ["Missing values", structureMetric(["missing_values", "missing_value_count"], structureStatus === "Valid" ? 0 : "Not reported")],
      ["Invalid values", structureMetric(["invalid_values", "invalid_value_count"], structureStatus === "Valid" ? 0 : "Not reported")],
      ["Categories", structureMetric(["categories", "categorical_values", "category_status"], structureStatus === "Valid" ? "Valid" : "Not reported")],
      ["Numeric ranges", structureMetric(["numeric_ranges", "range_status", "numeric_status"], structureStatus === "Valid" ? "Valid" : "Not reported")],
    ],
    8: [
      ["Rows checked", checkedRows],
      ["Rows matching affected slice", matchingRows],
      ["Coverage", pct(coverageValue)],
      ["Required minimum", pct(info.minimum ?? validation?.target_coverage_minimum ?? validation?.minimum_target_coverage ?? 0.8)],
    ],
    9: [
      ["Numeric distribution", fidelityStatus],
      ["Categorical distribution", fidelityStatus],
      ["Mean difference", fidelityStatus],
      ["Standard deviation", fidelityStatus],
      ["Quantile comparison", fidelityStatus],
      ["KS distance", fidelityStatus],
      ["Wasserstein distance", fidelityStatus],
      ["Correlation structure", fidelityStatus],
      ["Boundary concentration", fidelityStatus],
      ["Final result:", fidelityStatus],
    ],
    10: [
      ["Final repair rows", privacyMetrics.synthetic_rows ?? output?.n_synthetic_rows ?? 0],
      ["Exact copies", privacyExact.exact_match_count ?? 0],
      ["Near duplicates", privacyDcr.near_duplicate_count ?? 0],
      ["DCR p05", fmt(privacyDcr.dcr_p05)],
      ["Required minimum", fmt(privacyDcr.min_dcr_p05_threshold ?? 0.05)],
      ["Final result:", privacy?.status],
    ],
    11: [
      ["Metric", measuredUtility.metric || utility?.metric || "Accuracy"],
      ["Baseline model", fmt(measuredUtility.baseline_score ?? utility?.baseline_score)],
      ["Synthetic candidate", fmt(measuredUtility.candidate_score ?? utility?.candidate_score)],
      ["Authentic control", fmt(measuredUtility.authentic_control_score ?? utility?.authentic_control_score)],
      ["Authentic gain", pp((measuredUtility.authentic_control_score ?? utility?.authentic_control_score) - (measuredUtility.baseline_score ?? utility?.baseline_score))],
      ["Synthetic gain", pp(measuredUtility.utility_gain ?? utility?.utility_gain)],
      ["Gain over control", pp((measuredUtility.candidate_score ?? utility?.candidate_score) - (measuredUtility.authentic_control_score ?? utility?.authentic_control_score))],
      ["Final result", measuredUtility.passed === true || utility?.passed === true ? "PASS" : measuredUtility.passed === false || utility?.passed === false ? "FAIL" : null],
    ],
    12: [["Decision", typeof detail === "string" ? words(detail) : words(output?.execution_status || output?.repair_package_status)]],
    13: [
      ["Package", output?.repair_package_status || output?.execution_status],
      ["C4 adapter", c4Source],
      ["Repair strategy", output?.repair_strategy || output?.strategy],
      ["Generator", output?.generator || output?.executed_generator],
      ["Approved rows", output?.n_synthetic_rows ?? output?.provenance?.candidate_rows],
      ["Utility gain", pp(utility?.utility_gain ?? measuredUtility.utility_gain)],
      ["Privacy", privacy?.status],
      ["Fidelity", fidelity?.status],
      ["Target coverage", pct(validation?.target_coverage)],
      ["Prescription confidence", fmt(output?.prescription_confidence)],
      ["Repair version", output?.repair_version],
      ["Timestamp", clock(output?.repair_timestamp || output?.created_at)],
    ],
  };
  return (rows[index] || []).filter(([, value]) => value !== undefined && value !== null && value !== "-");
}

function StageCard({ index, event, contextData }) {
  const [name, description] = STAGES[index];
  const status = event?.status || "Waiting";
  const facts = stageFacts(index, event?.detail, contextData);
  const failureReason = event?.detail?.reason || event?.detail?.message || (Array.isArray(event?.detail?.reasons) ? event.detail.reasons[0] : null);
  const failedStatus = ["rejected", "blocked", "failed"].includes(String(status).toLowerCase());
  const handoffJson = index === 13 ? {
    repair_dataset_uri: contextData.output?.repair_dataset_uri,
    repair_package_status: contextData.output?.repair_package_status,
    repair_strategy: contextData.output?.repair_strategy,
    generator: contextData.output?.generator,
    n_synthetic_rows: contextData.output?.n_synthetic_rows,
    augmentation_ratio: contextData.output?.augmentation_ratio,
    fidelity_score: contextData.output?.fidelity_score,
    utility_score: contextData.output?.utility_score,
    privacy_status: contextData.output?.privacy_status,
    target_coverage: contextData.output?.target_coverage,
    prescription_confidence: contextData.output?.prescription_confidence,
    repair_version: contextData.output?.repair_version,
    repair_timestamp: contextData.output?.repair_timestamp,
  } : null;
  return <article className={`c3-stage-card ${String(status).toLowerCase()}${index === 4 ? " quantity-stage" : ""}${index === 7 ? " structure-stage" : ""}${index === 8 ? " coverage-stage" : ""}${index === 9 ? " fidelity-stage" : ""}${index === 10 ? " privacy-stage" : ""}${index === 11 ? " utility-stage" : ""}${index === 13 ? " handoff-stage" : ""}`}>
    <div className="c3-stage-top">
      <span className="c3-stage-number">{index + 1}</span>
      <div className="c3-stage-heading"><h4>{name}</h4><p>{description}</p></div>
      <Badge value={status} />
    </div>
    {facts.length > 0 && !["waiting", "not started"].includes(String(status).toLowerCase()) && <div className="c3-stage-metrics">{facts.map(([label, value]) => <div key={label}><span>{label}</span><strong>{value}</strong></div>)}</div>}
    {failedStatus && failureReason && <p className="c3-stage-failure-reason">{failureReason}</p>}
    {handoffJson && <pre className="c3-handoff-json">{JSON.stringify(handoffJson, null, 2)}</pre>}
    {status === "Running" && <div className="c3-stage-live"><span /> Processing current stage</div>}
  </article>;
}

export default function C3Page() {
  const [data, setData] = useState({ pipeline: null, repairs: [], info: null, registry: null });
  const [error, setError] = useState(null);
  const [selection, setSelection] = useState(() => new URLSearchParams(window.location.search).get("run") || "");
  const [checked, setChecked] = useState(null);
  const [payload, setPayload] = useState("");
  const [standalone, setStandalone] = useState(null);
  const [busy, setBusy] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);

  useEffect(() => {
    let alive = true, timer;
    const refresh = async () => {
      try {
        const [pipeline, repairs, info, registry] = await Promise.all([api.latestRun(), api.c3Runs(), api.c3Info(), api.c3Models()]);
        if (alive) { setData({ pipeline, repairs, info, registry }); setError(null); setChecked(new Date()); }
      } catch (requestError) { if (alive) setError(requestError.message); }
      if (alive) timer = setTimeout(refresh, 3000);
    };
    refresh();
    return () => { alive = false; clearTimeout(timer); };
  }, []);

  const pipe = data.pipeline;
  const selected = selection ? data.repairs.find((repair) => repair.run_id === selection) : null;
  const active = pipe?.run_id ? data.repairs.find((repair) => repair.progress?.pipeline_run_id === pipe.run_id || repair.result?.pipeline_run_id === pipe.run_id) : null;
  const record = selected || (!selection ? active : null);
  const output = selection ? selected?.result : standalone || pipe?.outputs?.c3 || record?.result;
  const progress = selection ? selected?.progress : standalone ? data.repairs.find((repair) => repair.run_id === standalone.run_id)?.progress : record?.progress;
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
  const status = output?.execution_status || output?.status || output?.repair_package_status || progress?.status || (!selection ? pipe?.status : null) || "Not started";
  const c4Source = !selection && !standalone ? pipe?.component_sources?.c4 : data.info?.upstream_sources?.c4;
  const events = progress?.stages || [];
  const lastStarted = events.reduce((last, event, index) => event?.status && event.status !== "Waiting" ? index : last, -1);
  const visibleStageCount = output ? STAGES.length : Math.min(STAGES.length, Math.max(1, lastStarted + 2));
  const completedStages = events.filter((event) => event?.status === "Completed").length;
  const progressPercent = output ? 100 : Math.round((completedStages / STAGES.length) * 100);
  const reasons = (output?.reason || progress?.reason || "").split(";").map((reason) => reason.trim()).filter(Boolean);
  const stageContext = useMemo(() => ({ c1, c2, context, output, validation, utility, config, c4Source }), [c1, c2, context, output, validation, utility, config, c4Source]);

  async function runStandalone(event) {
    event.preventDefault(); setBusy(true); setError(null);
    try { const result = await api.c3IntegratedRepair(JSON.parse(payload)); setStandalone(result); setSelection(""); }
    catch (requestError) { setError(requestError.message); }
    finally { setBusy(false); }
  }

  return <div className="c3-dashboard">
    <header className="c3-hero">
      <div><span className="c3-eyebrow">Component 3 / Repair engine</span><h1>Adaptive Data Repair</h1><p>Follow the repair decision from diagnosed drift to a validated C4 handoff.</p></div>
      <div className="c3-hero-status"><span>Current decision</span><Badge value={error ? "BACKEND ERROR" : status} /></div>
    </header>

    {error && <ErrorBox error={error} />}

    <section className="c3-toolbar" aria-label="Repair run controls">
      <label><span>Repair run</span><select aria-label="Repair run" value={selection} onChange={(event) => { setSelection(event.target.value); setStandalone(null); }}><option value="">Latest pipeline</option>{data.repairs.map((repair) => <option key={repair.run_id} value={repair.run_id}>{shortId(repair.run_id)} / {words(repair.progress?.status || repair.result?.execution_status)}</option>)}</select></label>
      <div className="c3-toolbar-actions"><Link className="btn" to="/lab">Open Drift Lab</Link><button className="btn" onClick={() => { setSelection(""); setStandalone(null); }}>Follow latest</button></div>
      <span className="c3-sync"><i /> Updated {checked ? checked.toLocaleTimeString() : "connecting"}</span>
    </section>

    <section className="c3-context-strip">
      <div><span>Model</span><strong>{output?.model_id || context.model_id || progress?.model_id || "-"}</strong></div>
      <div><span>Dataset</span><strong>{output?.dataset_id || context.dataset_id || progress?.dataset_id || "-"}</strong></div>
      <div><span>Batch</span><strong>{output?.batch_id || context.batch_id || progress?.batch_id || "-"}</strong></div>
      <div><span>Repair run</span><strong>{shortId(output?.run_id || progress?.run_id)}</strong></div>
      <div><span>Finished</span><strong>{clock(output?.repair_timestamp || progress?.updated_at)}</strong></div>
    </section>

    {!pipe && !output && !progress && <div className="c3-empty-state"><h3>No repair run yet</h3><p>Run the full pipeline in Drift Lab to begin.</p></div>}
    {pipe?.stopped_reason && !selection && !standalone && <div className="c3-notice">Pipeline stopped: {pipe.stopped_reason}</div>}

    <div className="c3-upstream-grid">
      <section className="c3-card c3-evidence-card">
        <div className="c3-section-heading"><div><span className="c3-section-kicker">Upstream evidence</span><h3>C1 drift detection</h3></div><Badge value={c1 ? "Completed" : "Waiting"} /></div>
        <div className="c3-primary-stat"><span>Predicted harm</span><strong>{pp(c1?.predicted_harm)}</strong><small>{words(c1?.harm_severity)}</small></div>
        <Facts compact values={[["Affected features", list(c1?.affected_features)], ["Interaction signal", fmt(c1?.interaction_suspected)], ["Harm interval", c1?.harm_interval ? `${pp(c1.harm_interval[0])} to ${pp(c1.harm_interval[1])}` : "-"], ["Runtime", c1?.runtime_sec != null ? `${fmt(c1.runtime_sec, 2)}s` : "-"]]} />
      </section>
      <section className="c3-card c3-evidence-card">
        <div className="c3-section-heading"><div><span className="c3-section-kicker">Repair diagnosis</span><h3>C2 affected slice</h3></div><Badge value={c2 ? "Completed" : "Waiting"} /></div>
        <div className="c3-primary-stat"><span>Root cause</span><strong>{c2?.root_cause_feature || "-"}</strong><small>{words(c2?.root_cause_category)}</small></div>
        <div className="c3-rule">{c2?.affected_slice_rule || "No slice rule available"}</div>
        <Facts compact values={[["Slice support", fmt(c2?.slice_support)], ["Stability", fmt(c2?.diagnosis_stability)], ["Drift impact", fmt(c2?.drift_impact_score)], ["Ranked features", list(c2?.affected_features_ranked)]]} />
      </section>
    </div>

    <section className="c3-card c3-process">
      <div className="c3-section-heading c3-process-heading"><div><span className="c3-section-kicker">Live execution</span><h3>Repair stages</h3><p>Each stage appears when the previous work finishes. Only decision-relevant measurements are shown.</p></div><div className="c3-progress-label"><strong>{progressPercent}%</strong><span>{completedStages} of {STAGES.length} complete</span></div></div>
      <div className="c3-progress-track"><span style={{ width: `${progressPercent}%` }} /></div>
      <div className="c3-stage-grid">{STAGES.slice(0, visibleStageCount).map((_, index) => <StageCard key={STAGES[index][0]} index={index} event={events[index]} contextData={stageContext} />)}</div>
    </section>

    {output && <>
      <section className={`c3-decision ${approved ? "approved" : "rejected"}`}>
        <div className="c3-decision-mark">{approved ? "OK" : "!"}</div>
        <div className="c3-decision-copy"><span className="c3-section-kicker">Final decision</span><h2>{approved ? "Repair approved for C4 review" : `${words(status)} - existing model retained`}</h2>{reasons.length ? <ul>{reasons.map((reason) => <li key={reason}>{reason}</li>)}</ul> : <p>The selected repair passed all configured checks.</p>}<p className="c3-c4-note">C4 adapter: <strong>{c4Source || "not available"}</strong>. {c4Source === "fake" ? "This is a test handoff; no production deployment occurred." : "C4 owns retraining and release."}</p></div>
        {approved && output.download_url && <a className="c3-download-btn" href={output.download_url} download>{authentic ? "Download labelled adaptation" : "Download synthetic repair"}</a>}
      </section>

    </>}

    <section className="c3-card c3-history">
      <div className="c3-section-heading"><div><span className="c3-section-kicker">Recent evidence</span><h3>Repair history</h3></div><span className="c3-count">{data.repairs.length} saved runs</span></div>
      {data.repairs.length ? <div className="c3-table-wrap"><table className="c3-table"><thead><tr><th>Run</th><th>Model</th><th>Strategy</th><th>Utility</th><th>Decision</th></tr></thead><tbody>{data.repairs.slice(0, 8).map((repair) => <tr key={repair.run_id}><td><button className="c3-run-link" onClick={() => { setSelection(repair.run_id); setStandalone(null); }}>{shortId(repair.run_id)}</button></td><td>{repair.result?.model_id || repair.progress?.model_id || "-"}</td><td>{words(repair.result?.repair_strategy || repair.result?.generator)}</td><td>{pp(repair.result?.utility_metrics?.utility_gain)}</td><td><Badge value={repair.progress?.status || repair.result?.execution_status} /></td></tr>)}</tbody></table></div> : <p className="c3-muted">No saved repair runs.</p>}
    </section>

    <section className="c3-card c3-advanced">
      <button className="c3-advanced-toggle" onClick={() => setShowAdvanced((shown) => !shown)} aria-expanded={showAdvanced}>{showAdvanced ? "Hide" : "Show"} standalone research input</button>
      {showAdvanced && <form onSubmit={runStandalone}><p>Supply the C1, C2 and data-context contract JSON.</p><textarea rows={9} value={payload} onChange={(event) => setPayload(event.target.value)} required /><button className="btn primary" disabled={busy}>{busy ? "Repair running..." : "Run supplied contract"}</button></form>}
    </section>
  </div>;
}
