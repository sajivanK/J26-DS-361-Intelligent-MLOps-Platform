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

function stageFacts(index, detail, { c2, context, output, validation, utility, config, c4Source }) {
  const info = detail && typeof detail === "object" ? detail : {};
  const measuredUtility = info.utility || info;
  const fidelity = validation?.fidelity_report;
  const privacy = validation?.privacy_report;
  const rows = {
    0: [["Contract", "Validated"]],
    1: [["Labels", context?.label_status], ["Stability", fmt(c2?.diagnosis_stability)]],
    2: [["Affected", info.affected_rows], ["Adaptation", info.adaptation_rows], ["Evaluation", info.evaluation_rows]],
    3: [["Generator", info.generator || output?.generator], ["Selection", info.selection || config?.selection_mode]],
    4: [["Requested", info.requested_rows], ["Ratio", info.augmentation_ratio], ["Maximum", info.max_rows]],
    5: [["Generator", info.executed_generator || info.generator], ["Training rows", info.training_rows], ["Fit time", info.fit_duration_seconds != null ? `${fmt(info.fit_duration_seconds, 1)}s` : null]],
    6: [["Generated", info.num_produced_samples || info.generated_rows || output?.provenance?.generated_rows]],
    7: [["Schema", validation?.structure?.passed === true ? "Valid" : validation?.structure?.passed === false ? "Invalid" : null]],
    8: [["Coverage", fmt(info.coverage ?? validation?.target_coverage)], ["Minimum", fmt(info.minimum)]],
    9: [["Result", fidelity?.status], ["Mean KS", fmt(fidelity?.metrics?.continuous_metrics_summary?.mean_ks_statistic)]],
    10: [["Result", privacy?.status], ["DCR p05", fmt(privacy?.metrics?.dcr_metrics?.dcr_p05)], ["Exact copies", privacy?.metrics?.exact_matches?.exact_match_count]],
    11: [["Gain", pp(measuredUtility.utility_gain ?? utility?.utility_gain)], ["Candidate", fmt(measuredUtility.candidate_score ?? utility?.candidate_score)], ["Baseline", fmt(measuredUtility.baseline_score ?? utility?.baseline_score)]],
    12: [["Decision", typeof detail === "string" ? words(detail) : words(output?.execution_status || output?.repair_package_status)]],
    13: [["Package", output?.repair_package_status], ["C4 adapter", c4Source]],
  };
  return (rows[index] || []).filter(([, value]) => value !== undefined && value !== null && value !== "-");
}

function StageCard({ index, event, contextData }) {
  const [name, description] = STAGES[index];
  const status = event?.status || "Waiting";
  const facts = stageFacts(index, event?.detail, contextData);
  return <article className={`c3-stage-card ${String(status).toLowerCase()}`}>
    <div className="c3-stage-top">
      <span className="c3-stage-number">{index + 1}</span>
      <div className="c3-stage-heading"><h4>{name}</h4><p>{description}</p></div>
      <Badge value={status} />
    </div>
    {facts.length > 0 && <div className="c3-stage-metrics">{facts.map(([label, value]) => <div key={label}><span>{label}</span><strong>{value}</strong></div>)}</div>}
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
  const stageContext = useMemo(() => ({ c2, context, output, validation, utility, config, c4Source }), [c2, context, output, validation, utility, config, c4Source]);

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
      <div className="c3-result-grid">
        <section className="c3-card">
          <div className="c3-section-heading"><div><span className="c3-section-kicker">Selected path</span><h3>Repair strategy</h3></div><Badge value={output.repair_package_status} /></div>
          <div className="c3-strategy-name">{words(output.repair_strategy || "No repair")}</div>
          <Facts values={[["Generator", output.generator], ["Requested synthetic rows", config.requested_rows], ["Safe candidate rows", provenance.candidate_rows], ["Approved synthetic rows", output.n_synthetic_rows], ["Prescription confidence", fmt(output.prescription_confidence ?? provenance.prescription_confidence)], ["Label status", context.label_status]]} />
        </section>

        <section className="c3-card">
          <div className="c3-section-heading"><div><span className="c3-section-kicker">Safety gates</span><h3>Validation scorecard</h3></div></div>
          <div className="c3-gate-list">
            {[
              ["Structure", validation.structure?.passed ? "Schema valid" : "Not passed", validation.structure?.passed ? "PASS" : validation.structure ? "FAIL" : null],
              ["Target coverage", authentic ? "Not applicable" : `${fmt(validation.target_coverage)} / ${fmt(thresholds.target_coverage_min)}`, authentic ? "N/A" : validation.target_coverage >= thresholds.target_coverage_min ? "PASS" : "FAIL"],
              ["Fidelity", authentic ? "Real data" : `KS ${fmt(fidelity?.metrics?.continuous_metrics_summary?.mean_ks_statistic)}`, authentic ? "N/A" : fidelity?.status],
              ["Privacy", authentic ? "Real labelled data" : `DCR ${fmt(privacy?.metrics?.dcr_metrics?.dcr_p05)}`, authentic ? "PASS" : privacy?.status],
              ["Utility", `${pp(utility.utility_gain)} gain`, validation.utility?.passed ? "PASS" : "FAIL"],
            ].map(([label, measured, result]) => <div className="c3-gate" key={label}><div><strong>{label}</strong><span>{measured}</span></div><Badge value={result || "Not evaluated"} /></div>)}
          </div>
        </section>
      </div>

      <section className="c3-card c3-utility">
        <div className="c3-section-heading"><div><span className="c3-section-kicker">Independent evaluation</span><h3>Model utility</h3></div><Badge value={validation.utility?.passed ? "PASS" : "FAIL"} /></div>
        <div className="c3-score-flow"><div><span>Original affected slice</span><strong>{fmt(utility.baseline_score)}</strong></div><b>-&gt;</b><div><span>Repair candidate</span><strong>{fmt(utility.candidate_score)}</strong></div><b>-&gt;</b><div className={utility.utility_gain >= (thresholds.minimum_utility_gain || .01) ? "positive" : "negative"}><span>Measured gain</span><strong>{pp(utility.utility_gain)}</strong></div></div>
        <Facts compact values={[["Required gain", `at least ${pp(thresholds.minimum_utility_gain || .01)}`], ["95% confidence interval", utility.bootstrap_ci_95 ? `${pp(utility.bootstrap_ci_95[0])} to ${pp(utility.bootstrap_ci_95[1])}` : "-"], ["Source retention", output.source_retention_metrics?.delta != null ? pp(output.source_retention_metrics.delta) : "-"], ["Overall performance", utility.overall_after != null ? `${fmt(utility.overall_before)} to ${fmt(utility.overall_after)}` : "-"]]} />
      </section>

      <section className={`c3-decision ${approved ? "approved" : "rejected"}`}>
        <div className="c3-decision-mark">{approved ? "OK" : "!"}</div>
        <div className="c3-decision-copy"><span className="c3-section-kicker">Final decision</span><h2>{approved ? "Repair approved for C4 review" : `${words(status)} - existing model retained`}</h2>{reasons.length ? <ul>{reasons.map((reason) => <li key={reason}>{reason}</li>)}</ul> : <p>The selected repair passed all configured checks.</p>}<p className="c3-c4-note">C4 adapter: <strong>{c4Source || "not available"}</strong>. {c4Source === "fake" ? "This is a test handoff; no production deployment occurred." : "C4 owns retraining and release."}</p></div>
        {approved && output.download_url && <a className="c3-download-btn" href={output.download_url} download>{authentic ? "Download labelled adaptation" : "Download synthetic repair"}</a>}
      </section>

      <div className="c3-secondary-grid">
        <section className="c3-card"><div className="c3-section-heading"><div><span className="c3-section-kicker">C4 package</span><h3>Handoff summary</h3></div></div><Facts compact values={[["Status", output.repair_package_status], ["Strategy", words(output.repair_strategy)], ["Generator", output.generator], ["Synthetic rows", output.n_synthetic_rows], ["Privacy", output.privacy_status], ["Utility gain", pp(output.utility_score)]]} /></section>
        <section className="c3-card"><div className="c3-section-heading"><div><span className="c3-section-kicker">Model registry</span><h3>Registered artifacts</h3></div></div>{data.registry?.error && <p className="c3-inline-error">{data.registry.error}</p>}{data.registry?.models?.map((model) => <div className="c3-model-row" key={model.model_id}><div><strong>{model.model_id}</strong><span>{model.dataset_id} / {model.algorithm || "Algorithm unavailable"}</span></div><Badge value={model.ready ? "Ready" : "Missing artifacts"} /></div>)}</section>
      </div>
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
