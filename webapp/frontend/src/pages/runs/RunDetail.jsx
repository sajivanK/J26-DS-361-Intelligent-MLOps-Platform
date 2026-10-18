import { Link, useParams } from "react-router-dom";
import { api } from "../../api.js";
import { useApi } from "../../hooks.js";
import { c1View, isClassicAlarm, OUTCOME_CLASS, OUTCOME_HELP, outcomeOf, sg, verdict } from "../../ui/c1format.js";
import { ErrorBox, JsonBox, KV } from "../../ui/parts.jsx";
import PipelineSteps from "../../ui/PipelineSteps.jsx";
import TruthVsC1 from "../../ui/TruthVsC1.jsx";
import { SOURCE_LABEL, TYPE_LABEL, when } from "./RunHistory.jsx";
import "./Runs.css";

const sevOf = (pp) => (pp < 1 ? "none" : pp < 3 ? "low" : pp < 7 ? "moderate" : pp < 15 ? "high" : "critical");
const injectedExtent = (t) => ({ none: "none", interaction: "none", marginal_single: "single", combined: "single", marginal_multi: "multi" }[t]);
const injectedInteraction = (t) => t === "interaction" || t === "combined";
const DETECTORS = [["KS", "ks_max", "ks", 0.75], ["Chi-squared", "chi2_max", "chi2", 1], ["PSI", "psi_max", "psi", 10],
  ["MMD", "mmd", "mmd", 0.2], ["LSDD", "lsdd", "lsdd", 0.4], ["Classifier (AUC)", "clf_auc", "classifier", 1]];
const STAT_LABEL = {
  conf_mean: "Mean confidence", conf_std: "Confidence spread (std)",
  conf_pct_lt_06: "Share of unsure predictions (confidence < 0.6)",
  mean_pos_prob: "Mean predicted probability of the positive class", pos_prob_std: "Spread of that probability",
  pct_pos_prob_lt_06: "Share with positive-class probability < 0.6",
};

function title(rec) {
  const inj = rec.injection;
  if (!inj) return `Uploaded batch: ${rec.upload?.file || "?"}`;
  if (inj.drift_type === "none") return "Clean batch (no drift)";
  const t = TYPE_LABEL[inj.drift_type];
  const f = inj.drift_type === "interaction" ? inj.features.join(" × ")
    : inj.drift_type === "combined" ? `${inj.features[0]} + ${inj.features[1]} × ${inj.features[2]}` : inj.features.join(", ");
  return `${t[0].toUpperCase()}${t.slice(1)}: ${f}`;
}

function rerunLink(inj) {
  const q = new URLSearchParams({ ds: inj.dataset, type: inj.drift_type, f: inj.features.join(","),
    mag: String(inj.magnitude || 0.8), bs: String(inj.batch_size), seed: String(inj.seed) });
  return `/lab?${q}`;
}

export default function RunDetail() {
  const { id } = useParams();
  const res = useApi(() => api.run(id), [id]);
  if (res.loading) return <p className="muted">Loading run {id}…</p>;
  if (res.error) return (<><div className="run-crumbs"><Link to="/runs">Run history</Link> › run {id}</div><ErrorBox error={res.error} /></>);

  const rec = res.data;
  const v = c1View(rec.c1);
  const lab = rec.source === "drift_lab";
  const inj = rec.injection;
  const truth = rec.truth;
  const t = truth?.true_harm_pp ?? null;
  const metric = (truth?.harm_metric || rec.c1?.harm_metric) === "auc_drop" ? "AUC" : "Accuracy";
  const out = v ? outcomeOf(t, v.sev) : "no truth";
  const inside = v && t != null ? t >= v.lo && t <= v.hi : null;

  return (
    <>
      <div className="run-crumbs"><Link to="/runs">Run history</Link> › run {rec.run_id}</div>
      <div className="head">
        <div className="titles">
          <span className="eyebrow">{SOURCE_LABEL[rec.source] || rec.source} run · {rec.mode === "pipeline" ? "full pipeline" : "C1 only"} · {when(rec.saved_at)}</span>
          <h1>{title(rec)}</h1>
        </div>
        <div className="run-actions">
          {lab && <Link className="btn primary" to={rerunLink(inj)}>Run again with these settings</Link>}
          <a className="btn" href={api.runFileUrl(rec.run_id)} download={`run_${rec.run_id}.json`}>Download run (JSON)</a>
          {lab && (rec.batch_csv
            ? <a className="btn" href={api.runBatchUrl(rec.run_id)} download>Download batch (CSV)</a>
            : <button className="btn" type="button" disabled title="The batch CSV is saved only when 'Save the batch CSV' was ticked">Download batch (CSV)</button>)}
        </div>
      </div>

      {!v ? <ErrorBox error={`C1 did not return an output for this run (${rec.pipeline?.stopped_reason || "unknown reason"}).`} /> : (<>
        <Verdict v={v} out={out} />
        {v.warnings.map((w) => <div key={w} className="warnline">{w}</div>)}

        <div className="grid2">
          {lab ? (
            <div className="card stack"><h3>What was injected</h3>
              <KV rows={[["Dataset", inj.dataset === "uci" ? "UCI Adult" : "IEEE-CIS"], ["Drift type", TYPE_LABEL[inj.drift_type]],
                ["Features", inj.features.join(", ") || "-"], ["Strength", inj.drift_type === "none" ? "-" : inj.magnitude.toFixed(2)],
                ["Batch size", `${inj.batch_size.toLocaleString()} rows`], ["Seed", inj.seed], ["Source rows", "test.csv (unseen data)"]]} />
              <p className="small muted">The same settings and seed rebuild exactly the same batch.</p></div>
          ) : (
            <div className="card stack"><h3>What was uploaded</h3>
              <KV rows={[["File", rec.upload?.file || "-"], ["Dataset", rec.dataset === "uci" ? "UCI Adult" : "IEEE-CIS"],
                ["Rows", rec.upload?.rows?.toLocaleString()], ["Labels", rec.upload?.labels_removed?.length ? "removed before C1" : "none in the file"]]} />
              <p className="small muted">No labels reach C1, so the true harm of an uploaded batch is unknown.</p></div>
          )}
          {truth ? (
            <div className="card stack"><h3>What it did to the model (truth, from labels)</h3>
              <KV rows={[[`${metric}, whole test set`, truth.baseline.toFixed(4)], [`${metric}, this batch`, truth.on_batch.toFixed(4)],
                ["True harm (fixed baseline)", `${sg(t)} pp · ${sevOf(t)}`],
                ...(truth.same_rows_before != null ? [["Same rows before the drift", truth.same_rows_before.toFixed(4)]] : []),
                ["Paired harm (same rows, before vs after)", `${sg(truth.true_harm_paired_pp)} pp`]]} />
              <p className="small muted">The fixed baseline is what C1 is trained to predict; the paired harm removes sampling noise.</p></div>
          ) : (
            <div className="card stack"><h3>What it did to the model</h3>
              <p className="muted">Unknown: an uploaded batch has no labels. Run the same kind of drift in the Drift Lab to see the true harm.</p></div>
          )}
        </div>

        <div className="card stack">
          <div className="head"><h3>C1's answer vs the truth</h3>
            {inside != null && <span className="chip" style={{ color: inside ? "var(--ok-fg)" : "var(--bad-fg)" }}>
              {inside ? "✓ truth inside the interval" : "✗ truth outside the interval"}</span>}</div>
          <div className="run-grid3">
            <div className="kpi"><span className="k">C1 predicted (no labels)</span><span className="v acc">{sg(v.harm)} pp</span><span className="small muted">severity {v.sev}</span></div>
            <div className="kpi"><span className="k">95% interval</span><span className="v mid">{sg(v.lo)} to {sg(v.hi)}</span><span className="small muted">margin for "{v.ext}" drift, {v.bs.toLocaleString()} rows</span></div>
            <div className="kpi"><span className="k">Error</span><span className="v mid">{t != null ? `${Math.abs(t - v.harm).toFixed(2)} pp` : "-"}</span><span className="small muted">{t != null ? `truth ${sg(t)} pp` : "truth unknown"}</span></div>
          </div>
          <TruthVsC1 truth={t} harm={v.harm} lo={v.lo} hi={v.hi} />
        </div>

        <div className="grid2">
          <Description v={v} inj={inj} />
          <div className="card stack"><h3>Classic monitor vs C1{t != null ? " vs the truth" : ""}</h3>
            <div className={`cmp ${isClassicAlarm(v) ? "alarm" : "calm"}`}><span className="k">A classic drift monitor would say</span>
              <b>{isClassicAlarm(v) ? `ALARM: ${v.fired.length} of 6 detectors fired` : v.fired.length ? `${v.fired.length} fired, no alarm` : "Nothing"}</b></div>
            <div className={`cmp ${v.sev !== "none" ? "alarm" : "calm"}`}><span className="k">C1 says</span>
              <b>{v.sev !== "none" ? `Repair needed · severity ${v.sev}` : "No repair needed"}</b></div>
            {t != null && <div className={`cmp ${t >= 1 ? "alarm" : "calm"}`}><span className="k">The truth says</span>
              <b>{t >= 1 ? `Repair needed · ${sevOf(t)}` : "No repair needed"} ({sg(t)} pp)</b></div>}
          </div>
        </div>

        <Detectors c1={rec.c1} />
        <PredictionStats batch={rec.c1.prediction_stats} reference={rec.reference_prediction_stats} />
      </>)}

      {rec.pipeline && (
        <div className="card stack">
          <div className="head"><h3>Full pipeline</h3><span className="small muted mono">run {rec.pipeline.run_id} · {rec.pipeline.status} · {rec.pipeline.runtime_sec} s</span></div>
          <PipelineSteps run={rec.pipeline} />
        </div>
      )}
      <JsonBox title="Saved run file (JSON)" data={rec} note={`platform_runs\\runs\\${rec.run_id}.json in the data folder`} />
    </>
  );
}

function Verdict({ v, out }) {
  const [title, text, tone] = verdict(v, "Repair recommended.");
  return (
    <div className={`verdict ${tone === "warn" ? "warn" : ""}`} role="status">
      <b>{title}</b><span>{text}</span>
      <span className="aside"><span className={`outcome ${OUTCOME_CLASS[out]}`} title={OUTCOME_HELP[out]}>{out}</span></span>
    </div>
  );
}

function Description({ v, inj }) {
  const tick = (ok) => (inj ? <span className={`run-ok ${ok ? "y" : "n"}`}>{ok ? "✓" : "✗"}</span> : <span />);
  const ext = inj ? injectedExtent(inj.drift_type) : "?";
  const inter = inj ? injectedInteraction(inj.drift_type) : null;
  const shifted = inj?.shifted || [];
  return (
    <div className="card stack"><h3>Drift description: {inj ? "injected vs found" : "what C1 found"}</h3>
      <div className="run-cmp">
        <span className="h" /><span className="h">{inj ? "Injected" : ""}</span><span className="h">C1 found</span><span className="h" />
        <span>Extent</span><span className="m">{inj ? ext : ""}</span><span className="m">{v.ext}</span>{tick(ext === v.ext)}
        <span>Interaction</span><span className="m">{inj ? (inter ? "yes" : "no") : ""}</span><span className="m">{Math.round(v.pint * 100)}% likely</span>{tick(inter === (v.pint >= 0.5))}
        <span>Features</span><span className="m">{inj ? shifted.join(", ") || "-" : ""}</span><span className="m">{v.aff.join(", ") || "none listed"}</span>{tick(shifted.length ? shifted.every((f) => v.aff.includes(f)) : v.aff.length === 0)}
      </div>
      <p className="small muted">An interaction changes no single feature, so none is listed for it. A feature is listed only when every detector that scores it passes its strictest clean-batch level.</p>
    </div>
  );
}

function Detectors({ c1 }) {
  const fired = c1.detectors_fired_list || [];
  const s = c1.detector_scores || {};
  return (
    <div className="card stack"><h3>The six detectors</h3>
      <div className="run-bars">
        {DETECTORS.map(([label, key, id, max]) => {
          const on = fired.includes(id), val = s[key];
          const w = val == null ? 0 : key === "clf_auc" ? Math.max(0, (val - 0.45) / 0.55) : Math.min(1, val / max);
          return [<span key={id + "l"}>{label}</span>,
            <div key={id + "b"} className="bar"><i className={on ? "on" : ""} style={{ width: `${(w * 100).toFixed(0)}%` }} /></div>,
            <span key={id + "v"} className="mono">{val == null ? "-" : val.toFixed(4)}</span>,
            <span key={id + "p"} className={`pill ${on ? "on" : ""}`}>{on ? "fired" : "quiet"}</span>];
        })}
      </div>
      <p className="small muted">{fired.length} of 6 fired. A detector fires when its score passes its clean-batch level for this batch size; a classic monitor alarms at 3 or more.</p>
    </div>
  );
}

function PredictionStats({ batch, reference }) {
  if (!batch) return null;
  const keys = Object.keys(STAT_LABEL).filter((k) => k in batch);
  return (
    <div className="card stack"><h3>How the model behaves on this batch</h3>
      <div className="scroll-x"><table className="tbl">
        <thead><tr><th>Signal (no labels needed)</th><th className="mono">Reference</th><th className="mono">This batch</th><th className="mono">Change</th></tr></thead>
        <tbody>{keys.map((k) => {
          const a = reference?.[k], b = batch[k];
          return (<tr key={k}><td>{STAT_LABEL[k]}</td><td className="mono">{a == null ? "-" : a.toFixed(3)}</td><td className="mono">{b.toFixed(3)}</td>
            <td className="mono">{a == null ? "-" : `${b - a >= 0 ? "+" : "−"}${Math.abs(b - a).toFixed(3)}`}</td></tr>);
        })}</tbody>
      </table></div>
      <p className="small muted">These are inputs to C1's harm prediction. Lower confidence and a shifted positive rate are early signs of harm. Reference: the model on 5,000 reference rows.</p>
    </div>
  );
}
