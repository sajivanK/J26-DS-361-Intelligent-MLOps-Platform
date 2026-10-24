import { useRef, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../../api.js";
import { useAction, useApi } from "../../hooks.js";
import { refreshAlerts } from "../../platform.jsx";
import { useSession } from "../../session.jsx";
import { c1View, classicText, driftKind, isC1Alert, isClassicAlarm, sg, verdict } from "../../ui/c1format.js";
import { Icon } from "../../layout/icons.jsx";
import { Empty, ErrorBox, FeatureTags, KV, PageHeader, Verdict } from "../../ui/parts.jsx";
import { IconChip, Spark } from "../../ui/viz.jsx";
import { ConfusionMatrix, CoverageBars, ErrorBars, HistoryChart } from "./C1Charts.jsx";
import "./C1Page.css";

const pct = (v) => `${(v * 100).toFixed(1).replace(/\.0$/, "")}%`;

export default function C1Page() {
  const [ds, setDs] = useState("uci");
  const [file, setFile] = useState(null);
  const fileRef = useRef(null);
  const [runs, setRuns] = useSession(`c1:${ds}`, []);
  const [sel, setSel] = useSession(`c1:${ds}:sel`, -1);
  const info = useApi(() => api.c1EngineInfo(ds), [ds]);
  const [detect, { busy, error }] = useAction(api.c1Detect);

  async function run() {
    const out = await detect(ds, file);
    if (!out) return;
    const short = file.name.replace(/\.csv$/i, "").slice(0, 14);
    setRuns((r) => [...r, { out, v: c1View(out), file: file.name, short, runId: out.runId }]);
    setSel(runs.length);
    refreshAlerts();                       // a C1 alert on an uploaded batch waits for C2
  }

  const cur = sel >= 0 ? runs[sel] : null;
  const nClassic = runs.filter((b) => isClassicAlarm(b.v)).length;
  const nAlert = runs.filter((b) => isC1Alert(b.v)).length;

  const dsLabel = ds === "uci" ? "UCI Adult" : "IEEE-CIS";
  const controls = (
    <div className="c1-ctrl">
      <label htmlFor="c1ds" className="sr">Dataset</label>
      <select id="c1ds" value={ds} onChange={(e) => setDs(e.target.value)}>
        <option value="uci">UCI Adult (income, XGBoost)</option>
        <option value="ieee">IEEE-CIS Fraud (LightGBM)</option>
      </select>
      <label className="btn c1-up" title="CSV with the model's features; a target column is removed">
        <Icon name="upload" size={17} className="" />{file ? file.name.slice(0, 22) : "Upload CSV"}
        <input ref={fileRef} type="file" accept=".csv,text/csv" className="sr" onChange={(e) => setFile(e.target.files[0] || null)} />
      </label>
      <button className={`btn primary ${busy ? "busy" : ""}`} type="button" disabled={!file || busy} onClick={run}>
        <Icon name="play" size={13} className="" />{busy ? "Detecting…" : "Run C1"}</button>
    </div>
  );

  return (
    <>
      <PageHeader eyebrow={`C1 Drift Detection › Sajivan K › ${dsLabel}`} title="C1 Drift Detection"
        subtitle="Most drift alarms are harmless. C1 tells you which ones are not, without labels."
        chip={info.data ? `engine ${info.data.engine_version} · contract v1.1` : "contract v1.1"} right={controls} />
      <ErrorBox error={error} />

      <section className="stack" style={{ gap: 14 }} aria-live="polite">
        {cur ? <Answer b={cur} runs={runs} nClassic={nClassic} nAlert={nAlert} />
          : <Empty>Upload a batch file (for example <span className="mono">batch_uci_multi.csv</span>) and press <b>Run C1</b>. To test with a known drift and its true harm, use the <b>Drift Lab</b>.</Empty>}
      </section>

      <section className="c1-row2">
        <div className="card stack tight hist">
          <div className="section-title">
            <div><h3>Drift severity across batches</h3><span className="small muted">Dot: predicted harm · bar: 95% interval · click a batch</span></div>
            <span className="small muted">{runs.length ? `${runs.length} batch${runs.length === 1 ? "" : "es"} · ${nClassic} classic alarm${nClassic === 1 ? "" : "s"} · ${nAlert} C1 alert${nAlert === 1 ? "" : "s"}` : ""}</span>
          </div>
          {runs.length ? <HistoryChart runs={runs} selected={sel} onSelect={setSel} /> : <p className="small muted">No batches yet.</p>}
        </div>
        <ClassicCard v={cur?.v} />
      </section>

      {info.error && <ErrorBox error={info.error} />}
      {info.data?.recorded && <Proof ds={ds} rec={info.data.recorded} cur={cur} />}
      {info.data && <ModelEngine info={info.data} out={cur?.out} />}
    </>
  );
}

function Answer({ b, runs, nClassic, nAlert }) {
  const v = b.v;
  const [t, s, tone] = verdict(v);
  const alert = v.sev !== "none";
  return (
    <>
      <Verdict title={t} text={s} tone={tone} aside={`${b.file} · ${v.bs.toLocaleString()} rows`}
        extra={b.runId && <Link className="small" style={{ fontWeight: 600, color: "inherit" }} to={`/runs/${b.runId}`}>Open full analysis →</Link>} />
      {v.warnings.map((w) => <div key={w} className="warnline">{w}</div>)}
      <div className="c1-tiles">
        <div className="card c1-tile">
          <IconChip name="bars" tone={alert ? "red" : "green"} />
          <div className="stack" style={{ gap: 5 }}>
            <span className="q">Predicted {v.metric} drop</span>
            <span className="num c1-big">{sg(v.harm)}<small> pp</small></span>
            <span className={`pill tone-${alert ? "red" : "green"}`} style={{ alignSelf: "flex-start" }}>● {alert ? `${v.sev} severity` : "no repair needed"}</span>
            <span className="s">No labels used.</span>
          </div>
        </div>
        <div className="card c1-tile">
          <IconChip name="shield" tone="teal" />
          <div className="stack" style={{ gap: 6, flex: 1, minWidth: 0 }}>
            <span className="q">C1 95% interval</span>
            <span className="num c1-mid">{sg(v.lo)} – {sg(v.hi)} pp</span>
            <MiniInterval v={v} />
            <span className="s">sized for "{v.ext}" drift and {v.bs.toLocaleString()} rows</span>
          </div>
        </div>
        <div className="card c1-tile">
          <IconChip name="trend" tone="green" />
          <div className="stack" style={{ gap: 6, minWidth: 0 }}>
            <span className="q">What kind of drift?</span>
            <span className="c1-kind">{KIND[driftKind(v)] || driftKind(v)}</span>
            <div className="row"><FeatureTags features={v.aff} /><span className="tagf none">interaction {Math.round(v.pint * 100)}%</span></div>
          </div>
        </div>
        <div className="card c1-tile">
          <IconChip name="db" tone="blue" />
          <div className="stack" style={{ gap: 4, minWidth: 0 }}>
            <span className="q">This session</span>
            <span className="c1-kind"><span className="num">{runs.length}</span> batch{runs.length === 1 ? "" : "es"}</span>
            <span className="s">{nClassic} classic alarm{nClassic === 1 ? "" : "s"} · {nAlert} C1 alert{nAlert === 1 ? "" : "s"}</span>
            <Spark values={runs.map((r) => r.v.harm)} w={120} h={26} color="#2563EB" caption={runs.length > 1 ? "C1 harm per batch" : null} />
          </div>
        </div>
      </div>
    </>
  );
}

const KIND = { "one feature": "One feature shifted", "several features": "Several features shifted", none: "No drift", "possible interaction": "Possible interaction" };

function MiniInterval({ v }) {
  const lo = Math.min(0, v.lo) - 1, hi = Math.max(v.hi, 8) + 1, x = (q) => `${((q - lo) / (hi - lo)) * 100}%`;
  return (
    <div className="c1-mini" aria-hidden="true">
      <div className="track" /><div className="rng" style={{ left: x(v.lo), width: `calc(${x(v.hi)} - ${x(v.lo)})` }} />
      <div className="dot1" style={{ left: x(v.harm) }} />
    </div>
  );
}

function ClassicCard({ v }) {
  return (
    <div className="card stack c1-classic">
      <h3>Classic monitor vs C1</h3>
      {v ? (<>
        <div className="cmpbox"><span className="eyebrow">Classic monitor</span><b>{classicText(v)}</b>
          <span className="small muted">Says something changed. Not whether it matters.</span></div>
        <div className={`cmpbox ${v.sev !== "none" ? "warn" : "ok"}`}><span className="eyebrow" style={{ color: "inherit" }}>C1 says</span>
          <b>{v.sev !== "none" ? `Repair needed · ${v.sev}` : "No repair needed"}</b>
          <span className="small">{sg(v.harm)} pp {v.metric} drop, interval {sg(v.lo)} to {sg(v.hi)}.</span></div>
      </>) : <p className="small muted">Run a batch to compare.</p>}
      <span className="small muted">Classic detectors fire on any change, harmful or not. C1 raises an alert only when the predicted harm reaches severity low (1 pp) or above.</span>
    </div>
  );
}

function Proof({ ds, rec, cur }) {
  const [p, setP] = useState("n1");
  const name = ds === "uci" ? "UCI" : "IEEE-CIS";
  const cards = [
    ["n1", "target", "teal", rec.n1_headline.value, rec.n1_headline.text, "Novelty 1"],
    ["n2", "bolt", "blue", rec.n2_coverage.overall.toFixed(3), "interval coverage (target 0.95)", "Novelty 2"],
    ["n3", "layers", "purple", rec.n3.extent_accuracy.toFixed(3), "drift extent accuracy", "Novelty 3"],
  ];
  return (
    <section className="c1-row3">
      <div className="card stack">
        <div className="section-title"><h3>Why trust C1?</h3><span className="small muted">{name} · {rec.benchmark_batches.toLocaleString()} batches · click a card</span></div>
        <div className="proof">
          {cards.map(([k, ic, tone, val, txt, d]) => (
            <button key={k} type="button" className="pcard" aria-pressed={p === k} onClick={() => setP(k)}>
              <IconChip name={ic} tone={tone} size="xs" iconSize={15} />
              <span className="v">{val}</span><span className="k">{txt}</span><span className="d eyebrow">{d}</span>
            </button>))}
        </div>
      </div>
      <div className="card stack tight">
        {p === "n1" && (<><h3 className="row" style={{ gap: 8 }}><Icon name="bars" size={18} className="" strokeWidth={2.4} style={{ color: "var(--accent)" }} />Comparison with baselines (MAE, pp)</h3>
          <p className="small muted">Lower is better. {rec.n1_headline.detail}</p><ErrorBars rows={rec.n1_mae} /></>)}
        {p === "n2" && (<><h3>Interval coverage by drift type</h3>
          <p className="small muted">Target 0.95 (line). Grey: one shared margin. Teal: per-extent margins (engine 1.2.3).</p>
          <CoverageBars cats={rec.n2_coverage.types} a={rec.n2_coverage.global} b={rec.n2_coverage.mondrian} />
          <div className="legend"><span><i style={{ background: "var(--muted)", opacity: .5 }} />one shared margin</span><span><i style={{ background: "var(--accent-2)" }} />per-extent margins</span></div></>)}
        {p === "n3" && (<><h3>Drift extent classification</h3>
          {rec.n3.matrix ? (<><p className="small muted">Rows: true extent. Cells: share predicted.</p><ConfusionMatrix labels={rec.n3.labels} matrix={rec.n3.matrix} /></>)
            : <p className="small muted">Extent accuracy {rec.n3.extent_accuracy.toFixed(3)} ({rec.n3.note}).</p>}</>)}
      </div>
      <BatchDetails v={cur?.v} />
    </section>
  );
}

function BatchDetails({ v }) {
  const alert = v && v.sev !== "none";
  return (
    <div className="card stack c1-kvcard">
      <h3 className="row" style={{ gap: 8 }}><Icon name="doc" size={18} className="" style={{ color: "var(--accent)" }} />Batch prediction details</h3>
      {v ? (
        <div className="kvl">
          <span>Drift detected</span><span>{String(v.det)}</span>
          <span>Drift extent</span><span>{v.ext}</span>
          <span>Affected features</span><span>{v.aff.join(", ") || "none listed"}</span>
          <span>Interaction suspected</span><span>{v.pint.toFixed(2)}</span>
          <span>Detectors fired</span><span>{v.fired.length} of 6</span>
          <span>Severity</span><span><span className={`pill tone-${alert ? "red" : "green"}`}>{v.sev}</span></span>
          <span>Next step</span><span><span className={`pill tone-${alert ? "amber" : "grey"}`}>{alert ? "sent to C2" : "no action"}</span></span>
        </div>) : <p className="small muted">Run a batch to see its details.</p>}
    </div>
  );
}

function ModelEngine({ info, out }) {
  const rec = info.recorded || {};
  const ft = rec.fresh_test || {};
  const auc = info.harm_metric === "auc_drop";
  const q = info.n2_quantile_by_extent || {};
  const sizes = info.calibrated_batch_sizes || [];
  const model = [
    ["Model", `${rec.model || "?"} (${info.model_id})`],
    ["Features", `${info.n_features} (${info.n_continuous} continuous, ${info.n_categorical} categorical)`],
    ...(auc ? [["Test AUC", info.fixed_baseline_auc?.toFixed(4)]] : []),
    ["Test accuracy", info.fixed_baseline_accuracy?.toFixed(4)],
    ["Test rows", rec.test_rows?.toLocaleString()],
    ["Positive class", rec.positive_rate],
    ["Harm measured as", auc ? "AUC drop (pp)" : "accuracy drop (pp)"],
  ];
  const eng = [
    ["Harm error (MAE)", `${ft.mae_pp} pp`],
    ["Interval coverage (target 0.95)", ft.coverage?.toFixed(3)],
    ["Drift extent accuracy", ft.extent_accuracy?.toFixed(3)],
    ["Interaction flag (AUC)", ft.interaction_auc?.toFixed(3)],
    ["False alarms on clean batches", ft.false_alarm_rate != null ? pct(ft.false_alarm_rate) : "-"],
    ["Fresh test batches", ft.batches],
    ["Interval quantile q, none / single / multi", ["none", "single", "multi"].map((k) => q[k]?.toFixed(2) ?? "-").join(" / ")],
    ["Calibrated batch sizes", sizes.length ? `${sizes[0].toLocaleString()} to ${sizes[sizes.length - 1].toLocaleString()} rows` : "-"],
    ["Runtime per batch (laptop)", rec.runtime_laptop],
  ];
  return (
    <section className="c1-row3">
      <div className="card stack c1-kvcard"><h3 className="row" style={{ gap: 8 }}><Icon name="db" size={18} className="" style={{ color: "var(--accent)" }} />Monitored model</h3><KV rows={model} /></div>
      <div className="card stack c1-kvcard"><h3 className="row" style={{ gap: 8 }}><Icon name="gear" size={18} className="" style={{ color: "var(--accent)" }} />C1 engine {info.engine_version} · fresh test ({ft.batches} unseen batches)</h3><KV rows={eng} /></div>
      <OutputCard out={out} />
    </section>
  );
}

const SHOWN = ["model_id", "engine_version", "batch_size", "drift_detected", "predicted_harm", "harm_interval", "harm_metric",
  "harm_severity", "marginal_extent", "affected_features", "detectors_fired"];

function OutputCard({ out }) {
  const [all, setAll] = useState(false);
  const text = out ? JSON.stringify(all ? out : Object.fromEntries(SHOWN.filter((k) => k in out).map((k) => [k, out[k]])), null, 2) : null;
  const download = () => {
    const url = URL.createObjectURL(new Blob([JSON.stringify(out, null, 2)], { type: "application/json" }));
    const a = document.createElement("a"); a.href = url; a.download = `c1_output_${out.evidence_version || "batch"}.json`; a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  return (
    <div className="card stack c1-json">
      <div className="section-title"><h3 className="row" style={{ gap: 8 }}><Icon name="code" size={18} className="" style={{ color: "var(--accent)" }} />C1 output (contract v1.1)</h3>
        {out && <div className="row"><button type="button" className="pbtn" onClick={() => setAll((x) => !x)}>{all ? "Show key fields" : "View all"}</button>
          <button type="button" className="pbtn" onClick={download}>Download</button></div>}</div>
      {out ? <pre className="dark">{text}{all ? "" : "\n…"}</pre> : <p className="small muted">The exact message C1 sends to C2, C3 and C4 appears here after a run.</p>}
    </div>
  );
}
