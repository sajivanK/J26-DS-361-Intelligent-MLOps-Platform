import { useRef, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../../api.js";
import { useAction, useApi } from "../../hooks.js";
import { useSession } from "../../session.jsx";
import { c1View, driftKind, isC1Alert, isClassicAlarm, sg, verdict } from "../../ui/c1format.js";
import ClassicVsC1 from "../../ui/ClassicVsC1.jsx";
import { Empty, ErrorBox, FeatureTags, JsonBox, KV, PageHeader, Verdict } from "../../ui/parts.jsx";
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
  }

  const cur = sel >= 0 ? runs[sel] : null;
  const nClassic = runs.filter((b) => isClassicAlarm(b.v)).length;
  const nAlert = runs.filter((b) => isC1Alert(b.v)).length;

  return (
    <>
      <PageHeader eyebrow="Component 1 · Sajivan K" title="C1 Drift Detection"
        subtitle="Most drift alarms are harmless. C1 tells you which ones are not."
        chip={info.data ? `engine ${info.data.engine_version} · contract v1.1` : "contract v1.1"} />

      <section className="card runbar" aria-label="Check a batch">
        <div className="field"><label htmlFor="c1ds">Dataset</label>
          <select id="c1ds" value={ds} onChange={(e) => setDs(e.target.value)}>
            <option value="uci">UCI Adult (income, XGBoost)</option>
            <option value="ieee">IEEE-CIS Fraud (LightGBM)</option>
          </select></div>
        <div className="field" style={{ flex: "2 1 300px" }}><label htmlFor="c1file">Batch file (CSV, same columns as the model's features; a target column is removed)</label>
          <input id="c1file" ref={fileRef} type="file" accept=".csv,text/csv" onChange={(e) => setFile(e.target.files[0] || null)} /></div>
        <button className={`btn primary ${busy ? "busy" : ""}`} type="button" disabled={!file || busy} onClick={run}>
          {busy ? "Detecting…" : "Run C1"}</button>
      </section>
      <ErrorBox error={error} />

      <section className="stack" style={{ gap: 12 }} aria-live="polite">
        {cur ? <Answer b={cur} /> : <Empty>Choose a batch file (for example <span className="mono">batch_uci_multi.csv</span>) and press <b>Run C1</b>. To test with a known drift and its true harm, use the <b>Drift Lab</b>.</Empty>}
      </section>

      <section className="card stack tight hist">
        <div className="section-title">
          <h3>{runs.length ? `Batches checked this session: ${nClassic} classic alarm${nClassic === 1 ? "" : "s"}, ${nAlert} C1 alert${nAlert === 1 ? "" : "s"}` : "Batches checked this session"}</h3>
          <span className="small muted">dot: predicted harm · bar: 95% interval · dashed: 1 pp, where severity starts · click a batch</span>
        </div>
        {runs.length ? <HistoryChart runs={runs} selected={sel} onSelect={setSel} /> : <p className="small muted">No batches yet.</p>}
      </section>

      {info.error && <ErrorBox error={info.error} />}
      {info.data?.recorded && <Proof ds={ds} rec={info.data.recorded} />}
      {info.data && <ModelEngine info={info.data} />}

      <JsonBox title="Raw output (JSON, contract v1.1)" data={cur?.out}
        note="The exact message C1 sends to C2, C3 and C4 for the selected batch." />
    </>
  );
}

function Answer({ b }) {
  const v = b.v;
  const [t, s, tone] = verdict(v);
  return (
    <>
      <Verdict title={t} text={s} tone={tone} aside={`${b.file} · ${v.bs.toLocaleString()} rows`} />
      {v.warnings.map((w) => <div key={w} className="warnline">{w}</div>)}
      {b.runId && <div className="row"><Link className="linkbtn" to={`/runs/${b.runId}`}>Open full analysis</Link>
        <span className="small muted mono">saved as run {b.runId}</span></div>}
      <div className="hero">
        <div><span className="q">How much will it hurt?</span><span className="a">{sg(v.harm)} pp</span>
          <span className="s">predicted {v.metric} drop, no labels used</span></div>
        <div><span className="q">How sure is C1?</span><span className="a sm">{sg(v.lo)} to {sg(v.hi)} pp</span>
          <span className="s">95% interval, sized for "{v.ext}" drift and {v.bs.toLocaleString()} rows</span></div>
        <div><span className="q">What kind of drift?</span><span className="a sm">{driftKind(v)}</span>
          <span className="row"><FeatureTags features={v.aff} /></span></div>
      </div>
      <ClassicVsC1 v={v} />
    </>
  );
}

function Proof({ ds, rec }) {
  const [p, setP] = useState("n1");
  const name = ds === "uci" ? "UCI Adult" : "IEEE-CIS";
  const cards = [
    ["n1", rec.n1_headline.value, rec.n1_headline.text, "Novelty 1 · predicted harm"],
    ["n2", rec.n2_coverage.overall.toFixed(3), "interval coverage (target 0.95)", "Novelty 2 · uncertainty"],
    ["n3", rec.n3.extent_accuracy.toFixed(3), "drift extent accuracy", "Novelty 3 · description"],
  ];
  return (
    <section className="stack">
      <div className="section-title"><span className="eyebrow">Why trust it · {name} benchmark, {rec.benchmark_batches.toLocaleString()} batches</span><span className="small muted">click a card</span></div>
      <div className="proof">
        {cards.map(([k, val, txt, d]) => (
          <button key={k} type="button" className="pcard" aria-pressed={p === k} onClick={() => setP(k)}>
            <span className="v">{val}</span><span className="k">{txt}</span><span className="d">{d}</span>
          </button>))}
      </div>
      <div className="card stack tight">
        {p === "n1" && (<><h3>Harm prediction error (MAE, pp, paired target)</h3>
          <p className="small muted">Lower is better. {rec.n1_headline.detail}</p><ErrorBars rows={rec.n1_mae} /></>)}
        {p === "n2" && (<><h3>Interval coverage by drift type (fixed target)</h3>
          <p className="small muted">Target 0.95 (line). Grey: one shared margin. Teal: per-extent margins (engine 1.2.3), which fix multi drift at a small cost on interaction.</p>
          <CoverageBars cats={rec.n2_coverage.types} a={rec.n2_coverage.global} b={rec.n2_coverage.mondrian} />
          <div className="legend"><span><i style={{ background: "var(--muted)", opacity: .5 }} />one shared margin</span><span><i style={{ background: "var(--accent-2)" }} />per-extent margins</span></div></>)}
        {p === "n3" && (<><h3>Drift extent classification</h3>
          {rec.n3.matrix ? (<><p className="small muted">Rows: true extent. Cells: share predicted.</p><ConfusionMatrix labels={rec.n3.labels} matrix={rec.n3.matrix} /></>)
            : <p className="small muted">Extent accuracy {rec.n3.extent_accuracy.toFixed(3)} ({rec.n3.note}).</p>}</>)}
      </div>
    </section>
  );
}

function ModelEngine({ info }) {
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
    <section className="stack">
      <div className="section-title"><span className="eyebrow">Model and engine · {rec.title || info.label}</span><span className="small muted">switches with the dataset</span></div>
      <div className="grid2">
        <div className="card stack"><h3>Monitored model</h3><KV rows={model} /></div>
        <div className="card stack"><h3>C1 engine {info.engine_version} · fresh test (unseen batches)</h3><KV rows={eng} /></div>
      </div>
    </section>
  );
}
