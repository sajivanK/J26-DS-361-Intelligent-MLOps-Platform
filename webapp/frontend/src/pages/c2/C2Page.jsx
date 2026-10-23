// C2 Root Cause Diagnosis page (Tharmika N, IT22247940).
// Shows the latest stored run (or the one picked from the history), its temporal SHAP analysis across
// all stored runs, the evidence behind the root cause, the affected group, and the run history.
// The business report opens on its own page (/c2/report/:run) in the same tab.
import { useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../../api.js";
import { useAction, useApi } from "../../hooks.js";
import { useSession } from "../../session.jsx";
import { Empty, ErrorBox, JsonBox, PageHeader } from "../../ui/parts.jsx";
import {
  EvidenceChart, HeatKey, ProofBars, RelianceChart, SIGNALS, ShareBars, TemporalChart, TemporalLegend,
  temporalWindow, ZHeatmap,
} from "./C2Charts.jsx";
import { answer, baselines, c1PredOf, clock, harmOf, pct, runsAgreeing, sg, STATE, time } from "./c2format.js";
import "./C2Page.css";

const DEFAULT_FORM = { drift_type: "marginal_single", features: ["capital_gain"], magnitude: 0.6, batch_size: 1000 };

export function StatePill({ state }) {
  const s = STATE[state] || STATE.clean;
  return <span className={`c2-pill ${s.pill}`}>{s.label}</span>;
}

export default function C2Page() {
  const info = useApi(api.c2Info, []);
  const hist = useApi(api.c2Runs, []);
  const [sel, setSel] = useSession("c2:sel", null);        // null = always show the latest run
  const runs = hist.data?.runs || [];
  const latest = runs[0]?.run ?? null;
  const n = sel ?? latest;
  const rec = useApi(() => (n ? api.c2Run(n) : Promise.resolve(null)), [n, latest]);
  const [form, setForm] = useSession("c2:form", DEFAULT_FORM);
  const [doRun, run] = useAction(api.c2RunNew);
  const [doReset, reset] = useAction(api.c2Reset);
  const [note, setNote] = useState(null);

  async function runNext() {
    const out = await doRun(form);
    if (!out) return;
    setNote(out.c1_note);
    setSel(null);
    hist.reload();
  }
  async function newPeriod() {
    if (!window.confirm("Start a new monitoring period? This clears the C2 run history and warms the engine up again.")) return;
    if (await doReset()) { setSel(null); hist.reload(); }
  }
  const pick = (r) => { setSel(r === latest ? null : r); window.scrollTo({ top: 0, behavior: "smooth" }); };

  return (
    <>
      <PageHeader eyebrow="Component 2 · Tharmika N" title="C2 Root Cause Diagnosis"
        subtitle="A drift alarm says something changed. C2 finds the feature whose change actually hurts the model, and the group of records it hurts."
        chip={info.data ? `engine 1.0 · contract v1.1 · ${info.data.runs_stored ?? runs.length} runs stored` : "contract v1.1"} />

      <RunBar info={info.data} form={form} setForm={setForm} busy={run.busy || reset.busy} onRun={runNext} onReset={newPeriod} />
      <ErrorBox error={info.error || hist.error || run.error || reset.error} />
      {note && <div className="warnline">{note}</div>}

      <section className="stack" style={{ gap: 12 }} aria-live="polite">
        {rec.data ? <Answer rec={rec.data} latest={latest} />
          : hist.loading || rec.loading ? <p className="muted">Loading…</p>
            : <Empty>No diagnosis stored yet. Pick a drift above and press <b>Run diagnosis</b> two or three times: C2 compares every run with the runs before it, so a cause is confirmed only when it persists.</Empty>}
      </section>

      {rec.data && (<>
        <section className="c2-two">
          <div className="card stack tight">
            <h3>What the model relies on: normal vs this run</h3>
            <p className="c2-sub">Average influence of each feature on a prediction (mean |SHAP|). Grey dot: normal, from clean data. Coloured dot: this run. Orange: the model leans on it more than usual; blue: less.</p>
            <RelianceChart rec={rec.data} />
            <div className="chart-legend"><span><i style={{ background: "var(--muted)", opacity: 0.45 }} />normal</span>
              <span><i style={{ background: "var(--chart-2)" }} />more than usual</span><span><i style={{ background: "var(--chart-1)" }} />less than usual</span></div>
          </div>
          <div className="card stack tight">
            <h3>Why this feature: the evidence behind the score</h3>
            <p className="c2-sub">Each bar is the Drift Impact Score (0 to 1), split into the weighted evidence that built it.</p>
            <EvidenceChart rec={rec.data} />
            <div className="chart-legend">{SIGNALS.map(([k, name, col]) => <span key={k}><i style={{ background: col }} />{name}</span>)}</div>
          </div>
        </section>

        {hist.data?.temporal_shap?.runs?.length > 0 && <Temporal matrix={hist.data.temporal_shap} rec={rec.data} onSelect={pick} />}

        <section className="c2-two">
          <Who rec={rec.data} />
          <Compare rec={rec.data} />
        </section>
      </>)}

      {runs.length > 0 && <History runs={runs} current={n} latest={latest} onSelect={pick} />}
      {info.data?.recorded && <Proof rec={info.data.recorded} />}
      <JsonBox title="Raw output to C3 and C4 (JSON, contract v1.1)" data={rec.data ? { to_c3: rec.data.to_c3, to_c4: rec.data.to_c4 } : null}
        note="The exact messages C2 sends for the run on screen." />
    </>
  );
}

function RunBar({ info, form, setForm, busy, onRun, onReset }) {
  const types = info?.drift_types || [];
  const t = types.find((x) => x.id === form.drift_type) || { min: 1, max: 1, help: "" };
  const feats = info?.features?.all || [];
  const set = (k, v) => setForm((f) => ({ ...f, [k]: v }));
  const toggle = (f) => setForm((s) => {
    const on = s.features.includes(f);
    let next = on ? s.features.filter((x) => x !== f) : [...s.features, f];
    if (t.max === 1) next = on ? [] : [f];
    return { ...s, features: next.slice(-t.max) };
  });
  const ok = form.features.length >= t.min && form.features.length <= t.max;
  return (
    <section className="card stack" aria-label="Run a diagnosis">
      <div className="c2-runbar">
        <div className="field"><label htmlFor="c2type">Drift to inject (Drift Lab recipe)</label>
          <select id="c2type" value={form.drift_type} onChange={(e) => setForm((f) => ({ ...f, drift_type: e.target.value, features: e.target.value === "none" ? [] : f.features }))}>
            {types.map((x) => <option key={x.id} value={x.id}>{x.id === "none" ? "none (clean batch)" : x.id.replace("_", " ")}</option>)}
          </select></div>
        <div className="field"><label htmlFor="c2mag">Strength: {form.magnitude.toFixed(2)}</label>
          <input id="c2mag" type="range" min="0.05" max="1" step="0.05" value={form.magnitude} disabled={form.drift_type === "none"}
            onChange={(e) => set("magnitude", +e.target.value)} /></div>
        <div className="field"><label htmlFor="c2bs">Batch size</label>
          <select id="c2bs" value={form.batch_size} onChange={(e) => set("batch_size", +e.target.value)}>
            {(info?.batch_sizes || [1000]).map((b) => <option key={b} value={b}>{b.toLocaleString()} rows</option>)}
          </select></div>
        <div className="c2-actions">
          <button className={`btn primary ${busy ? "busy" : ""}`} type="button" disabled={busy || !info || !ok} onClick={onRun}>{busy ? "Diagnosing…" : "Run diagnosis"}</button>
          <button className="btn" type="button" disabled={busy || !info} onClick={onReset}>New period</button>
        </div>
      </div>
      {form.drift_type !== "none" && (
        <div className="stack tight">
          <span className="small muted">{t.help} {ok ? "" : <b>Pick {t.min === t.max ? t.min : `${t.min} to ${t.max}`}.</b>}</span>
          <div className="row" role="group" aria-label="Features to drift">
            {feats.map((f) => (<button key={f} type="button" className="pbtn" aria-pressed={form.features.includes(f)} onClick={() => toggle(f)}>
              {f}{info.features.high.includes(f) ? " ·hi" : ""}</button>))}
          </div>
        </div>)}
    </section>
  );
}

function Answer({ rec, latest }) {
  const st = rec.state, s = STATE[st], c3 = rec.to_c3, a = answer(rec), harm = harmOf(rec), c1 = c1PredOf(rec);
  const agree = runsAgreeing(rec), stab = c3.diagnosis_stability, group = rec.plain?.group || c3.affected_slice_rule;
  const verdictText = {
    clean: "Nothing needs repair in this batch.",
    harmless: "It changed, but it costs almost no accuracy.",
    suspected: "Seen in one run so far. The next run will confirm it or clear it.",
    confirmed: "Above the alarm line in at least 2 of the last 3 runs. Repair recommended.",
  }[st];
  return (
    <>
      <div className="section-title">
        <span className="eyebrow">Run #{rec.run}{rec.run === latest ? " · latest" : ""} · {rec.source}</span>
        <span className="small muted mono">{time(rec.timestamp)} · {(rec.n_rows || 0).toLocaleString()} rows · {rec.latency_s?.toFixed(2)} s</span>
      </div>
      <div className={`verdict ${s.tone === "warn" ? "warn" : ""}`} role="status">
        <b>{st === "clean" ? s.head : `${s.head}: ${c3.root_cause_feature}`}</b><span>{verdictText}</span>
        <span className="aside"><StatePill state={st} /></span>
      </div>
      <p className="c2-plain">{a.sentence.map((p, i) => (typeof p === "string" ? p : <b key={i}>{p.b}</b>))}</p>
      <div className="c2-hero">
        <div><span className="q">What caused it?</span>
          {st === "clean" ? <><span className="a sm">Nothing harmful</span><span className="s">no feature above the alarm line in 2 of the last 3 runs</span></>
            : <><span className="a">{c3.root_cause_feature}</span><span className="s">{c3.root_cause_category} drift · top cause in {agree} of the last 3 runs</span></>}</div>
        <div><span className="q">How much does it hurt?</span>
          <span className="a">{sg(harm)} pp</span><span className="s">accuracy recovered when {c3.root_cause_feature} is put back to normal · score {c3.drift_impact_score.toFixed(2)}{c1 != null ? ` · C1 predicted ${sg(c1)} pp` : ""}</span></div>
        <div><span className="q">Who is affected?</span>
          {st === "clean" || st === "harmless" ? <><span className="a sm">No group</span><span className="s">nothing sent to C3 for repair</span></>
            : <><span className="a">{pct(c3.slice_support)}</span><span className="s">of this batch ({a.rows.toLocaleString()} rows){rec.localization.ref_share != null ? ` · ${pct(rec.localization.ref_share, 1)} in training data` : ""}</span></>}</div>
      </div>
      {rec.truth && rec.injection?.drift_type !== undefined && (
        <div className="c2-truth"><span className="eyebrow">Drift Lab answer key</span>
          <span>true accuracy drop <b className="mono">{sg(rec.truth.true_harm_pp)} pp</b></span>
          <span>C2 estimate <b className="mono">{sg(harm)} pp</b></span>
          <span>injected <b className="mono">{rec.truth.true_causes.join(", ") || "nothing"}</b></span>
          <span>C2 root cause <b className="mono">{c3.root_cause_feature}</b> {rec.truth.true_causes.length ? (rec.truth.true_causes.includes(c3.root_cause_feature) ? "✓" : "✗") : ""}</span>
        </div>)}
      <div className="row" style={{ justifyContent: "space-between" }}>
        <div className="row">
          <span className={`badge ${rec.c1?.status === "ok" ? "ok" : ""}`}>C1 input: {rec.c1?.status || "missing"}{rec.c1?.harm_severity ? ` · ${rec.c1.harm_severity}` : ""}</span>
          <span className="badge">stability {stab.toFixed(2)}{stab < 0.5 ? " · C4 escalates" : ""}</span>
          {st !== "clean" && <span className="badge c2-wrap" title={c3.affected_slice_rule}>group: {group}</span>}
        </div>
        <Link className="btn primary" to={`/c2/report/${rec.run}`}>Open report</Link>
      </div>
      {rec.warnings?.map((w) => <div key={w} className="warnline">{w}</div>)}
    </>
  );
}

function Temporal({ matrix, rec, onSelect }) {
  const win = temporalWindow(matrix);
  const order = Object.entries(rec.temporal_shap.ewma).sort((a, b) => b[1] - a[1]).map((x) => x[0]).filter((f) => f in win.z);
  return (
    <section className="card stack">
      <div className="section-title"><h3>Temporal SHAP across stored runs</h3>
        <span className="small muted">Novelty 1 · last {win.runs.length} of {matrix.runs.length} runs · click a run to open it</span></div>
      <p className="c2-sub">z = how many times larger than normal noise a feature's explanation change is. An alarm needs the same feature above the alarm line in 2 of the last 3 runs, so a single noisy run never raises one.</p>
      <TemporalChart win={win} selected={rec.run} onSelect={onSelect} />
      <TemporalLegend win={win} />
      <ZHeatmap win={win} order={order} selected={rec.run} onSelect={onSelect} />
      <HeatKey thresh={win.thresh} />
    </section>
  );
}

function Who({ rec }) {
  const st = rec.state, c3 = rec.to_c3, loc = rec.localization;
  if (st === "clean" || st === "harmless") {
    return (<div className="card stack"><h3>Who is affected</h3><p className="c2-plain">No group of records needs repair in this run.</p></div>);
  }
  return (
    <div className="card stack">
      <h3>Who is affected</h3>
      <p className="c2-plain">Records where <b>{rec.plain?.group || c3.affected_slice_rule}</b>.</p>
      <ShareBars training={loc.ref_share} batch={c3.slice_support} />
      <div className="c2-rule mono">{c3.affected_slice_rule}</div>
      <p className="c2-sub">
        {`${loc.ref_share ? `This group is ${(c3.slice_support / loc.ref_share).toFixed(1)} times more common now than in training. ` : ""}About ${Math.round(loc.harmed_rows_estimate || 0)} rows look harmed by the drift; the group holds an estimated ${pct(loc.est_recall)} of them.`}
        {" "}Groups under 20 rows are never reported (privacy).{rec.plain && !rec.plain.decoded ? " Values are scaled: add preprocessor.pkl to decode them." : ""}
      </p>
    </div>
  );
}

function Compare({ rec }) {
  const b = baselines(rec), clean = rec.state === "clean", harm = harmOf(rec);
  return (
    <div className="stack">
      <div className="section-title"><span className="eyebrow">What simpler tools would say about this run</span></div>
      <div className="c2-cmp">
        <div className="cmp"><span className="k">A drift monitor ranks</span><span className="v mono">{b.drift.join(", ")}</span><span className="gap">Says what changed, not what hurts.</span></div>
        <div className="cmp"><span className="k">One SHAP snapshot ranks</span><span className="v mono">{b.snapshot.join(", ")}</span><span className="gap">{clean ? "Names a cause even on clean data." : "No noise check, no run history."}</span></div>
        <div className="cmp c2-ans"><span className="k">C2 says</span><span className="v mono">{clean ? "no alarm" : `${rec.to_c3.root_cause_feature}${harm != null ? ` · ${sg(harm)} pp` : ""}`}</span>
          <span className="gap">{clean ? "Normal noise." : "Cause, damage and group, confirmed over runs."}</span></div>
      </div>
    </div>
  );
}

function History({ runs, current, latest, onSelect }) {
  return (
    <section className="card stack tight">
      <div className="section-title"><h3>Run history</h3><span className="small muted">every run is stored with its full temporal SHAP analysis · click a row to view it</span></div>
      <div className="scroll-x">
        <table className="tbl c2-hist">
          <thead><tr><th>Run</th><th>Time</th><th>Batch</th><th>Result</th><th>Root cause</th>
            <th className="num">Accuracy lost</th><th className="num">True drop</th><th className="num">Stability</th><th>Report</th></tr></thead>
          <tbody>
            {runs.map((r) => (
              <tr key={r.run} aria-selected={r.run === current} tabIndex="0" onClick={() => onSelect(r.run)}
                onKeyDown={(e) => { if (e.key === "Enter") onSelect(r.run); }}>
                <td className="mono">#{r.run}{r.run === latest && <span className="badge" style={{ marginLeft: 6 }}>latest</span>}</td>
                <td className="mono">{clock(r.timestamp)}</td>
                <td>{r.source.replace("Drift Lab · ", "")}</td>
                <td><StatePill state={r.state} /></td>
                <td className="mono">{r.root_cause_feature || "none"}</td>
                <td className="num">{sg(r.estimated_harm_pp)} pp</td>
                <td className="num">{r.truth_pp == null ? "–" : `${sg(r.truth_pp)} pp`}</td>
                <td className="num">{r.diagnosis_stability.toFixed(2)}</td>
                <td><Link className="c2-link" to={`/c2/report/${r.run}`} onClick={(e) => e.stopPropagation()}>View report</Link></td>
              </tr>))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function Proof({ rec }) {
  const [p, setP] = useState("n1");
  const cur = rec[p];
  return (
    <section className="stack">
      <div className="section-title"><span className="eyebrow">Why trust it · recorded UCI Adult benchmark, 158 monitoring streams, 3 seeds</span><span className="small muted">click a card</span></div>
      <div className="c2-proof">
        {["n1", "n2", "n3"].map((k) => (
          <button key={k} type="button" className="c2-pcard" aria-pressed={p === k} onClick={() => setP(k)}>
            <span className="v">{rec[k].headline}</span><span className="k">{rec[k].headline_label}</span><span className="d">{rec[k].title}</span>
          </button>))}
      </div>
      <div className="card stack tight">
        <h3>{cur.title}</h3>
        <p className="c2-sub">{cur.note}</p>
        {cur.charts.map((c) => (<div key={c.label} className="stack tight">{cur.charts.length > 1 && <span className="small">{c.label}</span>}<ProofBars chart={c} /></div>))}
      </div>
    </section>
  );
}
