// C2 Root Cause Diagnosis page (Tharmika N, IT22247940).
// Shows the latest stored run (or the one picked from the history), its temporal SHAP analysis across
// all stored runs, the evidence behind the root cause, the affected group, and the run history.
// The business report opens on its own page (/c2/report/:run) in the same tab.
import { useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../../api.js";
import { useAction, useApi } from "../../hooks.js";
import { useSession } from "../../session.jsx";
import DriftAlerts from "../../ui/DriftAlerts.jsx";   // platform: drift alerts waiting for C2 (v0.3)
import { Empty, ErrorBox, JsonBox, PageHeader } from "../../ui/parts.jsx";
import {
  EvidenceChart, HeatKey, ProofBars, RelianceChart, SIGNALS, ShareBars, TemporalChart, TemporalLegend,
  temporalWindow, ZHeatmap,
} from "./C2Charts.jsx";
import { answer, baselines, c1PredOf, clock, decOf, harmOf, pct, runsAgreeing, sg, STATE, time } from "./c2format.js";
import "./C2Page.css";

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
  const [doReset, reset] = useAction(api.c2Reset);

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

      <DriftAlerts onDiagnosed={() => { setSel(null); hist.reload(); }} />
      <SourceBar busy={reset.busy} ready={!!info.data} onReset={newPeriod} />
      <ErrorBox error={info.error || hist.error || reset.error} />

      <section className="stack" style={{ gap: 12 }} aria-live="polite">
        {rec.data ? <Answer rec={rec.data} latest={latest} />
          : hist.loading || rec.loading ? <p className="muted">Loading…</p>
            : <Empty>No diagnosis stored yet. Create test batches in the <Link to="/lab">Drift Lab</Link> (<b>Run C1 → C2</b>, or <b>Run C1 only</b> and then <b>Diagnose</b> the alert above), two or three times: C2 compares every run with the runs before it, so a cause is confirmed only when it persists.</Empty>}
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
      {rec.data?.c1_output && <JsonBox title={`C1 output C2 received (${rec.data.c1_source === "fake" ? "fake C1" : "real C1"}, contract v1.1)`} data={rec.data.c1_output}
        note={rec.data.c1_source === "fake"
          ? "Built by the team's fake C1 (shared/contracts/c1_fake_output.py) from the answer key, because the C1 engine is not on this machine. Same fields and types as the real C1, so this tests the C1 → C2 handoff. C2 uses marginal_extent and interaction_suspected as hints for the drift category. predicted_harm here is copied from the true drop, so the page does not show it as a C1 prediction."
          : "Produced by the real C1 engine on this same batch. C2 uses marginal_extent and interaction_suspected as hints for the drift category and shows predicted_harm next to its own measurement."} />}
      <JsonBox title="Raw output to C3 and C4 (JSON, contract v1.1)" data={rec.data ? { to_c3: rec.data.to_c3, to_c4: rec.data.to_c4 } : null}
        note="The exact messages C2 sends for the run on screen." />
    </>
  );
}

// v0.3: the drift simulator moved out of this page. Test batches come from the Drift Lab
// (Run C1 -> C2, or Run C1 only + Diagnose an alert). The /api/c2/run route stays for research use.
function SourceBar({ busy, ready, onReset }) {
  return (
    <section className="card c2-sourcebar" aria-label="Where C2's batches come from">
      <span className="small muted">C2 diagnoses batches that C1 flagged. Create test batches in the <Link to="/lab"><b>Drift Lab</b></Link>: <b>Run C1 → C2</b>, or <b>Run C1 only</b> and press <b>Diagnose</b> on the alert above.</span>
      <button className="btn" type="button" disabled={busy || !ready} onClick={onReset}>{busy ? "Clearing…" : "New period"}</button>
    </section>
  );
}

function Answer({ rec, latest }) {
  const st = rec.state, s = STATE[st], c3 = rec.to_c3, a = answer(rec), harm = harmOf(rec), dec = decOf(rec), c1 = c1PredOf(rec);
  const agree = runsAgreeing(rec), stab = c3.diagnosis_stability, group = rec.plain?.group || c3.affected_slice_rule;
  const verdictText = {
    clean: "Nothing needs repair in this batch.",
    harmless: "It changed, but it barely changes the model's decisions.",
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
          <span className="a">{sg(dec)}%</span><span className="s">of decisions flip back when {c3.root_cause_feature} is put back to normal · score {c3.drift_impact_score.toFixed(2)}{harm != null ? ` · est. accuracy loss ${sg(harm)} pp` : ""}{c1 != null ? ` · C1 predicted ${sg(c1)} pp` : ""}</span></div>
        <div><span className="q">Who is affected?</span>
          {st === "clean" || st === "harmless" ? <><span className="a sm">No group</span><span className="s">nothing sent to C3 for repair</span></>
            : <><span className="a">{pct(c3.slice_support)}</span><span className="s">of this batch ({a.rows.toLocaleString()} rows){rec.localization.ref_share != null ? ` · ${pct(rec.localization.ref_share, 1)} in training data` : ""}</span></>}</div>
      </div>
      {rec.truth && rec.injection?.drift_type !== undefined && (
        <div className="c2-truth"><span className="eyebrow">Answer key · hidden from C2</span>
          <span>true accuracy drop <b className="mono">{sg(rec.truth.true_harm_pp)} pp</b></span>
          <span>C2 decisions changed <b className="mono">{sg(dec)}%</b></span>
          <span>injected <b className="mono">{rec.truth.true_causes.join(", ") || "nothing"}</b></span>
          <span>C2 root cause <b className="mono">{c3.root_cause_feature}</b> {rec.truth.true_causes.length ? (rec.truth.true_causes.includes(c3.root_cause_feature) ? "✓" : "✗") : ""}</span>
        </div>)}
      <div className="row" style={{ justifyContent: "space-between" }}>
        <div className="row">
          <span className={`badge ${rec.c1?.status === "ok" ? "ok" : ""}`} title="C1's JSON is checked against the C1 contract before C2 uses it">
            C1 input: {rec.c1?.status || "missing"}{rec.c1?.harm_severity ? ` · ${rec.c1.harm_severity}` : ""}{rec.c1_source === "fake" ? " · fake C1" : rec.c1_source === "real" ? " · real C1" : ""}</span>
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
        {`${loc.ref_share ? `This group is ${(c3.slice_support / loc.ref_share).toFixed(1)} times more common now than in training. ` : ""}About ${Math.round(loc.rows_at_risk || 0)} decisions change because of the drift; the group holds an estimated ${pct(loc.est_recall)} of the movement. Found without labels.`}
        {" "}Groups under 20 rows are never reported (privacy).{rec.plain && !rec.plain.decoded ? " Values are scaled: add preprocessor.pkl to decode them." : ""}
      </p>
    </div>
  );
}

function Compare({ rec }) {
  const b = baselines(rec), clean = rec.state === "clean", dec = decOf(rec);
  return (
    <div className="stack">
      <div className="section-title"><span className="eyebrow">What simpler tools would say about this run</span></div>
      <div className="c2-cmp">
        <div className="cmp"><span className="k">A drift monitor ranks</span><span className="v mono">{b.drift.join(", ")}</span><span className="gap">Says what changed, not what hurts.</span></div>
        <div className="cmp"><span className="k">One SHAP snapshot ranks</span><span className="v mono">{b.snapshot.join(", ")}</span><span className="gap">{clean ? "Names a cause even on clean data." : "No noise check, no run history."}</span></div>
        <div className="cmp c2-ans"><span className="k">C2 says</span><span className="v mono">{clean ? "no alarm" : `${rec.to_c3.root_cause_feature}${dec != null ? ` · ${sg(dec)}% decisions` : ""}`}</span>
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
            <th className="num">Decisions changed</th><th className="num">True drop</th><th className="num">Stability</th><th>Report</th></tr></thead>
          <tbody>
            {runs.map((r) => (
              <tr key={r.run} aria-selected={r.run === current} tabIndex="0" onClick={() => onSelect(r.run)}
                onKeyDown={(e) => { if (e.key === "Enter") onSelect(r.run); }}>
                <td className="mono">#{r.run}{r.run === latest && <span className="badge" style={{ marginLeft: 6 }}>latest</span>}</td>
                <td className="mono">{clock(r.timestamp)}</td>
                <td>{r.source.replace("Drift Lab · ", "")}</td>
                <td><StatePill state={r.state} /></td>
                <td className="mono">{r.root_cause_feature || "none"}</td>
                <td className="num">{r.decision_change_pct == null ? "–" : `${sg(r.decision_change_pct)}%`}</td>
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
      <div className="section-title"><span className="eyebrow">Why trust it · recorded UCI Adult benchmark, no labels used, graded with the answer key</span><span className="small muted">click a card</span></div>
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
