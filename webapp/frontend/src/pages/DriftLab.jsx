import { useEffect, useRef, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { api } from "../api.js";
import { useAction, useApi } from "../hooks.js";
import { refreshAlerts } from "../platform.jsx";
import { useSession } from "../session.jsx";
import { c1View, isC1Alert, isClassicAlarm, OUTCOME_CLASS, OUTCOME_HELP, outcomeOf, sg } from "../ui/c1format.js";
import ClassicVsC1 from "../ui/ClassicVsC1.jsx";
import { ErrorBox, PageHeader, Verdict } from "../ui/parts.jsx";
import PipelineSteps from "../ui/PipelineSteps.jsx";
import TruthVsC1 from "../ui/TruthVsC1.jsx";
import "./DriftLab.css";

const TYPE_LABEL = {
  none: "No drift (clean)", marginal_single: "One feature shifts", marginal_multi: "Several features shift",
  interaction: "Hidden interaction", combined: "Shift + interaction",
};
const SHORT = { none: "none", marginal_single: "single", marginal_multi: "multi", interaction: "interaction", combined: "combined" };
const SHOW_OTHER = 8;   // features never drifted in calibration: show this many, then "+N more"

// Presets use the dataset's own feature groups. Strengths are a starting point: tune them here.
function presetSettings(p, ds, F) {
  if (p === "clean") return { type: "none", sel: [] };
  if (p === "harm") return ds === "uci"
    ? { type: "marginal_single", sel: F.high.slice(0, 1), mag: 0.8 }
    : { type: "marginal_multi", sel: F.high.slice(0, 2), mag: 0.8 };
  if (p === "loud") return { type: "marginal_multi", sel: F.low.slice(0, 3), mag: 0.8 };
  return { type: "interaction", sel: F.high.slice(0, 2), mag: 0.8 };
}

export default function DriftLab() {
  // "Run again with these settings" (run detail page) opens /lab?ds=..&type=..&f=..&mag=..&bs=..&seed=..
  const [params] = useSearchParams();
  const fromQuery = useRef(params.get("type") ? params : null);
  const [ds, setDs] = useState(params.get("ds") || "uci");
  const opts = useApi(() => api.labOptions(ds), [ds]);
  const O = opts.data;
  const [type, setType] = useState("none");
  const [sel, setSel] = useState([]);
  const [mag, setMag] = useState(0.8);
  const [bs, setBs] = useState(1500);
  const [seed, setSeed] = useState(42);
  const [preset, setPreset] = useState(null);
  const [showAll, setShowAll] = useState(false);
  const [last, setLast] = useSession("lab:last", null);
  const [runs, setRuns] = useSession("lab:runs", []);
  const [doRun, { busy, error }] = useAction(api.labRun);
  const [mode, setMode] = useState(null);
  const [saveBatch, setSaveBatch] = useState(false);

  const rule = O?.drift_types.find((d) => d.id === type) || { min: 0, max: 0, help: "" };
  const valid = sel.length >= rule.min && sel.length <= rule.max;
  const unseen = O ? sel.filter((f) => O.features.other.includes(f)) : [];

  // when the options for a dataset arrive: settings from the link if any, else default batch size and
  // the "loud but harmless" preset
  useEffect(() => {
    if (!O) return;
    const q = fromQuery.current;
    if (q && O.dataset === (q.get("ds") || "uci")) {
      fromQuery.current = null;
      setType(q.get("type")); setSel((q.get("f") || "").split(",").filter(Boolean));
      setMag(+q.get("mag") || 0.8); setBs(+q.get("bs") || O.default_batch_size); setSeed(+q.get("seed") || 0);
      setPreset(null);
      return;
    }
    setBs(O.default_batch_size);
    apply("loud", O);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [O]);

  function apply(p, o = O) {
    const s = presetSettings(p, ds, o.features);
    setType(s.type); setSel(s.sel); if (s.mag) setMag(s.mag); setPreset(p);
  }
  function toggle(f) {
    setPreset(null);
    setSel((cur) => {
      if (cur.includes(f)) return cur.filter((x) => x !== f);
      const next = [...cur, f];
      return next.length > rule.max ? next.slice(next.length - rule.max) : next;
    });
  }
  async function run(m) {
    setMode(m);
    const res = await doRun({ dataset: ds, drift_type: type, features: sel, magnitude: mag, batch_size: bs, seed, mode: m, save_batch: saveBatch });
    if (!res) return;
    const entry = { ...res, at: Date.now(), dataset: ds };
    setLast(entry);
    setRuns((r) => [...r, entry]);
    refreshAlerts();                       // a "Run C1 only" alert waits for C2 (badge on the C2 tab)
  }

  const chip = (f, cls = "") => (
    <button key={f} type="button" className={`fchip ${cls}`} aria-pressed={sel.includes(f)} onClick={() => toggle(f)}>{f}</button>
  );
  const other = O ? (showAll ? O.features.other : O.features.other.slice(0, SHOW_OTHER)) : [];

  return (
    <>
      <PageHeader eyebrow="Platform · test bench" title="Drift Lab"
        subtitle="Create a known drift, then check the system's answer against the truth. The lab knows the true harm because it keeps the labels; the components never see them."
        chip={O ? `${O.label.split(" (")[0]} · ${O.harm_metric === "auc_drop" ? "AUC" : "accuracy"} drop` : null} />

      <div className="lab-grid">
        <section className="card settings" aria-label="Drift settings">
          <h2>1 · Create a drift</h2>
          <div className="field"><label htmlFor="ds">Dataset</label>
            <select id="ds" value={ds} onChange={(e) => { setDs(e.target.value); setShowAll(false); }}>
              <option value="uci">UCI Adult (income, XGBoost)</option>
              <option value="ieee">IEEE-CIS Fraud (LightGBM)</option>
            </select></div>
          <ErrorBox error={opts.error} />
          {O && (<>
            <div className="field"><span className="lab">Presets</span>
              <div className="presets">
                {[["clean", "Clean"], ["harm", "Harmful"], ["loud", "Loud but harmless"], ["hidden", "Hidden interaction"]].map(([k, l]) => (
                  <button key={k} type="button" className="pbtn" aria-pressed={preset === k} onClick={() => apply(k)}>{l}</button>
                ))}
              </div></div>
            <div className="field"><label htmlFor="type">Drift type</label>
              <select id="type" value={type} onChange={(e) => { setType(e.target.value); setSel([]); setPreset(null); }}>
                {O.drift_types.map((d) => <option key={d.id} value={d.id}>{TYPE_LABEL[d.id]}</option>)}
              </select></div>
            <p className="help">{rule.help}</p>
            {type !== "none" ? (
              <div className="field"><span className="lab">Features</span>
                <div className="grp">High importance</div><div className="chips">{O.features.high.map((f) => chip(f))}</div>
                <div className="grp">Low importance</div><div className="chips">{O.features.low.map((f) => chip(f))}</div>
                {O.features.other.length > 0 && (<>
                  <div className="grp">Not drifted in calibration</div>
                  <div className="chips">
                    {other.map((f) => chip(f, "unseen"))}
                    {O.features.other.length > SHOW_OTHER && (
                      <button type="button" className="fchip more" onClick={() => setShowAll((s) => !s)}>
                        {showAll ? "show fewer" : `+${O.features.other.length - SHOW_OTHER} more`}
                      </button>)}
                  </div>
                </>)}
                <p className="small">
                  {valid
                    ? <span className="muted">{type === "combined" ? `Shift: ${sel[0]} · interaction: ${sel[1]} × ${sel[2]}` : type === "interaction" ? `Interaction: ${sel[0]} × ${sel[1]}` : `${sel.length} selected`}</span>
                    : <span style={{ color: "var(--bad-fg)" }}>Pick {rule.min === rule.max ? rule.min : `${rule.min} to ${rule.max}`} feature{rule.max > 1 ? "s" : ""} ({sel.length} selected)</span>}
                </p>
                {unseen.length > 0 && <div className="warnline">{unseen.join(", ")}: never drifted during calibration. C1 should still detect it, but its interval may be less reliable.</div>}
                {O.note && <p className="small muted">{O.note}</p>}
              </div>
            ) : <p className="small muted">No features for a clean batch.</p>}
            {type !== "none" && (
              <div className="field"><label htmlFor="mag">Strength: <span className="mono">{mag.toFixed(2)}</span></label>
                <input id="mag" type="range" min="0.05" max="1" step="0.05" value={mag} onChange={(e) => { setMag(+e.target.value); setPreset(null); }} /></div>
            )}
            <div className="grid2">
              <div className="field"><label htmlFor="bs">Batch size</label>
                <select id="bs" value={bs} onChange={(e) => setBs(+e.target.value)}>
                  {O.batch_sizes.map((s) => <option key={s} value={s}>{s.toLocaleString()}</option>)}
                </select></div>
              <div className="field"><label htmlFor="seed">Seed</label>
                <input id="seed" type="number" value={seed} onChange={(e) => setSeed(parseInt(e.target.value || "0", 10))} /></div>
            </div>
            <h2 style={{ marginTop: 4 }}>2 · Run</h2>
            <div className="btnrow">
              <button className={`btn primary ${busy ? "busy" : ""}`} type="button" disabled={!valid || busy} onClick={() => run("c1")}>
                {busy && mode === "c1" ? "Running…" : "Run C1 only"}</button>
              <button className={`btn ${busy ? "busy" : ""}`} type="button" disabled={!valid || busy} onClick={() => run("c1c2")}
                title="Runs C1, then C2 on the same batch, and stops before C3">
                {busy && mode === "c1c2" ? "Running…" : "Run C1 → C2"}</button>
              <button className={`btn ${busy ? "busy" : ""}`} type="button" disabled={!valid || busy} onClick={() => run("pipeline")}>
                {busy && mode === "pipeline" ? "Running…" : "Run full pipeline"}</button>
            </div>
            <label className="row small lab-save"><input type="checkbox" checked={saveBatch} onChange={(e) => setSaveBatch(e.target.checked)} />
              Save the batch CSV with this run (about 0.2 MB for UCI, several MB for IEEE)</label>
          </>)}
          {opts.loading && <p className="small muted">Loading features…</p>}
        </section>

        <section className="results" aria-live="polite">
          <ErrorBox error={error} />
          {last ? <LabResult r={last} /> : (
            <div className="empty">Pick a preset and press <b>Run C1 only</b>. Start with <b>Loud but harmless</b>.</div>)}
        </section>
      </div>

      <Scoreboard runs={runs} onClear={() => setRuns([])} />
    </>
  );
}

function injectedTags(inj) {
  if (inj.drift_type === "none") return [];
  if (inj.drift_type === "interaction") return [`${inj.features[0]} × ${inj.features[1]}`];
  if (inj.drift_type === "combined") return [inj.features[0], `${inj.features[1]} × ${inj.features[2]}`];
  return inj.features;
}

function LabResult({ r }) {
  const v = c1View(r.c1);
  if (!v) return <ErrorBox error={`C1 did not return an output (${r.pipeline?.stopped_reason || "unknown"})`} />;
  const t = r.truth.true_harm_pp;
  const inside = t >= v.lo && t <= v.hi;
  const flagged = isC1Alert(v);
  const [title, text, tone] = !v.det && !flagged ? ["No drift", "C1 found nothing that needs attention.", "ok"]
    : !flagged ? ["Drift found, low priority", "Real change, but predicted cost under 1 pp.", "ok"]
      : [`Severity ${v.sev}`, `Predicted cost about ${v.harm.toFixed(1)} pp: repair recommended.`, "warn"];
  const inj = r.injection;
  const shifted = inj.shifted || [];
  const fh = r.truth.feature_harm_pp || {};
  const sm = r.summary || {};
  return (
    <div className="stack" style={{ gap: 14 }}>
      <Verdict title={title} text={text} tone={tone}
        aside={`${inj.dataset.toUpperCase()} · ${SHORT[inj.drift_type]} · ${inj.batch_size.toLocaleString()} rows · seed ${inj.seed}`} />
      {v.warnings.map((w) => <div key={w} className="warnline">{w}</div>)}
      <div className="row lab-runref">
        <span className={`outcome ${OUTCOME_CLASS[outcomeOf(t, v.sev)]}`} title={OUTCOME_HELP[outcomeOf(t, v.sev)]}>{outcomeOf(t, v.sev)}</span>
        {r.run_id && <Link className="linkbtn" to={`/runs/${r.run_id}`}>Open full analysis</Link>}
        {r.run_id && <span className="small muted mono">run {r.run_id}</span>}
      </div>
      <div className="card grid2">
        <div className="kpi"><span className="k">Truth (from labels)</span><span className="v">{sg(t)} pp</span>
          <span className="small muted">{v.metric} on this batch vs the whole test set · paired {sg(r.truth.true_harm_paired_pp)} pp</span></div>
        <div className="kpi"><span className="k">C1 predicted (no labels)</span><span className="v acc">{sg(v.harm)} pp</span>
          <span className="small muted">error {Math.abs(t - v.harm).toFixed(2)} pp</span></div>
      </div>
      <div className="card stack tight">
        <div className="head"><h3>Is the truth inside C1's 95% interval?</h3>
          <span className="chip" style={{ color: inside ? "var(--ok-fg)" : "var(--bad-fg)" }}>{inside ? "✓ inside" : "✗ outside"}</span></div>
        <TruthVsC1 truth={t} harm={v.harm} lo={v.lo} hi={v.hi} />
        <p className="small muted">C1's interval is sized for "{v.ext}" drift and {v.bs.toLocaleString()} rows. About 1 batch in 20 should fall outside.</p>
      </div>
      <ClassicVsC1 v={v} />
      <div className="card stack">
        <h3>Which features drifted?</h3>
        <div className="frow2"><span className="lbl">Injected</span>
          {injectedTags(inj).length ? injectedTags(inj).map((f) => <span key={f} className="tagf none">{f}</span>) : <span className="tagf none">none</span>}</div>
        <div className="frow2"><span className="lbl">C1 found</span>
          {v.aff.length ? v.aff.map((f) => <span key={f} className="tagf">{f}{shifted.includes(f) ? " ✓" : ""}</span>) : <span className="tagf none">none listed</span>}</div>
        {Object.keys(fh).length > 0 && (
          <div className="frow2"><span className="lbl" title="With the labels: put each shifted feature back to normal and measure how much comes back">True harm if undone</span>
            {Object.entries(fh).map(([f, pp]) => (
              <span key={f} className={`tagf ${f === r.truth.main_cause ? "" : "none"}`}>{f} {pp >= 0 ? "+" : ""}{pp.toFixed(2)} pp{f === r.truth.main_cause ? " · main cause" : ""}</span>))}</div>)}
        {r.truth.cause_note && inj.drift_type !== "none" && <p className="small muted">Answer key: {r.truth.cause_note}.</p>}
        <p className="small muted">{inj.drift_type === "interaction"
          ? "An interaction changes no single feature, so none is listed; C1 reports it through the interaction probability."
          : "A feature is listed only when every detector that scores it passes its strictest clean-batch level."}</p>
      </div>
      {r.pipeline && (
        <div className="card stack">
          <div className="head"><h3>{r.mode === "c1c2" ? "C1 → C2" : "Full pipeline"}</h3><span className="small muted mono">run {r.pipeline.run_id} · {r.pipeline.status} · {r.pipeline.runtime_sec} s</span></div>
          <PipelineSteps run={r.pipeline} />
          {sm.c2_root && (
            <p className="small">C2 root cause <b className="mono">{sm.c2_root}</b>{" "}
              {sm.c2_ok === true ? <span style={{ color: "var(--ok-fg)" }}>✓ matches the answer key</span>
                : sm.c2_ok === false ? <span style={{ color: "var(--bad-fg)" }}>✗ answer key: {inj.drift_type === "interaction" ? `interaction ${inj.features.join(" × ")}` : sm.true_cause}</span>
                  : <span className="muted">(no single true cause to grade against)</span>}</p>)}
          <p className="small muted">Badges show what each component did in this run: real, or fake when it used its contract example (the reason is shown under it).</p>
        </div>
      )}
    </div>
  );
}

function Scoreboard({ runs, onClear }) {
  const rows = runs.map((r) => ({ r, v: c1View(r.c1) })).filter((x) => x.v);
  const n = rows.length;
  const inside = rows.filter(({ r, v }) => r.truth.true_harm_pp >= v.lo && r.truth.true_harm_pp <= v.hi).length;
  const mae = n ? rows.reduce((a, { r, v }) => a + Math.abs(r.truth.true_harm_pp - v.harm), 0) / n : 0;
  return (
    <section className="card stack">
      <div className="head"><h3>3 · Session scoreboard</h3>
        <div className="row"><button className="pbtn" type="button" onClick={onClear}>Clear</button>
          <Link className="pbtn lab-histlink" to="/runs">Open run history</Link></div></div>
      <div className="score">
        <div className="s"><b>{n}</b><span>runs</span></div>
        <div className="s"><b>{n ? `${inside}/${n}` : "-"}</b><span>truth inside interval</span></div>
        <div className="s"><b>{n ? `${mae.toFixed(2)} pp` : "-"}</b><span>mean error</span></div>
        <div className="s"><b>{n ? rows.filter(({ v }) => isClassicAlarm(v)).length : "-"}</b><span>classic alarms</span></div>
        <div className="s"><b>{n ? rows.filter(({ v }) => isC1Alert(v)).length : "-"}</b><span>C1 repair alerts</span></div>
      </div>
      <div className="scroll-x">
        <table className="tbl">
          <thead><tr><th>#</th><th>Data</th><th>Drift</th><th>Features</th><th className="mono">Strength</th><th className="mono">Truth</th><th className="mono">C1</th><th>Inside?</th><th>Classic</th><th>C1 says</th><th>Outcome</th><th>C2 cause</th><th /></tr></thead>
          <tbody>
            {n ? rows.slice().reverse().map(({ r, v }, i) => {
              const t = r.truth.true_harm_pp;
              return (
                <tr key={r.at}>
                  <td className="mono">{n - i}</td><td>{r.dataset.toUpperCase()}</td><td>{SHORT[r.injection.drift_type]}</td>
                  <td className="mono">{r.injection.features.join(", ") || "-"}</td>
                  <td className="mono">{r.injection.drift_type === "none" ? "-" : r.injection.magnitude.toFixed(2)}</td>
                  <td className="mono">{sg(t)}</td><td className="mono">{sg(v.harm)}</td>
                  <td>{t >= v.lo && t <= v.hi ? "✓" : "✗"}</td>
                  <td>{isClassicAlarm(v) ? "ALARM" : "-"}</td><td>{isC1Alert(v) ? "repair" : "no repair"}</td>
                  <td><span className={`outcome ${OUTCOME_CLASS[outcomeOf(t, v.sev)]}`} title={OUTCOME_HELP[outcomeOf(t, v.sev)]}>{outcomeOf(t, v.sev)}</span></td>
                  <td className="mono">{r.summary?.c2_root ? `${r.summary.c2_root}${r.summary.c2_ok === true ? " ✓" : r.summary.c2_ok === false ? " ✗" : ""}` : "-"}</td>
                  <td>{r.run_id && <Link className="linkbtn" to={`/runs/${r.run_id}`}>View</Link>}</td>
                </tr>);
            }) : <tr><td colSpan="13" className="muted">No runs yet.</td></tr>}
          </tbody>
        </table>
      </div>
    </section>
  );
}
