import { useEffect, useRef, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { api } from "../api.js";
import { useAction, useApi } from "../hooks.js";
import { refreshAlerts } from "../platform.jsx";
import { useSession } from "../session.jsx";
import { c1View, isC1Alert, isClassicAlarm, OUTCOME_CLASS, OUTCOME_HELP, outcomeOf, sg } from "../ui/c1format.js";
import ClassicVsC1 from "../ui/ClassicVsC1.jsx";
import { Icon } from "../layout/icons.jsx";
import { ErrorBox, PageHeader, Verdict } from "../ui/parts.jsx";
import PipelineSteps from "../ui/PipelineSteps.jsx";
import { stepTile } from "../ui/runTiles.jsx";
import TruthVsC1 from "../ui/TruthVsC1.jsx";
import { MiniBars, Spark } from "../ui/viz.jsx";
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
      <PageHeader icon="lab" eyebrow="Platform · test bench" title="Drift Lab"
        subtitle="Create a known drift, then check every component's answer against the truth. The lab keeps the labels; the components never see them."
        chip={O ? `${O.label.split(" (")[0]} · ${O.harm_metric === "auc_drop" ? "AUC" : "accuracy"} drop` : null} />

      <div className="lab-grid">
        <section className="card settings" aria-label="Drift settings">
          <h2 className="lab-h"><span className="ic xs tone-teal"><Icon name="gear" size={15} className="" strokeWidth={2.2} /></span>1 · Create a drift</h2>
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
            <h2 className="lab-h" style={{ marginTop: 4 }}><span className="ic xs tone-teal"><Icon name="play" size={12} className="" /></span>2 · Run</h2>
            <div className="btnrow">
              <button className={`btn primary full ${busy ? "busy" : ""}`} type="button" disabled={!valid || busy} onClick={() => run("pipeline")}>
                <Icon name="play" size={14} className="" />{busy && mode === "pipeline" ? "Running…" : "Run full pipeline · C1 → C4"}</button>
              <button className={`btn ${busy ? "busy" : ""}`} type="button" disabled={!valid || busy} onClick={() => run("c1")}>
                {busy && mode === "c1" ? "Running…" : "C1 only"}</button>
              <button className={`btn ${busy ? "busy" : ""}`} type="button" disabled={!valid || busy} onClick={() => run("c1c2")}
                title="Runs C1, then C2 on the same batch, and stops before C3">
                {busy && mode === "c1c2" ? "Running…" : "C1 → C2"}</button>
            </div>
            <label className="row small lab-save"><input type="checkbox" checked={saveBatch} onChange={(e) => setSaveBatch(e.target.checked)} />
              Save the batch CSV with this run (about 0.2 MB for UCI, several MB for IEEE)</label>
          </>)}
          {opts.loading && <p className="small muted">Loading features…</p>}
        </section>

        <section className="results" aria-live="polite">
          <ErrorBox error={error} />
          {last ? <LabResult r={last} runs={runs} /> : (
            <div className="empty">Pick a preset and press <b>Run full pipeline</b> (or <b>C1 only</b>). Start with <b>Loud but harmless</b>.</div>)}
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

// this session's runs (oldest -> newest) as small chart series
function sessionSeries(runs) {
  const rows = runs.map((r) => ({ r, v: c1View(r.c1) })).filter((x) => x.v).slice(-12);
  return {
    truth: rows.map(({ r }) => r.truth.true_harm_pp),
    c1: rows.map(({ v }) => v.harm),
    inside: rows.map(({ r, v }) => { const ok = r.truth.true_harm_pp >= v.lo && r.truth.true_harm_pp <= v.hi; return { v: ok ? 1 : 0.25, c: ok ? "#22A05B" : "#F04438" }; }),
    c2: rows.filter(({ r }) => r.summary?.c2_ok === true || r.summary?.c2_ok === false).map(({ r }) => ({ v: r.summary.c2_ok ? 1 : 0.3, c: r.summary.c2_ok ? "#22A05B" : "#B7DFC6" })),
    c3: rows.filter(({ r }) => r.pipeline?.outputs?.c3).map(({ r }) => {
      const ok = r.pipeline.outputs.c3.repair_package_status === "APPROVED";
      return { v: ok ? 1 : 0.6, c: ok ? "#16A34A" : "#98A2B3" };
    }),
  };
}

function LabResult({ r, runs }) {
  const v = c1View(r.c1);
  if (!v) return <ErrorBox error={`C1 did not return an output (${r.pipeline?.stopped_reason || "unknown"})`} />;
  const t = r.truth.true_harm_pp;
  const inside = t >= v.lo && t <= v.hi;
  const flagged = isC1Alert(v);
  const inj = r.injection;
  const shifted = inj.shifted || [];
  const fh = r.truth.feature_harm_pp || {};
  const sm = r.summary || {};
  const [title, text, tone] = !v.det && !flagged ? ["No drift", "C1 found nothing that needs attention.", "ok"]
    : !flagged ? ["Drift found, low priority", "Real change, but predicted cost under 1 pp.", "ok"]
      : [`Harmful drift · severity ${v.sev}`, `${(shifted.length ? shifted : injectedTags(inj)).join(", ") || "The data"} shifted. C1 predicts the model loses ${v.harm.toFixed(1)} pp of ${v.metric}; the truth is ${t.toFixed(1)} pp.`, "warn"];
  const ss = sessionSeries(runs);
  const n = ss.c1.length;
  const pipe = r.pipeline;
  const tiles = ["c2", "c3", "c4"].map((c) => [c, stepTile(c, pipe, sm)]);
  const label = { c2: "C2 cause", c3: "C3 repair", c4: "C4 release" };
  const icon = { c2: "diagnose", c3: "repair", c4: "release" };
  const chart = { c2: <MiniBars items={ss.c2} w={110} h={24} caption={ss.c2.length > 1 ? `C2 correct, last ${ss.c2.length}` : null} />,
    c3: <MiniBars items={ss.c3} w={110} h={24} caption={ss.c3.length > 1 ? `C3 results, last ${ss.c3.length}` : null} />, c4: null };
  return (
    <div className="stack" style={{ gap: 16 }}>
      <Verdict title={title} text={text} tone={tone}
        aside={`${inj.dataset.toUpperCase()} · ${SHORT[inj.drift_type]} · ${inj.batch_size.toLocaleString()} rows · seed ${inj.seed}`} />
      {v.warnings.map((w) => <div key={w} className="warnline">{w}</div>)}
      <div className="row lab-runref">
        <span className={`outcome ${OUTCOME_CLASS[outcomeOf(t, v.sev)]}`} title={OUTCOME_HELP[outcomeOf(t, v.sev)]}>{outcomeOf(t, v.sev)}</span>
        {r.run_id && <Link className="linkbtn" to={`/runs/${r.run_id}`}>Open full analysis</Link>}
        {r.run_id && <span className="small muted mono">run {r.run_id}</span>}
      </div>

      <section className="lab-strip" aria-label="Every component against the truth">
        <div className="card lab-tile">
          <div className="lt-h"><span className="ic xs tone-amber"><Icon name="bars" size={14} className="" strokeWidth={2.4} /></span>True harm</div>
          <span className="num lt-v" style={{ color: "var(--t-amber-ic)" }}>{sg(t)}<small> pp</small></span><span className="lt-s">measured with labels</span>
          <Spark values={ss.truth} w={110} h={26} color="#E69F00" caption={n > 1 ? `truth, last ${n} runs` : null} />
        </div>
        <div className="card lab-tile">
          <div className="lt-h"><span className="ic xs tone-teal"><Icon name="detect" size={14} className="" strokeWidth={2.4} /></span>C1 predicted</div>
          <span className="num lt-v" style={{ color: "var(--accent)" }}>{sg(v.harm)}<small> pp</small></span><span className="lt-s">no labels used</span>
          <Spark values={ss.c1} w={110} h={26} color="#0D9488" caption={n > 1 ? `C1, last ${n} runs` : null} />
        </div>
        <div className={`ttile lab-tile tone-${inside ? "green" : "red"}`}>
          <div className="lt-h"><span className="ic xs"><Icon name={inside ? "check" : "warn"} size={14} className="" strokeWidth={2.6} /></span>Inside 95% interval</div>
          <span className="lt-v" style={{ fontWeight: 800 }}>{inside ? "Yes" : "No"}</span><span className="lt-s mono">{sg(v.lo)} to {sg(v.hi)}</span>
          <MiniBars items={ss.inside} w={110} h={24} caption={n > 1 ? `inside, last ${n}` : null} />
        </div>
        {tiles.map(([c, tl]) => (
          <div key={c} className={`ttile lab-tile tone-${tl.tone} ${tl.source === "fake" || tl.skipped ? "dashed" : ""}`}>
            <div className="lt-h"><span className="ic xs"><Icon name={icon[c]} size={14} className="" strokeWidth={2.2} /></span>{label[c]}</div>
            <span className={`lt-t ${tl.mono ? "mono" : ""}`}>{tl.title}</span>
            <span className="lt-s">{tl.skipped ? (r.mode === "c1" ? "not run: C1 only" : tl.text) : c === "c2" ? (sm.c2_ok === true ? "✓ matches answer key" : sm.c2_ok === false ? `✗ answer key: ${sm.true_cause || "-"}` : "") : (tl.text || "").split(" · ")[0]}</span>
            {!tl.skipped && chart[c]}
          </div>))}
      </section>

      <div className="card stack">
        <div className="head"><h3 className="row" style={{ gap: 10 }}><Icon name="bars" size={20} className="" strokeWidth={2.4} style={{ color: "#2563EB" }} />Is the truth inside C1's 95% interval?</h3>
          <span className={`pill tone-${inside ? "green" : "red"}`}>{inside ? "✓ inside" : "✗ outside"}</span></div>
        <TruthVsC1 truth={t} harm={v.harm} lo={v.lo} hi={v.hi} />
        <p className="small muted">C1's interval is sized for "{v.ext}" drift and {v.bs.toLocaleString()} rows. About 1 batch in 20 should fall outside.</p>
      </div>

      {pipe && (
        <div className="card stack" style={{ gap: 14 }}>
          <div className="head"><h3 className="row" style={{ gap: 10 }}><Icon name="nodes" size={20} className="" style={{ color: "var(--accent)" }} />{r.mode === "c1c2" ? "C1 → C2" : "Full pipeline"}</h3>
            <span className="small muted mono">run {pipe.run_id} · {pipe.status.replaceAll("_", " ")} · {pipe.runtime_sec} s</span></div>
          <PipelineSteps run={pipe} grade={sm} />
          <p className="small muted">Badges show what each component did in this run: real, or fake when it used its contract example (the reason is shown under it).</p>
        </div>
      )}

      <ClassicVsC1 v={v} />

      <div className="card stack">
        <h3 className="row" style={{ gap: 10 }}><Icon name="target" size={20} className="" style={{ color: "var(--accent)" }} />Which feature drifted, and did it matter?</h3>
        <div className="lab-feat">
          <div><span className="lbl">Injected</span><div className="row">
            {injectedTags(inj).length ? injectedTags(inj).map((f) => <span key={f} className="tagf none">{f}</span>) : <span className="tagf none">none</span>}</div></div>
          <div><span className="lbl">C1 found</span><div className="row">
            {v.aff.length ? v.aff.map((f) => <span key={f} className="tagf">{f}{shifted.includes(f) ? " ✓" : ""}</span>) : <span className="tagf none">none listed</span>}</div></div>
          {Object.keys(fh).length > 0 && (
            <div><span className="lbl" title="With the labels: put each shifted feature back to normal and measure how much comes back">True harm if undone</span><div className="row">
              {Object.entries(fh).map(([f, pp]) => (
                <span key={f} className={`tagf ${f === r.truth.main_cause ? "harm" : "none"}`}>{f} {pp >= 0 ? "+" : ""}{pp.toFixed(2)} pp{f === r.truth.main_cause ? " · main cause" : ""}</span>))}</div></div>)}
        </div>
        {r.truth.cause_note && inj.drift_type !== "none" && <p className="small muted">Answer key: {r.truth.cause_note}.</p>}
        <p className="small muted row" style={{ gap: 8, flexWrap: "nowrap" }}><Icon name="info" size={15} className="" style={{ flex: "none" }} />{inj.drift_type === "interaction"
          ? "An interaction changes no single feature, so none is listed; C1 reports it through the interaction probability."
          : "A feature is listed only when every detector that scores it passes its strictest clean-batch level."}</p>
      </div>
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
      <div className="head"><h3 className="row" style={{ gap: 10 }}><Icon name="bars" size={20} className="" strokeWidth={2.4} style={{ color: "#2563EB" }} />3 · Session scoreboard</h3>
        <div className="row"><button className="pbtn" type="button" onClick={onClear}>Clear</button>
          <Link className="pbtn lab-histlink" to="/runs">Open run history</Link></div></div>
      <div className="score">
        <div className="s"><b>{n}</b><span>runs</span></div>
        <div className="s"><b>{n ? `${inside}/${n}` : "-"}</b><span>truth inside interval</span></div>
        <div className="s"><b>{n ? `${mae.toFixed(2)} pp` : "-"}</b><span>mean error</span></div>
        <div className="s"><b>{n ? rows.filter(({ v }) => isClassicAlarm(v)).length : "-"}</b><span>classic alarms</span></div>
        <div className="s"><b>{n ? rows.filter(({ v }) => isC1Alert(v)).length : "-"}</b><span>C1 repair alerts</span></div>
        {n > 1 && (
          <div className="s pairs" role="img" aria-label="Truth and C1 per run">
            <div className="pbars">{rows.slice(-10).map(({ r, v }, i) => {
              const m = Math.max(1, ...rows.slice(-10).map((x) => Math.max(x.r.truth.true_harm_pp, x.v.harm)));
              return (<span key={i} className="pb"><i style={{ height: `${Math.max(6, (Math.max(0, r.truth.true_harm_pp) / m) * 100)}%`, background: "#F5B740" }} />
                <i style={{ height: `${Math.max(6, (Math.max(0, v.harm) / m) * 100)}%`, background: "#0D9488" }} /></span>);
            })}</div>
            <span>truth / C1 per run</span>
          </div>)}
      </div>
      <div className="scroll-x">
        <table className="tbl">
          <thead><tr><th>#</th><th>Data</th><th>Drift</th><th>Features</th><th className="mono">Strength</th><th className="mono">Truth</th><th className="mono">C1</th><th>Inside?</th><th>Classic</th><th>C1 says</th><th>Outcome</th><th>C2 cause</th><th className="mono">Time</th><th /></tr></thead>
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
                  <td className="mono">{r.pipeline ? `${r.pipeline.runtime_sec} s` : "-"}</td>
                  <td>{r.run_id && <Link className="linkbtn" to={`/runs/${r.run_id}`}>View</Link>}</td>
                </tr>);
            }) : <tr><td colSpan="14" className="muted">No runs yet.</td></tr>}
          </tbody>
        </table>
      </div>
    </section>
  );
}
