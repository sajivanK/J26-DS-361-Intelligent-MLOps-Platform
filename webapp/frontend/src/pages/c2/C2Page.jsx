// C2 Root Cause Diagnosis page (Tharmika N, IT22247940), v2 design. Two tabs:
//   Diagnosis   C1's drift alerts, and the full diagnosis of the selected run: cause and value range,
//               damage, group, the evidence, temporal SHAP across runs, run history, recorded results.
//   Monitoring  temporal SHAP on EVERY batch C1 checked (not only alerts): C2's own alarms, the
//               timeline, the watchlist, the heatmap and the batch log (C2Monitor.jsx).
// The business report opens on its own page (/c2/report/:run) in the same tab.
import { useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../../api.js";
import { useAction, useApi } from "../../hooks.js";
import { refreshAlerts } from "../../platform.jsx";
import { useSession } from "../../session.jsx";
import { Empty, ErrorBox, JsonBox, PageHeader } from "../../ui/parts.jsx";
import {
  Donut, EvidenceChart, HeatKey, LINE_COLS, ProofBars, RangeBar, RelianceChart, SIGNALS, TemporalChart,
  temporalWindow, topFeatures, ZHeatmap,
} from "./C2Charts.jsx";
import C2Monitor from "./C2Monitor.jsx";
import { I } from "./c2icons.jsx";
import { answer, baselines, shortSource, c1PredOf, decOf, harmOf, pct, relianceOf, runsAgreeing, sg, STATE, time } from "./c2format.js";
import "./C2Page.css";

export function StatePill({ state }) {
  const s = STATE[state] || STATE.clean;
  return <span className={`c2-pill ${s.pill}`}>{s.label}</span>;
}

export default function C2Page() {
  const info = useApi(api.c2Info, []);
  const hist = useApi(api.c2Runs, []);
  const [tab, setTab] = useSession("c2:tab", "diagnosis");
  const [sel, setSel] = useSession("c2:sel", null);        // null = always show the latest run
  const runs = hist.data?.runs || [];
  const latest = runs[0]?.run ?? null;
  const n = sel ?? latest;
  const rec = useApi(() => (n ? api.c2Run(n) : Promise.resolve(null)), [n, latest]);
  const mon = useApi(api.c2Monitor, [tab === "monitoring", latest]);
  const [doReset, reset] = useAction(api.c2Reset);

  async function resetMonitoring() {
    if (!window.confirm("Reset monitoring? This clears the C2 run history and the monitoring log, and relearns normal noise from clean data.")) return;
    if (await doReset()) { setSel(null); hist.reload(); mon.reload(); }
  }
  const pick = (r) => { setSel(r === latest ? null : r); setTab("diagnosis"); window.scrollTo({ top: 0, behavior: "smooth" }); };
  const onDiagnosed = () => { setSel(null); hist.reload(); mon.reload(); };
  const alarm = mon.data?.c2_alarm;

  return (
    <div className="c2x">
      <PageHeader eyebrow="Component 2 · Tharmika N" title="C2 Root Cause Diagnosis"
        subtitle="A drift alarm says something changed. C2 finds the feature whose change actually hurts the model, and the group of records it hurts."
        chip={info.data ? `engine 1.0 · contract v1.1 · ${info.data.runs_stored ?? runs.length} runs · ${mon.data?.kpis?.total ?? info.data.batches_monitored ?? 0} batches monitored` : "contract v1.1"} />

      <div className="c2-tabs" role="tablist" aria-label="C2 views">
        <button type="button" role="tab" aria-selected={tab === "diagnosis"} onClick={() => setTab("diagnosis")}>
          <I n="search" size={17} />Diagnosis</button>
        <button type="button" role="tab" aria-selected={tab === "monitoring"} onClick={() => setTab("monitoring")}>
          <I n="pulse" size={17} />Monitoring{alarm && <span className="c2-chip red" style={{ padding: "1px 8px", fontSize: ".72rem" }}>C2 alarm</span>}</button>
      </div>
      <ErrorBox error={info.error || hist.error || reset.error} />

      {tab === "monitoring" ? (
        <C2Monitor mon={mon} onDiagnosed={onDiagnosed} onOpenRun={pick} onReset={resetMonitoring} resetting={reset.busy} />
      ) : (<>
        <Alerts onDiagnosed={onDiagnosed} />
        <section className="c2x" aria-live="polite">
          {rec.data ? <Answer rec={rec.data} latest={latest} />
            : hist.loading || rec.loading ? <p className="muted">Loading…</p>
              : <Empty>No diagnosis stored yet. Create test batches in the <Link to="/lab">Drift Lab</Link> (<b>Run C1 → C2</b>, or <b>Run C1 only</b> and then <b>Diagnose</b> the alert above), two or three times: C2 compares every run with the runs before it, so a cause is confirmed only when it persists.</Empty>}
        </section>
        {rec.data && (<>
          <section className="c2-grid2">
            <div className="c2-card">
              <div className="c2-head"><span className="c2-tile t-blue"><I n="bars" size={28} /></span>
                <div className="t"><h3>What the model relies on: normal vs this run</h3>
                  <p>Average influence of each feature on a prediction (mean |SHAP|). Grey dot: normal, from clean data. Coloured dot: this run. Orange: the model leans on it more than usual; blue: less.</p></div></div>
              <RelianceChart rec={rec.data} />
              <div className="c2-legendrow"><span><i className="c2-dot" style={{ background: "var(--c2-dot)", margin: 0 }} />Normal</span>
                <span><i className="c2-dot" style={{ background: "var(--c2-orange)", margin: 0 }} />More than usual</span>
                <span><i className="c2-dot" style={{ background: "var(--c2-blue)", margin: 0 }} />Less than usual</span></div>
            </div>
            <div className="c2-card">
              <div className="c2-head"><span className="c2-tile t-purple"><I n="target" size={28} /></span>
                <div className="t"><h3>Why this feature: the evidence behind the score</h3>
                  <p>Each bar shows the Drift Impact Score (0 to 1), split into the weighted evidence that built it.</p></div>
                <span title="Score = 0.35 × explanation change + 0.15 × data drift + 0.35 × decision change + 0.15 × prediction shift, each scaled to 0–1. No labels are used." style={{ color: "var(--c2-ink-3)", padding: 6 }}><I n="info" size={22} sw={1.8} /></span></div>
              <EvidenceChart rec={rec.data} />
              <div className="c2-legend4">{SIGNALS.map(([k, name, d, col, bg]) => (
                <div key={k} style={{ background: bg }}><i className="c2-dot" style={{ background: col, marginTop: 2 }} /><span><b>{name}</b><span className="s">{d}</span></span></div>))}</div>
            </div>
          </section>
          {hist.data?.temporal_shap?.runs?.length > 0 && <Temporal matrix={hist.data.temporal_shap} rec={rec.data} onSelect={pick} />}
          <section className="c2-grid2" style={{ gridTemplateColumns: "minmax(0, 1.05fr) minmax(0, .95fr)" }}>
            <Who rec={rec.data} />
            <Compare rec={rec.data} />
          </section>
        </>)}
        {runs.length > 0 && <History runs={runs} current={n} latest={latest} onSelect={pick} onReset={resetMonitoring} resetting={reset.busy} />}
        {info.data?.recorded && <Proof rec={info.data.recorded} />}
        {rec.data?.c1_output && <JsonBox title={`C1 output C2 received (${rec.data.c1_source === "fake" ? "fake C1" : "real C1"}, contract v1.1)`} data={rec.data.c1_output}
          note={rec.data.c1_source === "fake"
            ? "Built by the team's fake C1 from the answer key, because the C1 engine is not on this machine. C2 uses marginal_extent and interaction_suspected as hints; predicted_harm here copies the true drop, so it is not shown as a prediction."
            : "Produced by the real C1 engine on this same batch. C2 uses marginal_extent and interaction_suspected as hints for the drift category and shows predicted_harm next to its own measurement."} />}
        <JsonBox title="Raw output to C3 and C4 (JSON, contract v1.1)" data={rec.data ? { to_c3: rec.data.to_c3, to_c4: rec.data.to_c4 } : null}
          note="The exact messages C2 sends for the run on screen." />
      </>)}
    </div>
  );
}

// ── Drift alerts: C1 found harmful drift; Diagnose runs C2 on the same batch ──────────────────────
const SEV = { none: ["None", "grey", "var(--c2-dot)"], low: ["Low", "green", "var(--c2-green)"], moderate: ["Moderate", "amber", "var(--c2-amber)"],
  high: ["High", "orange", "var(--c2-orange)"], critical: ["Critical", "red", "var(--c2-red)"] };
const SRC = { drift_lab: "Drift Lab", c1_upload: "C1 page upload", pipeline_upload: "Pipeline upload" };

function Alerts({ onDiagnosed }) {
  const list = useApi(api.alerts, []);
  const [doDiag, diag] = useAction(api.alertDiagnose);
  const [doDismiss, dis] = useAction(api.alertDismiss);
  const [filter, setFilter] = useSession("c2:alertsev", "all");
  const [busyId, setBusyId] = useState(null);
  const items = list.data?.items || [];
  if (list.loading && !list.data) return null;
  if (!items.length) return null;
  const waiting = items.filter((i) => i.alert === "waiting");
  const shown = items.filter((i) => filter === "all" || i.severity === filter).slice(0, 12);
  const busy = diag.busy || dis.busy;
  async function act(fn, id, diagnosed) {
    setBusyId(id);
    const out = await fn(id);
    setBusyId(null);
    if (!out) return;
    list.reload(); refreshAlerts();
    if (diagnosed) onDiagnosed?.(out);
  }
  const what = (s) => (s.source === "drift_lab"
    ? `${(s.drift_type || "").replace("marginal_", "")} · ${(s.features || []).join(", ") || "clean"}${s.magnitude ? ` · ${s.magnitude}` : ""}`
    : s.file ? String(s.file).split(/[\\/]/).pop() : "uploaded batch");
  const sevs = ["all", "low", "moderate", "high", "critical"];
  const copy = (t) => { try { navigator.clipboard.writeText(t); } catch { /* clipboard not available */ } };
  return (
    <section className="c2-card" aria-label="Drift alerts">
      <div className="c2-headrow">
        <div className="c2-head"><span className="c2-tile sm t-purple"><I n="bell" size={24} /></span>
          <div className="t"><h3>Drift alerts</h3><span className="d">Runs where C1 found harmful drift (Drift Lab "Run C1 only", the C1 page and uploads)</span></div></div>
        <div className="c2-seg" role="group" aria-label="Filter by C1 severity">
          {sevs.map((k) => <button key={k} type="button" aria-pressed={filter === k} onClick={() => setFilter(k)}>{k === "all" ? "All" : SEV[k][0]}</button>)}</div>
      </div>
      <div className="c2-grid4">
        <Kpi tone="k-blue" tile="t-blue" icon="clipboard" k="Total runs" v={items.length} />
        <Kpi tone="k-green" tile="t-green" icon="check" k="Diagnosed" v={items.filter((i) => i.alert === "diagnosed").length} />
        <Kpi tone="k-amber" tile="t-amber" icon="clock" k="Waiting" v={waiting.length} />
        <Kpi tone="k-red" tile="t-red" icon="warn" k="Critical" v={items.filter((i) => i.severity === "critical").length} />
      </div>
      <div className="c2-tbl-wrap">
        <table className="c2-tbl" style={{ minWidth: 960 }}>
          <thead><tr>{["Run", "When", "Source", "Drift", "C1 severity", "C1 harm", "Status"].map((h) => <th key={h} scope="col">{h}</th>)}<th scope="col">Actions</th></tr></thead>
          <tbody>
            {shown.map((s) => {
              const sv = SEV[s.severity] || SEV.none;
              const edge = s.alert === "waiting" ? "var(--c2-amber)" : s.severity === "critical" ? "var(--c2-red)" : "var(--c2-blue)";
              return (
                <tr key={s.run_id}>
                  <td className="c2-edge" style={{ "--edge": edge, fontWeight: 400 }}>
                    <span style={{ display: "inline-flex", gap: 6, alignItems: "center" }}><Link className="c2-mono" to={`/runs/${s.run_id}`}>{s.run_id}</Link>
                      <button type="button" aria-label={`Copy run id ${s.run_id}`} onClick={() => copy(s.run_id)} style={{ background: "none", border: 0, padding: 4, cursor: "pointer", color: "var(--c2-ink-3)", display: "flex" }}><I n="copy" size={15} /></button></span></td>
                  <td style={{ whiteSpace: "nowrap" }}>{new Date(s.saved_at).toLocaleDateString([], { month: "short", day: "numeric", year: "numeric" })}<span className="c2-sub2">{new Date(s.saved_at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</span></td>
                  <td><span className="c2-srcchip"><I n="db" size={14} />{SRC[s.source] || s.source} · {s.dataset.toUpperCase()}</span></td>
                  <td style={{ maxWidth: 200 }}>{what(s)}</td>
                  <td><span className={`c2-chip ${sv[1]}`}><i className="c2-dot" style={{ background: sv[2], width: 8, height: 8, margin: 0 }} />{sv[0]}</span></td>
                  <td style={{ minWidth: 110 }}><b style={{ fontWeight: 500 }}>{sg(s.c1_pp, 2)} pp</b>
                    <span className="c2-bar" style={{ marginTop: 6 }}><span style={{ width: `${Math.min(100, ((s.c1_pp || 0) / 18) * 100)}%`, background: sv[2] }} /></span></td>
                  <td>{s.alert === "waiting" ? <span className="c2-chip amber"><I n="clock" size={15} />Waiting</span>
                    : s.alert === "diagnosed" ? (<span style={{ display: "flex", flexDirection: "column", gap: 4, alignItems: "flex-start" }}>
                      <span className="c2-chip green"><I n="checkCircle" size={15} />Diagnosed</span>
                      <span className="c2-sub2">{s.c2_category}: {s.c2_root}{s.c2_ok === true ? " ✓" : s.c2_ok === false ? ` ✗ (truth: ${s.true_cause || "–"})` : ""}</span></span>)
                      : s.alert === "unsupported" ? <span className="c2-sub2">C2 supports UCI only</span>
                        : <span className="c2-chip grey">Dismissed</span>}</td>
                  <td style={{ whiteSpace: "nowrap" }}>{s.alert === "waiting" ? (
                    <span style={{ display: "inline-flex", gap: 6 }}>
                      <button className="c2-btn primary" type="button" disabled={busy} onClick={() => act(doDiag, s.run_id, true)}><I n="search" size={16} />{busyId === s.run_id && diag.busy ? "Diagnosing…" : "Diagnose"}</button>
                      <button className="c2-btn ghost" type="button" disabled={busy} onClick={() => act(doDismiss, s.run_id, false)}><I n="x" size={15} />Dismiss</button></span>)
                    : <Link className="c2-btn ghost" to={`/runs/${s.run_id}`}><I n="doc" size={15} />View<I n="chevron" size={14} /></Link>}</td>
                </tr>);
            })}
            {!shown.length && <tr><td colSpan="8" className="c2-muted">No alerts with this severity.</td></tr>}
          </tbody>
        </table>
      </div>
      {(diag.error || dis.error) && <div className="error">{String(diag.error || dis.error)}</div>}
      <div className="c2-note"><I n="info" size={20} /><span>Diagnose runs C2 on the same batch, without its labels, and saves the result with that run. Dismiss stops counting it without a diagnosis. New test batches come from the <Link to="/lab"><b>Drift Lab</b></Link>: <b>Run C1 → C2</b>, or <b>Run C1 only</b> and then Diagnose here.</span></div>
    </section>
  );
}

export function Kpi({ tone, tile, icon, k, v, s }) {
  return (<div className={`c2-kpi ${tone}`}><span className={`c2-tile ${tile}`} style={{ width: 52, height: 52 }}><I n={icon} size={26} /></span>
    <div style={{ display: "flex", flexDirection: "column", minWidth: 0 }}><span className="k">{k}</span><span className="n">{v}</span>{s && <span className="s">{s}</span>}</div></div>);
}

// ── The answer for one run ─────────────────────────────────────────────────────────────────────────
function Answer({ rec, latest }) {
  const st = rec.state, s = STATE[st], c3 = rec.to_c3, a = answer(rec), harm = harmOf(rec), dec = decOf(rec), c1 = c1PredOf(rec);
  const agree = runsAgreeing(rec), stab = c3.diagnosis_stability, group = rec.plain?.group || c3.affected_slice_rule;
  const root = c3.root_cause_feature, quiet = st === "clean" || st === "harmless";
  const range = (c3.affected_value_ranges || {})[root];
  const n = rec.n_rows || 0, rows = a.rows, refShare = rec.localization?.ref_share;
  const verdictText = {
    clean: "Nothing needs repair in this batch.",
    harmless: "It changed, but it barely changes the model's decisions.",
    suspected: "Seen in one run so far. The next run will confirm it or clear it.",
    confirmed: "Above the alarm line in at least 2 of the last 3 runs. Repair recommended.",
  }[st];
  const t = rec.truth;
  return (
    <>
      <div className="c2-runline">
        <span style={{ display: "inline-flex", gap: 8, alignItems: "center" }}><I n="calendar" size={17} sw={1.8} />{time(rec.timestamp)} · {n.toLocaleString()} rows · {rec.latency_s?.toFixed(2)} s</span>
      </div>
      <div className={`c2-verdict ${quiet ? "ok" : ""}`} role="status">
        <span className={`c2-tile xs ${quiet ? "t-green" : "t-solid-orange"}`} style={{ width: 40, height: 40 }}><I n={quiet ? "check" : "warn"} size={22} /></span>
        <b>{st === "clean" ? s.head : `${s.head}: ${root}`}</b><span className="sep" />
        <span className="m">{verdictText}</span>
        <span className="pillw">{s.label}</span>
      </div>
      <div className="c2-sentence"><span className="c2-tile t-green" style={{ width: 52, height: 52, borderRadius: "50%" }}><I n="sparkle" size={26} sw={1.4} /></span>
        <p>{a.sentence.map((p, i) => (typeof p === "string" ? p : <b key={i}>{p.b}</b>))}</p></div>

      <section className="c2-grid3">
        <div className="c2-ans blue">
          <div className="c2-head"><span className="c2-tile t-blue"><I n="bars" size={28} /></span>
            <div className="t"><span className="q">What caused it?</span>
              {st === "clean" ? <><span className="big sm" style={{ color: "var(--c2-green-deep)" }}>Nothing harmful</span><span className="s">no feature above the alarm line in 2 of the last 3 runs</span></>
                : <><span className="big" style={{ color: "var(--c2-green-deep)", fontSize: root.length > 11 ? "1.75rem" : undefined }}>{root.replace(/_/g, "_\u200b")}</span><span className="s">{c3.root_cause_category} drift · top cause in {agree} of the last 3 runs</span></>}</div></div>
          {!quiet && c3.affected_slice_rule && (<>
            <span style={{ color: "var(--c2-ink-2)" }}>Range that caused it</span>
            {range ? <span className="c2-rangechip">{root} {range[0] != null && range[1] != null ? `between ${+range[0].toFixed(4)} and ${+range[1].toFixed(4)}` : range[0] != null ? `> ${+range[0].toFixed(4)}` : `≤ ${+range[1].toFixed(4)}`}</span>
              : <span className="c2-rangechip">{c3.affected_slice_rule}</span>}
            {range && (range[0] == null) !== (range[1] == null) && <RangeBar lo={range[0]} hi={range[1]} refShare={refShare} />}
            <div className="c2-note" style={{ background: "var(--c2-blue-bg)", borderColor: "var(--c2-blue-line)" }}><I n="info" size={18} />
              <span>{rec.plain?.group ? `${rec.plain.group}: ` : ""}{pct(c3.slice_support)} of this batch vs {refShare != null ? pct(refShare, 1) : "–"} in training.{rec.plain && !rec.plain.decoded ? " Values are scaled: add preprocessor.pkl to show real units." : ""}</span></div>
          </>)}
        </div>
        <div className="c2-ans orange">
          <div className="c2-head"><span className="c2-tile t-orange"><I n="trend" size={28} /></span>
            <div className="t"><span className="q">How much does it hurt?</span>
              <span className="big" style={{ color: "var(--c2-orange-strong)" }}>{sg(dec)}%</span>
              <span className="s">of decisions flip back when {root} is put back to normal.</span></div></div>
          <div className="c2-stat3">
            <div><span style={{ color: "var(--c2-orange-strong)" }}><I n="target" size={24} /></span>Impact score<b>{c3.drift_impact_score.toFixed(2)}</b></div>
            <div><span style={{ color: "var(--c2-red)" }}><I n="warn" size={24} /></span>Est. accuracy loss<b>{harm != null ? `${sg(harm)} pp` : "–"}</b></div>
            <div><span style={{ color: "var(--c2-green)" }}><I n="chartUp" size={24} /></span>C1 predicted<b>{c1 != null ? `${sg(c1)} pp` : "–"}</b></div>
          </div>
        </div>
        <div className="c2-ans green">
          <div className="c2-head"><span className="c2-tile t-green"><I n="people" size={28} /></span>
            <div className="t"><span className="q">Who is affected?</span>
              {quiet ? <><span className="big sm" style={{ color: "var(--c2-green-deep)" }}>No group</span><span className="s">nothing sent to C3 for repair</span></>
                : <><span className="big" style={{ color: "var(--c2-green-deep)" }}>{pct(c3.slice_support)}</span>
                  <span className="s">of this batch ({rows.toLocaleString()} rows){refShare != null ? ` · ${pct(refShare, 1)} in training data` : ""}</span></>}</div></div>
          {!quiet && (<div className="c2-donutrow">
            <Donut share={c3.slice_support} />
            <div className="c2-legendbox">
              <span className="i"><i className="c2-dot" style={{ background: "var(--c2-green)" }} /><span><b>{rows.toLocaleString()} rows</b><span className="c2-sub2">affected in this batch</span></span></span>
              <span className="i"><i className="c2-dot" style={{ background: "var(--c2-dot)" }} /><span><b>{(n - rows).toLocaleString()} rows</b><span className="c2-sub2">not affected</span></span></span>
            </div></div>)}
        </div>
      </section>

      {t && rec.injection?.drift_type !== undefined && (
        <section className="c2-key" aria-label="Answer key, hidden from C2">
          <span className="c2-tile sm t-purple" style={{ width: 48, height: 48 }}><I n="sparkle" size={26} sw={1.4} /></span>
          <div style={{ flex: "1 1 700px", display: "flex", flexDirection: "column", gap: 10 }}>
            <span className="title">ANSWER KEY · HIDDEN FROM C2</span>
            <div className="items">
              <div><span className="ic"><I n="target" size={22} /></span><span><span className="c2-sub2">True accuracy drop</span><b>{sg(t.true_harm_pp)} pp</b></span></div>
              <div><span className="ic"><I n="chartUp" size={22} /></span><span><span className="c2-sub2">C2 decisions changed</span><b>{sg(dec)}%</b></span></div>
              <div><span className="ic"><I n="db" size={22} /></span><span><span className="c2-sub2">Injected features</span><b>{(t.true_causes || []).join(", ") || "nothing"}</b></span></div>
              <div><span className="ic"><I n="link" size={22} /></span><span><span className="c2-sub2">C2 root cause{t.main_cause ? ` (true main cause: ${t.main_cause})` : ""}</span>
                <b>{root} {(t.main_cause ? t.main_cause === root : (t.true_causes || []).includes(root)) ? "✓" : (t.true_causes || []).length ? "✗" : ""}</b></span></div>
            </div>
          </div>
        </section>)}

      <div className="c2-badges">
        <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
          <span className={`badge ${rec.c1?.status === "ok" ? "ok" : ""}`} title="C1's JSON is checked against the C1 contract before C2 uses it">
            C1 input: {rec.c1?.status || "missing"}{rec.c1?.harm_severity ? ` · ${rec.c1.harm_severity}` : ""}{rec.c1_source === "fake" ? " · fake C1" : rec.c1_source === "real" ? " · real C1" : ""}</span>
          <span className="badge">stability {stab.toFixed(2)}{stab < 0.5 ? " · C4 escalates" : ""}</span>
          {!quiet && <span className="badge" style={{ whiteSpace: "normal" }} title={c3.affected_slice_rule}>group: {group}</span>}
          {rec.monitor_batch && <span className="badge">monitoring batch #{rec.monitor_batch}</span>}
        </div>
        <Link className="c2-btn primary" to={`/c2/report/${rec.run}`}>Open report</Link>
      </div>
      {rec.warnings?.map((w) => <div key={w} className="warnline">{w}</div>)}
    </>
  );
}

// ── Temporal SHAP across stored runs ───────────────────────────────────────────────────────────────
function Temporal({ matrix, rec, onSelect }) {
  const [scale, setScale] = useSession("c2:scale", "log");
  const [view, setView] = useSession("c2:heatview", "heatmap");
  const [hidden, setHidden] = useState({});
  const win = temporalWindow(matrix);
  const feats = topFeatures(win, 5);
  const order = Object.entries(rec.temporal_shap.ewma).sort((a, b) => b[1] - a[1]).map((x) => x[0]).filter((f) => f in win.z);
  return (
    <section className="c2-card">
      <div className="c2-headrow">
        <div className="c2-head" style={{ flex: "1 1 520px" }}><span className="c2-tile t-purple"><I n="chartUp" size={28} /></span>
          <div className="t"><h3>Temporal SHAP across stored runs</h3>
            <p>z = how many times larger than normal noise a feature's explanation change is. An alarm needs the same feature above the alarm line in 2 of the last 3 runs, so a single noisy run never raises one.</p></div></div>
        <div style={{ display: "flex", flexDirection: "column", gap: 10, alignItems: "flex-end" }}>
          <span className="small c2-muted">Novelty 1 · last {win.runs.length} of {matrix.runs.length} runs · click a run to open it</span>
          <label><span className="c2-sr">Chart scale</span>
            <select value={scale} onChange={(e) => setScale(e.target.value)} style={{ font: "inherit", padding: "0 12px", minHeight: 42, border: "1px solid var(--c2-line)", borderRadius: 8, background: "var(--surface)" }}>
              <option value="log">Log scale</option><option value="linear">Linear scale</option></select></label>
        </div>
      </div>
      <div className="c2-tpanel">
        <div className="chart"><TemporalChart win={win} feats={feats} selected={rec.run} onSelect={onSelect} scale={scale} hidden={hidden} /></div>
        <div className="legend" role="group" aria-label="Lines shown">{feats.map((f, j) => (
          <button key={f} type="button" aria-pressed={!hidden[f]} onClick={() => setHidden((h) => ({ ...h, [f]: !h[f] }))} style={{ opacity: hidden[f] ? 0.4 : 1 }}>
            <i className="c2-dot" style={{ background: LINE_COLS[j], margin: 0 }} />{f}</button>))}
          <span className="c2-sub2">Click a name to hide or show its line.</span></div>
      </div>
      <div className="c2-card" style={{ padding: 16, boxShadow: "none", borderColor: "var(--c2-line-soft)" }}>
        <div className="c2-headrow" style={{ alignItems: "center" }}><h4 style={{ margin: 0, fontSize: "1.05rem" }}>Feature values across runs (z)</h4>
          <div className="c2-seg blue" role="group" aria-label="View">
            <button type="button" aria-pressed={view === "heatmap"} onClick={() => setView("heatmap")}><I n="grid" size={16} />Heatmap</button>
            <button type="button" aria-pressed={view === "table"} onClick={() => setView("table")}><I n="list" size={16} />Table</button></div></div>
        {view === "heatmap" ? <ZHeatmap win={win} order={order} selected={rec.run} onSelect={onSelect} root={rec.to_c3.root_cause_feature} /> : (
          <div className="c2-tbl-wrap"><table className="c2-tbl"><thead><tr><th scope="col">Feature</th>{win.runs.map((r) => <th key={r} className="num" scope="col">#{r}</th>)}</tr></thead>
            <tbody>{order.map((f) => <tr key={f}><td className="c2-mono">{f}</td>{win.z[f].map((v, i) => <td key={i} className="num c2-mono" style={v > win.thresh ? { fontWeight: 700 } : null}>{sg(v)}</td>)}</tr>)}</tbody></table></div>)}
      </div>
      <HeatKey thresh={win.thresh} />
    </section>
  );
}

// ── Who is affected, and what simpler tools would say ──────────────────────────────────────────────
function Who({ rec }) {
  const st = rec.state, c3 = rec.to_c3, loc = rec.localization;
  const head = (extra) => (
    <div className="c2-head"><span className="c2-tile t-blue"><I n="people" size={28} sw={1.9} /></span>
      <div className="t"><div className="c2-headrow" style={{ alignItems: "center" }}><h3>Who is affected</h3>{extra}</div>
        {st !== "clean" && st !== "harmless" && <span>Records where <b style={{ color: "var(--accent)" }}>{rec.plain?.group || c3.affected_slice_rule}</b>.</span>}</div></div>);
  if (st === "clean" || st === "harmless") {
    return <div className="c2-card">{head(null)}<p style={{ margin: 0 }}>No group of records needs repair in this run.</p></div>;
  }
  const n = rec.n_rows || 0, rows = Math.round(c3.slice_support * n), ref = loc.ref_share;
  const times = ref ? c3.slice_support / ref : null;
  const max = Math.max(c3.slice_support, ref || 0) * 1.25 || 1;
  return (
    <div className="c2-card">
      {head(<span className="c2-chip blue"><I n="person" size={15} />{pct(c3.slice_support, 1)} of batch affected</span>)}
      <div className="c2-sharegrid">
        <span>Training data</span><span style={{ display: "flex", gap: 10, alignItems: "center" }}><span style={{ display: "inline-block", height: 24, width: `${((ref || 0) / max) * 100}%`, background: "var(--c2-dot)", borderRadius: 4 }} /><b style={{ fontWeight: 600 }}>{ref != null ? pct(ref, 1) : "–"}</b></span><span className="c2-muted" style={{ textAlign: "right" }}>of training</span>
        <span>This batch</span><span style={{ display: "flex", gap: 10, alignItems: "center" }}><span style={{ display: "inline-block", height: 24, width: `${(c3.slice_support / max) * 100}%`, background: "var(--c2-orange)", borderRadius: 4 }} /><b style={{ fontWeight: 600 }}>{pct(c3.slice_support, 1)}</b></span><span className="c2-muted" style={{ textAlign: "right" }}>{rows.toLocaleString()} rows</span>
      </div>
      <div style={{ display: "flex", gap: 12, alignItems: "center", background: "var(--c2-panel)", border: "1px solid var(--c2-line-soft)", borderRadius: 10, padding: "10px 12px", flexWrap: "wrap" }}>
        <span className="c2-tile xs t-red"><I n="code" size={17} /></span><span className="c2-code">{c3.affected_slice_rule}</span><span className="c2-sub2">rule sent to C3</span></div>
      <div className="c2-grid2" style={{ gap: 10 }}>
        <div className="c2-mini k-blue"><span style={{ color: "var(--c2-blue)" }}><I n="bars" size={26} /></span><div><b style={{ color: "var(--c2-blue-fg)" }}>{times ? `${times.toFixed(1)}×` : "–"}</b><span>more common than in training</span></div></div>
        <div className="c2-mini k-amber"><span style={{ color: "var(--c2-orange-strong)" }}><I n="doc" size={26} /></span><div><b style={{ color: "var(--c2-orange-fg)" }}>≈ {Math.round(loc.rows_at_risk || 0).toLocaleString()}</b><span>decisions change due to this drift</span></div></div>
        <div className="c2-mini k-green"><span style={{ color: "var(--c2-green)" }}><I n="pie" size={26} /></span><div><b style={{ color: "var(--c2-green-fg)" }}>{pct(loc.est_recall)}</b><span>of the movement, estimated</span></div></div>
        <div className="c2-mini k-amber"><span style={{ color: "var(--c2-amber-fg)" }}><I n="shield" size={26} /></span><div><b style={{ color: "var(--c2-orange-fg)", fontSize: "1rem" }}>Private</b><span>groups under 20 rows are never reported</span></div></div>
      </div>
      <p style={{ margin: 0 }} className="c2-muted small">Found without labels.{rec.plain && !rec.plain.decoded ? " Values are scaled: add preprocessor.pkl to decode them." : ""}</p>
    </div>
  );
}

function Compare({ rec }) {
  const b = baselines(rec), clean = rec.state === "clean", dec = decOf(rec);
  return (
    <div className="c2-card" style={{ gap: 12 }}>
      <div className="c2-head" style={{ alignItems: "center" }}><span className="c2-tile t-amber"><I n="bulb" size={28} sw={1.9} /></span><div className="t"><h3>What simpler tools would say about this run</h3></div></div>
      <div className="c2-tool" style={{ background: "var(--c2-blue-bg)", borderColor: "var(--c2-blue-line)" }}>
        <span className="c2-tile sm t-blue" style={{ width: 40, height: 40 }}><I n="bars" size={22} /></span>
        <div className="t"><div className="r"><b style={{ color: "var(--c2-blue-fg)" }}>A drift monitor ranks</b><span className="c2-chip orange" style={{ fontSize: ".72rem" }}>Top 3 features</span></div>
          <span className="m">{b.drift.join(", ")}</span><span className="c2-sub2">Says what changed, not what hurts.</span></div></div>
      <div className="c2-tool" style={{ background: "var(--c2-purple-bg)", borderColor: "var(--c2-purple-line)" }}>
        <span className="c2-tile sm t-purple" style={{ width: 40, height: 40 }}><I n="doc" size={22} /></span>
        <div className="t"><div className="r"><b style={{ color: "var(--c2-purple-fg)" }}>One SHAP snapshot ranks</b><span className="c2-chip red" style={{ fontSize: ".72rem" }}>Top 3 features</span></div>
          <span className="m">{b.snapshot.join(", ")}</span><span className="c2-sub2">{clean ? "Names a cause even on clean data." : "No noise check, no run history."}</span></div></div>
      <div className="c2-tool" style={{ background: "var(--c2-green-bg)", borderColor: "var(--c2-green-line)" }}>
        <span className="c2-tile sm t-green" style={{ width: 40, height: 40 }}><I n="search" size={22} /></span>
        <div className="t"><div className="r"><b style={{ color: "var(--accent)" }}>C2 says</b>{!clean && <span className="c2-chip solid-green" style={{ fontSize: ".72rem" }}><I n="check" size={13} sw={3} />{STATE[rec.state].label}</span>}</div>
          <span className="m" style={{ color: "var(--c2-green-deep)" }}>{clean ? "no alarm" : `${rec.to_c3.root_cause_feature}${dec != null ? ` · ${sg(dec)}% decisions` : ""}`}</span>
          <span className="c2-sub2">{clean ? "Normal noise." : "Cause, damage and group, confirmed over runs."}</span></div></div>
    </div>
  );
}

// ── Run history ────────────────────────────────────────────────────────────────────────────────────
const EDGE = { confirmed: "var(--c2-amber)", suspected: "var(--c2-orange)", clean: "var(--c2-green)", harmless: "var(--c2-green)" };

function History({ runs, current, latest, onSelect, onReset, resetting }) {
  return (
    <section className="c2-card">
      <div className="c2-headrow" style={{ alignItems: "center" }}>
        <div className="c2-head" style={{ alignItems: "center" }}><span className="c2-tile sm t-blue" style={{ width: 48, height: 48 }}><I n="clock" size={26} /></span>
          <div className="t"><h3>Run history</h3><span className="d">Every run is stored with its full temporal SHAP analysis · click a row to view it</span></div></div>
        <div style={{ display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
          <span className="c2-sub2" style={{ maxWidth: 280, textAlign: "right" }}>Clears these runs and the monitoring log, and relearns normal noise from clean data. Use before a demo or after a repair.</span>
          <button className="c2-btn danger" type="button" disabled={resetting} onClick={onReset}><I n="reset" size={17} />{resetting ? "Resetting…" : "Reset monitoring"}</button>
        </div>
      </div>
      <div className="c2-tbl-wrap">
        <table className="c2-tbl" style={{ minWidth: 860 }}>
          <thead><tr><th scope="col">Run</th><th scope="col">Time</th><th scope="col">Batch</th><th scope="col">Result</th><th scope="col">Root cause</th>
            <th scope="col" className="num">Decisions changed</th><th scope="col" className="num">True drop</th><th scope="col" className="num">Stability</th><th scope="col">Report</th></tr></thead>
          <tbody>
            {runs.map((r) => (
              <tr key={r.run} className={`click ${r.run === current ? "sel" : ""}`} aria-selected={r.run === current} tabIndex="0" onClick={() => onSelect(r.run)}
                onKeyDown={(e) => { if (e.key === "Enter") onSelect(r.run); }}>
                <td className="c2-edge" style={{ "--edge": r.run === current ? "var(--accent)" : EDGE[r.state] }}>
                  #{r.run}{r.run === latest && <span className="c2-chip green" style={{ marginLeft: 8, padding: "1px 8px", fontSize: ".72rem" }}>Latest</span>}</td>
                <td style={{ maxWidth: 110 }}>{time(r.timestamp)}</td>
                <td style={{ maxWidth: 220, overflowWrap: "anywhere" }} title={r.source}>{shortSource(r.source).replace("Drift Lab · ", "")}</td>
                <td style={{ whiteSpace: "nowrap" }}><StatePill state={r.state} /></td>
                <td className="c2-mono">{r.root_cause_feature || "none"}</td>
                <td className="num">{r.decision_change_pct == null ? "–" : `${sg(r.decision_change_pct)}%`}</td>
                <td className="num" style={{ whiteSpace: "nowrap" }}>{r.truth_pp == null ? "–" : `${sg(r.truth_pp)} pp`}</td>
                <td className="num">{r.diagnosis_stability.toFixed(2)}</td>
                <td><Link className="c2-link" to={`/c2/report/${r.run}`} onClick={(e) => e.stopPropagation()}>View report <I n="arrow" size={15} /></Link></td>
              </tr>))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

// ── Why trust it: recorded benchmark ────────────────────────────────────────────────────────────────
const PROOF = {
  n1: ["var(--c2-green-bg)", "var(--c2-green-line)", "t-green", "shield", "var(--c2-green-deep)"],
  n2: ["var(--c2-blue-bg)", "var(--c2-blue-line)", "t-blue", "target", "var(--c2-blue-fg)"],
  n3: ["var(--c2-purple-bg)", "var(--c2-purple-line)", "t-purple", "trophy", "var(--c2-purple-fg)"],
};
const PCHART = ["var(--c2-amber)", "var(--c2-red)"];

function Proof({ rec }) {
  const [p, setP] = useSession("c2:proof", "n1");
  const cur = rec[p];
  const split = (label) => { const m = label.match(/^(.*?)\s*\(([^)]*)\)\s*$/); return m ? [m[1], `(${m[2]})`] : [label, ""]; };
  return (
    <section className="c2x" style={{ gap: 14 }}>
      <div className="c2-headrow"><span className="c2-eyebrow">Why trust it · recorded UCI Adult benchmark, no labels used, graded with the answer key</span><span className="small c2-muted">click a card</span></div>
      <div className="c2-grid3" style={{ gap: 14 }}>
        {["n1", "n2", "n3"].map((k) => {
          const [bg, line, tile, icon, fg] = PROOF[k], [main, sub] = split(rec[k].headline_label);
          return (
            <button key={k} type="button" className="c2-proofcard" aria-pressed={p === k} onClick={() => setP(k)}
              style={{ background: bg, borderColor: p === k ? fg : line }}>
              <span style={{ display: "flex", gap: 14, alignItems: "flex-start" }}><span className={`c2-tile ${tile}`} style={{ width: 52, height: 52 }}><I n={icon} size={28} /></span>
                <span style={{ display: "flex", flexDirection: "column", gap: 4 }}><span className="v" style={{ color: fg }}>{rec[k].headline}</span><span>{main}</span>{sub && <span className="c2-muted small">{sub}</span>}</span></span>
              <span className="foot"><span>{rec[k].title}</span></span>
            </button>);
        })}
      </div>
      <div className="c2-card">
        <div className="c2-head"><span className="c2-tile t-purple" style={{ width: 52, height: 52 }}><I n="cube" size={26} /></span>
          <div className="t"><h3>{cur.title}</h3><p>{cur.note}</p></div></div>
        <div className={cur.charts.length > 1 ? "c2-grid2" : ""}>
          {cur.charts.map((c, i) => (
            <div key={c.label} className="c2-pchart" style={{ "--h": PCHART[i % 2] }}>
              <span className="h">{c.label}</span><ProofBars chart={c} /></div>))}
        </div>
      </div>
    </section>
  );
}

export { relianceOf };
