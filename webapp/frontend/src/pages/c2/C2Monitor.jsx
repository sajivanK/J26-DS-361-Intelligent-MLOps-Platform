// C2 Monitoring tab (Tharmika N, IT22247940). Temporal SHAP runs on EVERY batch C1 checks, not only
// on alerts: one SHAP pass per batch, compared with normal noise. A feature above the alarm line in 2 of
// the last 3 batches raises a C2 alarm, also when C1 reported no harm (interaction drift that each
// column alone hides). The full diagnosis runs only on Diagnose. Data: GET /api/c2/monitor.
import { useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../../api.js";
import { useAction } from "../../hooks.js";
import { useSession } from "../../session.jsx";
import { Empty, ErrorBox } from "../../ui/parts.jsx";
import { Kpi } from "./C2Page.jsx";
import { LINE_COLS, Sparkline } from "./C2Charts.jsx";
import { I } from "./c2icons.jsx";
import { clock, sg, shortSource } from "./c2format.js";

const STATE = { alarm: ["Alarm", "alarm", "warn"], watch: ["Watching", "watch", "eye"], normal: ["Normal", "normal", "checkCircle"], unavailable: ["No batch", "unavailable", "info"] };
const EDGE = { alarm: "var(--c2-red)", watch: "var(--c2-amber)", normal: "var(--c2-green)", unavailable: "var(--c2-dot)" };
const C1V = (c1) => {
  if (!c1) return ["none", "No C1 result"];
  const sev = c1.harm_severity;
  if (sev && sev !== "none") return ["harm", `Harmful drift · ${sev}`];
  if (c1.drift_detected) return ["drift", "Drift, no harm"];
  return ["none", "No drift"];
};
const C1COL = { harm: "var(--c2-red)", drift: "var(--c2-amber)", none: "var(--c2-dot)" };
const C1CHIP = { harm: "red", drift: "orange", none: "grey" };
const heatBin = (v) => (v == null ? 0 : v < 1 ? 0 : v < 2 ? 1 : v < 4 ? 2 : v < 10 ? 3 : v < 50 ? 4 : 5);

export default function C2Monitor({ mon, onDiagnosed, onOpenRun, onReset, resetting }) {
  const [doDiag, diag] = useAction(api.c2MonitorDiagnose);
  const [busy, setBusy] = useState(null);
  const [hideAlarm, setHideAlarm] = useSession("c2:hidealarm", null);
  const d = mon.data;
  async function diagnose(batch) {
    setBusy(batch);
    const out = await doDiag(batch);
    setBusy(null);
    if (out?.run) { mon.reload(); onDiagnosed?.(); onOpenRun?.(out.run.run); }
  }
  if (mon.loading && !d) return <p className="muted">Checking the batches C1 has seen…</p>;
  if (mon.error && !d) return <ErrorBox error={mon.error} />;
  if (!d || !d.kpis.total) {
    return (<>
      <Live d={d} />
      <RunPanel d={d} reload={mon.reload} diagnose={diagnose} busy={busy} diagBusy={diag.busy} onOpenRun={onOpenRun} />
      {diag.error && <div className="error">{String(diag.error)}</div>}
      <Empty>No batch monitored yet. Press <b>Run temporal SHAP check</b> to monitor every batch C1 has already checked (Drift Lab, C1 page, pipeline runs), or check the latest rows as one batch. New batches from the <Link to="/lab">Drift Lab</Link> or the C1 page are added here automatically.</Empty>
    </>);
  }
  const alarm = d.c2_alarm && hideAlarm !== d.c2_alarm.batch ? d.c2_alarm : null;
  const k = d.kpis;
  return (
    <>
      <Live d={d} />
      <RunPanel d={d} reload={mon.reload} diagnose={diagnose} busy={busy} diagBusy={diag.busy} onOpenRun={onOpenRun} />
      {alarm && (
        <section className="c2-alarmbox" role="alert">
          <span className="c2-tile t-red" style={{ width: 52, height: 52 }}><I n="warn" size={28} /></span>
          <div className="t">
            <b>C2 alarm: the model's reasoning changed, and C1 did not flag it</b>
            <span>Since batch #{alarm.since} the model relies on {alarm.features.map((f, i) => (
              <span key={f}>{i ? (i === alarm.features.length - 1 ? " and " : ", ") : ""}<b>{f}</b> (z {sg(alarm.z[f])})</span>))} far more than normal,
              in {alarm.count} batch{alarm.count === 1 ? "" : "es"}. C1 reported no harmful drift for {alarm.count === 1 ? "it" : "these batches"}: each column alone still looks normal,
              which {alarm.features.length > 1 ? <>points to <b>interaction drift</b> (the columns changed together)</> : "can mean a change C1's per-column tests do not see"}.</span>
          </div>
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
            <button className="c2-btn primary" type="button" disabled={diag.busy} onClick={() => diagnose(alarm.batch)}><I n="search" size={17} />{busy === alarm.batch ? "Diagnosing…" : "Run full diagnosis"}</button>
            <button className="c2-btn ghost" type="button" onClick={() => setHideAlarm(alarm.batch)}>Dismiss</button>
          </div>
        </section>)}
      <div className="c2-grid5">
        <Kpi tone="k-blue" tile="t-blue" icon="pulse" k="Batches monitored" v={k.total} s="since the last reset" />
        <Kpi tone="k-green" tile="t-green" icon="checkCircle" k="Normal" v={k.normal} s="no feature above the line" />
        <Kpi tone="k-amber" tile="t-amber" icon="eye" k="Watching" v={k.watch} s="above the line once, not confirmed" />
        <Kpi tone="k-red" tile="t-red" icon="warn" k="Alarm" v={k.alarm} s="persistent: 2 of the last 3" />
        <Kpi tone="k-purple" tile="t-purple" icon="search" k="Found by C2 only" v={k.c1_missed} s="alarm batches C1 called harmless" />
      </div>
      <Timeline m={d.matrix} />
      <section className="c2-grid2" style={{ gridTemplateColumns: "minmax(0, .9fr) minmax(0, 1.1fr)" }}>
        <Watchlist list={d.watchlist} line={d.matrix.z_thresh} />
        <Heat m={d.matrix} order={d.watchlist.map((w) => w.feature)} />
      </section>
      <BatchLog rows={d.batches} diagnose={diagnose} busy={busy} diagBusy={diag.busy} onOpenRun={onOpenRun} onReset={onReset} resetting={resetting} />
      {diag.error && <div className="error">{String(diag.error)}</div>}
      {k.unavailable > 0 && <div className="warnline">{k.unavailable} saved run{k.unavailable === 1 ? "" : "s"} could not be rebuilt as a batch (no saved file or recipe), so {k.unavailable === 1 ? "it is" : "they are"} listed but not monitored.</div>}
      <div className="c2-note"><I n="info" size={20} /><span>Monitoring is light (one SHAP pass per batch). The full diagnosis (impact score, value range, group, output to C3) runs only when you press Diagnose, from here or from the Diagnosis tab.</span></div>
    </>
  );
}

// ── Run the monitor by hand: check saved C1 runs now, check a CSV, run the full diagnosis ───────────
function RunPanel({ d, reload, diagnose, busy, diagBusy, onOpenRun }) {
  const [doCheck, check] = useAction(api.c2MonitorCheck);
  const [doLatest, fileSt] = useAction(api.c2MonitorCheckLatest);
  const [nRows, setNRows] = useSession("c2:latestrows", 1000);
  const [msg, setMsg] = useState(null);
  const [pick, setPick] = useState("");
  const rows = (d?.batches || []).filter((b) => b.state !== "unavailable");
  const chosen = rows.find((b) => String(b.batch) === pick) || rows.find((b) => !b.c2_run) || rows[0];
  async function runCheck() {
    setMsg(null);
    const out = await doCheck();
    if (!out) return;
    const c = out.last_check || {};
    const parts = [`${c.new ?? 0} new batch${c.new === 1 ? "" : "es"} monitored`];
    if (c.waiting) parts.push(`${c.waiting} Drift Lab run${c.waiting === 1 ? "" : "s"} wait for the C1 engine (start the backend with C1 loaded, then check again)`);
    if (c.unavailable) parts.push(`${c.unavailable} without a kept batch`);
    setMsg({ ok: true, text: `Temporal SHAP check done: ${parts.join(" · ")}. ${out.kpis.total} batches in the log.` });
    reload();
  }
  async function runLatest() {
    setMsg(null);
    const out = await doLatest(Number(nRows));
    if (!out) return;
    const b = out.checked;
    const short = b.n_rows < Number(nRows) ? ` Only ${b.n_rows.toLocaleString()} rows were available.` : "";
    setMsg({ ok: true, text: out.already ? `These rows have not changed since batch #${b.batch}, so they were not counted again.`
      : `Batch #${b.batch} (newest ${b.n_rows.toLocaleString()} rows from ${(b.from_runs || []).length} C1 run${(b.from_runs || []).length === 1 ? "" : "s"}): ${STATE[b.state][0]}. Top feature ${b.feature}, z ${sg(b.feature_z)}.${short}${b.c1_note ? ` ${b.c1_note}.` : ""}` });
    setPick(String(b.batch));
    reload();
  }
  const err = check.error || fileSt.error;
  return (
    <section className="c2-card" aria-label="Run the monitor">
      <div className="c2-head"><span className="c2-tile t-green"><I n="pulse" size={28} /></span>
        <div className="t"><h3>Run the monitor</h3>
          <p>New C1 runs are monitored automatically. Run the check by hand to pick up every saved C1 run (Drift Lab, C1 page, pipeline) not monitored yet, or check the newest rows C1 has seen as one batch. Any monitored batch can then get the full diagnosis.</p></div></div>
      <div className="c2-grid3" style={{ gap: 12 }}>
        <div className="c2-tool" style={{ background: "var(--c2-green-bg)", borderColor: "var(--c2-green-line)", flexDirection: "column", alignItems: "stretch" }}>
          <b style={{ color: "var(--c2-green-fg)" }}>1 · Temporal SHAP check on C1's runs</b>
          <span className="c2-sub2">One SHAP pass per batch, compared with normal noise. Old runs are checked oldest first, so the 2-of-3 rule follows their real order.</span>
          <button className="c2-btn primary" type="button" disabled={check.busy || fileSt.busy} onClick={runCheck} style={{ alignSelf: "flex-start" }}>
            <I n="pulse" size={17} />{check.busy ? "Checking…" : "Run temporal SHAP check"}</button>
        </div>
        <div className="c2-tool" style={{ background: "var(--c2-blue-bg)", borderColor: "var(--c2-blue-line)", flexDirection: "column", alignItems: "stretch" }}>
          <b style={{ color: "var(--c2-blue-fg)" }}>2 · Check the latest rows</b>
          <span className="c2-sub2">The newest rows C1 has checked, pooled from its most recent runs, as one batch. C1 gives its verdict when its engine is loaded.</span>
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
            <label><span className="c2-sr">Number of latest rows</span>
              <select value={nRows} onChange={(e) => setNRows(Number(e.target.value))}
                style={{ font: "inherit", minHeight: 42, padding: "0 10px", border: "1px solid var(--c2-line)", borderRadius: 8, background: "var(--surface)" }}>
                {[500, 1000, 2000, 5000].map((n) => <option key={n} value={n}>Last {n.toLocaleString()} rows</option>)}
              </select></label>
            <button className="c2-btn outline" type="button" disabled={fileSt.busy || check.busy} onClick={runLatest}>
              <I n="pulse" size={17} />{fileSt.busy ? "Checking…" : "Check rows"}</button>
          </div>
        </div>
        <div className="c2-tool" style={{ background: "var(--c2-purple-bg)", borderColor: "var(--c2-purple-line)", flexDirection: "column", alignItems: "stretch" }}>
          <b style={{ color: "var(--c2-purple-fg)" }}>3 · Run full diagnosis</b>
          <span className="c2-sub2">Root cause, value range, damage, affected group and the output to C3 and C4, for the batch you pick.</span>
          {rows.length ? (
            <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
              <label><span className="c2-sr">Batch to diagnose</span>
                <select value={chosen ? String(chosen.batch) : ""} onChange={(e) => setPick(e.target.value)}
                  style={{ font: "inherit", minHeight: 42, padding: "0 10px", border: "1px solid var(--c2-line)", borderRadius: 8, background: "var(--surface)", maxWidth: 230 }}>
                  {rows.map((b) => <option key={b.batch} value={String(b.batch)}>#{b.batch} · {STATE[b.state][0]} · {b.feature}{b.c2_run ? " · diagnosed" : ""}</option>)}
                </select></label>
              {chosen?.c2_run
                ? <button className="c2-btn ghost" type="button" onClick={() => onOpenRun(chosen.c2_run)}><I n="eye" size={17} />Open diagnosis</button>
                : <button className="c2-btn primary" type="button" disabled={diagBusy || !chosen} onClick={() => diagnose(chosen.batch)}>
                  <I n="search" size={17} />{busy === chosen?.batch ? "Diagnosing…" : "Run full diagnosis"}</button>}
            </div>) : <span className="c2-sub2"><b>Run the temporal SHAP check first.</b></span>}
        </div>
      </div>
      {msg && <div className="c2-msg" role="status">{msg.text}</div>}
      {err && <div className="error">{String(err)}</div>}
    </section>
  );
}

function Live({ d }) {
  const last = d?.batches?.find((b) => b.state !== "unavailable");
  return (
    <div className="c2-live" role="status">
      <span className="on">Monitoring every batch C1 checks</span>
      <span>Last batch <b>{last ? `#${last.batch} · ${clock(last.timestamp)}` : "none yet"}</b></span>
      {(d?.matrix?.z_thresh ?? d?.z_line) != null && <span>Alarm line <b>z {sg(d.matrix?.z_thresh ?? d.z_line, 2)}</b></span>}
      <span>Rule <b>same feature above the line in {d?.persist_k ?? 2} of the last {d?.window ?? 3} batches</b></span>
      {d?.last_check && <span>Last check <b>{clock(d.last_check.at)}</b></span>}
      {d && d.c1_loaded === false && <span>C1 engine <b>not loaded</b></span>}
    </div>
  );
}

// ── Timeline: top features per batch (log z), C1's verdict as dots under the axis ─────────────────
function Timeline({ m }) {
  const [win, setWin] = useSession("c2:monwin", "all");
  const [hover, setHover] = useState(null);
  const today = new Date().toDateString();
  let idx = m.batches.map((_, i) => i);
  if (win === "10") idx = idx.slice(-10);
  if (win === "day") idx = idx.filter((i) => new Date(m.times[i]).toDateString() === today);
  if (!idx.length) idx = m.batches.map((_, i) => i).slice(-1);
  const feats = Object.keys(m.z);
  const peak = (f) => Math.max(...idx.map((i) => m.z[f][i] ?? 0));
  const top = [...feats].sort((a, b) => peak(b) - peak(a)).slice(0, 3);
  const rest = feats.filter((f) => !top.includes(f));
  const series = [...top.map((f, j) => ({ f, col: LINE_COLS[j], v: idx.map((i) => m.z[f][i]) })),
    ...(rest.length ? [{ f: "all other features (max)", col: "var(--c2-dot)", dash: "4 3", v: idx.map((i) => Math.max(...rest.map((f) => m.z[f][i] ?? 0))) }] : [])];
  const W = 960, x0 = 70, x1 = 930, y0 = 20, y1 = 270, n = idx.length;
  const zmax = Math.max(m.z_thresh * 2, ...series.flatMap((s) => s.v.map((x) => x ?? 0)));
  const hiL = Math.pow(10, Math.ceil(Math.log10(zmax)));
  const X = (k) => (n === 1 ? (x0 + x1) / 2 : x0 + 20 + (k * (x1 - x0 - 40)) / (n - 1));
  const Y = (v) => y1 - ((Math.log10(Math.max(v ?? 0.1, 0.1)) + 1) / (Math.log10(hiL) + 1)) * (y1 - y0);
  const half = n === 1 ? 30 : (x1 - x0 - 40) / (n - 1) / 2;
  const grid = [0.1, 1, 10, 100, 1000, 10000].filter((v) => v <= hiL);
  const windows = [["10", "Last 10"], ["all", `All ${m.batches.length}`], ["day", "Today"]];
  const h = hover != null ? idx[hover] : null;
  return (
    <section className="c2-card">
      <div className="c2-headrow">
        <div className="c2-head" style={{ flex: "1 1 520px" }}><span className="c2-tile t-purple"><I n="chartUp" size={28} /></span>
          <div className="t"><h3>Temporal SHAP, batch by batch</h3>
            <p>z = how many times larger than normal noise each feature's explanation change is (log scale). Every batch is a point, alarm or not.</p></div></div>
        <div className="c2-seg" role="group" aria-label="Time window">
          {windows.map(([k, l]) => <button key={k} type="button" aria-pressed={win === k} onClick={() => setWin(k)}>{l}</button>)}</div>
      </div>
      <div className="scroll-x"><div style={{ position: "relative", minWidth: 620 }}>
        <svg viewBox={`0 0 ${W} 320`} className="c2-svgchart" role="img" onMouseLeave={() => setHover(null)}
          aria-label={`Explanation change per batch for ${top.join(", ")}; alarm line z ${sg(m.z_thresh, 2)}`}>
          {idx.map((i, k) => m.states[i] === "alarm" && (
            <rect key={`b${k}`} x={X(k) - half} y={y0} width={half * 2} height={y1 - y0}
              fill={m.missed?.[i] ? "var(--c2-red-bg)" : "var(--c2-amber-bg)"} />))}
          {grid.map((g) => (<g key={g}><line x1={x0} x2={x1} y1={Y(g)} y2={Y(g)} stroke="var(--c2-line-soft)" />
            <text x={x0 - 10} y={Y(g) + 4} textAnchor="end" fontSize="12" fill="var(--c2-ink-3)">{g}</text></g>))}
          <line x1={x0} x2={x1} y1={Y(m.z_thresh)} y2={Y(m.z_thresh)} stroke="var(--c2-alarm-line)" strokeWidth="2" strokeDasharray="6 4" />
          <text x={x1} y={Y(m.z_thresh) - 6} textAnchor="end" fontSize="12" fontWeight="600" fill="var(--c2-amber-fg)">alarm line z {sg(m.z_thresh, 2)}</text>
          {series.map((s) => (
            <g key={s.f}><polyline points={s.v.map((v, k) => `${X(k).toFixed(1)},${Y(v).toFixed(1)}`).join(" ")} fill="none" stroke={s.col} strokeWidth="2" strokeDasharray={s.dash} strokeLinejoin="round" />
              {s.v.map((v, k) => <circle key={k} cx={X(k)} cy={Y(v)} r={hover === k ? 5 : 3.5} fill={s.col} stroke="var(--surface)" strokeWidth="2" />)}</g>))}
          {idx.map((i, k) => {
            const v = C1V(m.c1[i] && { harm_severity: m.c1[i][0], drift_detected: m.c1[i][1] }, true)[0];
            return (<g key={`t${k}`}>
              <circle cx={X(k)} cy={y1 + 18} r="5" fill={m.c1[i] ? C1COL[v] : "none"} stroke={C1COL[v]} strokeWidth="1.5" />
              {(n <= 20 || k % Math.ceil(n / 20) === 0) && <text x={X(k)} y={y1 + 42} textAnchor="middle" fontSize="12" fill="var(--c2-ink-3)">{m.batches[i]}</text>}
              <rect x={X(k) - half} y={y0} width={half * 2} height={y1 - y0 + 30} fill="transparent" onMouseEnter={() => setHover(k)} onFocus={() => setHover(k)} tabIndex="-1" />
            </g>);
          })}
          {hover != null && <line x1={X(hover)} x2={X(hover)} y1={y0} y2={y1} stroke="var(--c2-ink-3)" strokeDasharray="2 3" />}
          <text x={18} y={(y0 + y1) / 2} fontSize="12" fill="var(--c2-ink-3)" transform={`rotate(-90 18 ${(y0 + y1) / 2})`} textAnchor="middle">Explanation change (z)</text>
        </svg>
        {h != null && (
          <div role="tooltip" style={{ position: "absolute", top: 8, [hover > n / 2 ? "left" : "right"]: 12, background: "var(--surface)", border: "1px solid var(--c2-line)", borderRadius: 10, padding: "10px 12px", fontSize: ".82rem", boxShadow: "0 4px 14px rgba(0,0,0,.08)", minWidth: 200, pointerEvents: "none" }}>
            <b>Batch #{m.batches[h]} · {clock(m.times[h])}</b>
            {series.map((s, j) => <div key={s.f} style={{ display: "flex", justifyContent: "space-between", gap: 12 }}><span><i className="c2-dot" style={{ background: s.col, margin: "0 6px 0 0" }} />{s.f}</span><b>{sg(s.v[hover])}</b></div>)}
            <div className="c2-sub2" style={{ marginTop: 4 }}>C1: {C1V(m.c1[h] && { harm_severity: m.c1[h][0], drift_detected: m.c1[h][1] })[1]} · C2: {STATE[m.states[h]][0]}</div>
          </div>)}
      </div></div>
      <div className="c2-legendrow">
        {series.map((s) => <span key={s.f}><i className="c2-dot" style={{ background: s.col, margin: 0 }} />{s.f}</span>)}
        <span><i style={{ display: "inline-block", width: 14, height: 12, background: "var(--c2-amber-bg)", border: "1px solid var(--c2-amber-line)" }} />Alarm (C1 flagged it, or the end of an earlier change)</span>
        <span><i style={{ display: "inline-block", width: 14, height: 12, background: "var(--c2-red-bg)", border: "1px solid var(--c2-red-line)" }} />C2-only alarm (C1: no harm)</span>
        <span style={{ display: "block" }}>Dots under the axis, C1's verdict: <b style={{ color: "var(--c2-red)" }}>●</b> harmful · <b style={{ color: "var(--c2-amber)" }}>●</b> drift, no harm · <b style={{ color: "var(--c2-dot)" }}>●</b> no drift</span>
      </div>
    </section>
  );
}

function Watchlist({ list, line }) {
  return (
    <div className="c2-card">
      <div className="c2-head"><span className="c2-tile t-amber"><I n="eye" size={26} /></span>
        <div className="t"><h3>Feature watchlist</h3><p>Latest z, the trend over the last 8 batches, and how many of the last 3 were above the line.</p></div></div>
      <div className="scroll-x"><div className="c2-watch" role="table" aria-label="Feature watchlist" style={{ minWidth: 420 }}>
        {["Feature", "z now", "Trend", "Last 3", "State"].map((h) => <span key={h} className="h" role="columnheader">{h}</span>)}
        {list.map((w) => {
          const col = w.state === "alarm" ? "var(--c2-red)" : w.state === "watch" ? "var(--c2-amber)" : "var(--c2-dot)";
          return [
            <span key={`${w.feature}f`} className="c2-mono" role="cell">{w.feature}</span>,
            <span key={`${w.feature}z`} role="cell" style={{ fontWeight: w.z > line ? 700 : 400 }}>{sg(w.z)}</span>,
            <span key={`${w.feature}t`} role="cell"><Sparkline values={w.trend} line={line} color={col} /></span>,
            <span key={`${w.feature}3`} role="cell">{w.last3} of 3</span>,
            <span key={`${w.feature}s`} role="cell"><span className={`c2-statepill ${w.state}`} style={{ padding: "2px 9px", fontSize: ".76rem" }}>{STATE[w.state][0]}</span></span>,
          ];
        })}
      </div></div>
    </div>
  );
}

function Heat({ m, order }) {
  const n = m.batches.length;
  return (
    <div className="c2-card">
      <div className="c2-head"><span className="c2-tile t-blue"><I n="grid" size={26} /></span>
        <div className="t"><h3>Heatmap: every feature, every batch</h3><p>Darker = further from normal. Amber outline = above the alarm line.</p></div></div>
      <div className="scroll-x">
        <div className="c2-mheat" style={{ gridTemplateColumns: `120px repeat(${n}, minmax(14px, 1fr))`, minWidth: 120 + n * 16 }}>
          <span />
          {m.batches.map((b) => <span key={b} className="n">{b}</span>)}
          {order.map((f) => [
            <span key={f} className="f" title={f}>{f}</span>,
            ...m.z[f].map((v, i) => <span key={`${f}${i}`} className={`c ${v != null && v > m.z_thresh ? "on" : ""}`}
              style={{ background: `var(--c2-heat-${heatBin(v)})` }} title={`${f} · batch #${m.batches[i]} · z ${sg(v)}`} />),
          ])}
        </div>
      </div>
      <div className="c2-legendrow" style={{ fontSize: ".8rem" }}>z:
        {["under 1", "1–2", "2–4", "4–10", "10–50", "50+"].map((l, i) => <span key={l}><i style={{ display: "inline-block", width: 14, height: 12, borderRadius: 2, background: `var(--c2-heat-${i})`, border: "1px solid var(--c2-line-soft)" }} />{l}</span>)}</div>
    </div>
  );
}

// ── Batch log ─────────────────────────────────────────────────────────────────────────────────────
function BatchLog({ rows, diagnose, busy, diagBusy, onOpenRun, onReset, resetting }) {
  const [sort, setSort] = useState(null);
  const shown = [...rows];
  if (sort === "z") shown.sort((a, b) => (b.feature_z ?? -1) - (a.feature_z ?? -1));
  if (sort === "state") { const o = { alarm: 0, watch: 1, normal: 2, unavailable: 3 }; shown.sort((a, b) => o[a.state] - o[b.state]); }
  const srcIcon = (s) => (/upload/i.test(s || "") ? "upload" : /lab/i.test(s || "") ? "flask" : "doc");
  return (
    <section className="c2-card">
      <div className="c2-headrow" style={{ alignItems: "center" }}>
        <div className="c2-head" style={{ alignItems: "center" }}><span className="c2-tile sm t-blue" style={{ width: 48, height: 48 }}><I n="list" size={26} /></span>
          <div className="t"><h3>Batch log</h3><span className="d">Every batch C1 checked, what C1 said, and what C2's monitor saw. Newest first.</span></div></div>
        <div style={{ display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
          <div className="c2-seg" role="group" aria-label="Sort">
            <button type="button" aria-pressed={sort == null} onClick={() => setSort(null)}>Newest</button>
            <button type="button" aria-pressed={sort === "z"} onClick={() => setSort("z")}>Highest z</button>
            <button type="button" aria-pressed={sort === "state"} onClick={() => setSort("state")}>Alarms first</button></div>
          <span className="c2-sub2" style={{ maxWidth: 260, textAlign: "right" }}>Reset clears the log and relearns normal noise from clean data. Use before a demo or after a repair.</span>
          <button className="c2-btn danger" type="button" disabled={resetting} onClick={onReset}><I n="reset" size={17} />{resetting ? "Resetting…" : "Reset monitoring"}</button>
        </div>
      </div>
      <div className="c2-tbl-wrap">
        <table className="c2-tbl" style={{ minWidth: 1020 }}>
          <thead><tr>{["#", "Time", "Source", "C1 says", "C2 monitor: feature to watch", "C2 state", "Diagnosis", "Actions"].map((h) => <th key={h} scope="col">{h}</th>)}</tr></thead>
          <tbody>
            {shown.slice(0, 50).map((b) => {
              const [cv, ct] = C1V(b.c1), st = STATE[b.state] || STATE.normal;
              const z = b.feature_z, barCol = b.state === "normal" ? "var(--c2-green)" : z >= 8 ? "var(--c2-red)" : "var(--c2-amber)";
              let dx, dxSub, dxCol = "var(--c2-ink-2)";
              if (b.c2_run) { dx = `Run #${b.c2_run}${b.c2_root ? ` · ${b.c2_root}` : ""}`; dxSub = "Diagnosed"; dxCol = "var(--accent)"; }
              else if (b.state === "alarm" && b.c1_missed) { dx = "C2 alarm"; dxSub = "Persistent · C1 missed it"; dxCol = "var(--c2-red-fg)"; }
              else if (b.state === "alarm") { dx = "Alarm"; dxSub = C1V(b.c1)[0] === "harm" ? "Persistent · C1 agrees" : "Persistent · earlier change fading"; dxCol = "var(--c2-red-fg)"; }
              else if (b.state === "watch") { dx = "Watching"; dxSub = "Keep monitoring"; dxCol = "var(--c2-amber-fg)"; }
              else if (b.state === "unavailable") { dx = "Not monitored"; dxSub = "batch could not be rebuilt"; }
              else { dx = "Not needed"; dxSub = "Within normal range"; }
              const canDiag = !b.c2_run && b.state !== "unavailable";
              const strong = b.state === "alarm" || b.state === "watch";
              const runNo = b.c2_run || null;
              return (
                <tr key={b.batch}>
                  <td className="c2-edge" style={{ "--edge": EDGE[b.state] }}>#{b.batch}</td>
                  <td style={{ whiteSpace: "nowrap" }}>{new Date(b.timestamp).toLocaleDateString([], { month: "short", day: "numeric", year: "numeric" })}<span className="c2-sub2">{clock(b.timestamp)}</span></td>
                  <td><span className="c2-srcchip"><I n={srcIcon(b.source)} size={14} />{shortSource(b.source)}</span>{b.n_rows ? <span className="c2-sub2">{b.n_rows.toLocaleString()} rows</span> : null}</td>
                  <td><span className={`c2-chip ${C1CHIP[cv]}`}>{ct}</span></td>
                  <td style={{ minWidth: 200 }}>{b.feature ? (<div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
                    <span style={{ display: "flex", justifyContent: "space-between", gap: 10 }}><span className="c2-mono">{b.feature}</span><b style={{ fontWeight: 600 }}>z = {sg(z)}</b></span>
                    <span className="c2-bar"><span style={{ width: `${Math.min(100, (Math.log10((z ?? 0) + 1) / Math.log10(20)) * 100)}%`, background: barCol }} /></span>
                    <span className="c2-sub2">{b.state === "normal" ? "highest this batch" : `${b.feature_hits ?? 1} of last 3 above the line`}</span></div>) : "–"}</td>
                  <td><span className={`c2-statepill ${st[1]}`}><I n={st[2]} size={15} />{st[0]}</span></td>
                  <td><span style={{ color: dxCol, fontWeight: 600 }}>{dx}</span><span className="c2-sub2">{dxSub}</span></td>
                  <td style={{ whiteSpace: "nowrap" }}>
                    {canDiag ? <button className={`c2-btn ${strong ? "primary" : "outline"}`} type="button" disabled={diagBusy} onClick={() => diagnose(b.batch)}><I n="search" size={16} />{busy === b.batch ? "Diagnosing…" : "Diagnose"}<I n="chevron" size={14} /></button>
                      : runNo ? <button className="c2-btn ghost" type="button" onClick={() => onOpenRun(runNo)}><I n="eye" size={16} />View<I n="chevron" size={14} /></button>
                        : b.run_ref && b.state !== "unavailable" && /^\d{8}|^[a-z0-9_-]{6,}$/i.test(b.run_ref) ? <Link className="c2-btn ghost" to={`/runs/${b.run_ref}`}><I n="eye" size={16} />View<I n="chevron" size={14} /></Link>
                          : <span className="c2-sub2">–</span>}
                  </td>
                </tr>);
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}
