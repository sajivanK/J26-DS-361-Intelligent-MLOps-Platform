// C2 business report (FR7) for one stored run: /c2/report/:run, in the same tab.
// Back returns to the C2 page; Export saves it as PDF (print), Markdown or JSON.
// The text is written by the configured language model from aggregate evidence only and checked
// number by number (claim check); without a model, or when the check fails, template text is used.
import { useState } from "react";
import { Link, useLocation, useNavigate, useParams } from "react-router-dom";
import { api } from "../../api.js";
import { useApi } from "../../hooks.js";
import { useSession } from "../../session.jsx";
import { ErrorBox } from "../../ui/parts.jsx";
import { RunBars } from "./C2Charts.jsx";
import { StatePill } from "./C2Page.jsx";
import { I } from "./c2icons.jsx";
import { sg, STATE, time } from "./c2format.js";
import "./C2Page.css";

export default function C2Report() {
  const { run } = useParams();
  const navigate = useNavigate();
  const location = useLocation();
  const [refresh, setRefresh] = useState(0);
  const rec = useApi(() => api.c2Run(run), [run]);
  const rep = useApi(() => api.c2Report(run, refresh > 0), [run, refresh]);
  const [msg, setMsg] = useState(null);

  const back = () => (location.key !== "default" ? navigate(-1) : navigate("/c2"));
  const saveJson = () => {
    const blob = new Blob([JSON.stringify({ report: rep.data, run: rec.data }, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob); a.download = `c2-report-run-${run}.json`; a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  };
  const savePdf = () => { setMsg("In the print dialog choose “Save as PDF”. The sidebar and buttons are left out."); setTimeout(() => window.print(), 50); };

  return (
    <div className="c2x">
      <div className="c2-rtop">
        <button className="c2-btn ghost" type="button" onClick={back}>← Back to diagnosis</button>
        <div className="c2-actions">
          <button className="c2-btn ghost" type="button" disabled={rep.loading} onClick={() => setRefresh((r) => r + 1)} title="Ask the writer for new wording (numbers stay the same)">Write again</button>
          <button className="c2-btn primary" type="button" disabled={!rep.data} onClick={savePdf}>Export PDF</button>
          <a className="c2-btn ghost" href={api.c2ReportMdUrl(run)} download={`c2-report-run-${run}.md`}>Markdown</a>
          <button className="c2-btn ghost" type="button" disabled={!rep.data} onClick={saveJson}>JSON</button>
        </div>
      </div>
      {msg && <div className="c2-msg">{msg}</div>}
      <ErrorBox error={rec.error || rep.error} />
      {rep.loading || rec.loading ? (
        <div className="c2-wait" role="status">
          <b>{refresh ? "Writing the report again…" : "Preparing the report…"}</b>
          <span className="small muted">The first time a run's report is opened, the language model writes it from the run's evidence and every number is checked. This usually takes 10 to 20 seconds. If the model does not answer within about 35 seconds, the checked template text is used instead. Later visits open instantly.</span>
        </div>)
        : rep.data && rec.data && <Paper rep={rep.data} rec={rec.data} />}
    </div>
  );
}

const SecHead = ({ tile, icon, title, children }) => (
  <div className="c2-head"><span className={`c2-tile sm ${tile}`} style={{ width: 48, height: 48 }}><I n={icon} size={26} /></span>
    <div className="t"><h3>{title}</h3>{children}</div></div>);

function Paper({ rep, rec }) {
  const k = rep.kpis, d = rep.decision, ev = rep.evidence, st = rec.state;
  const quiet = st === "clean" || st === "harmless";
  const decided = d.level === "ok" ? "ok" : d.level === "watch" ? "watch" : "";
  const writer = rep.writer === "template" ? "template text" : `written by ${rep.writer.split(":").pop()}`;
  return (
    <article className="c2-paper">
      <header className="stack tight">
        <span className="c2-eyebrow">Model health report · UCI Adult income model · C2 Root Cause Diagnosis</span>
        <h2>{rep.title}</h2>
        <div className="row small muted" style={{ flexWrap: "wrap", gap: 10, alignItems: "center" }}><span className="c2-mono">{time(rec.timestamp)} · {rec.to_c4.diagnosis_version}</span><StatePill state={st} />
          <span className={`badge ${rep.claim_check.passed ? "ok" : ""}`}>{writer} · claim check {rep.claim_check.passed ? "passed" : "failed"}</span></div>
        {rep.fallback_reason && <span className="small muted">Template text used: {rep.fallback_reason}.</span>}
      </header>

      <div className="c2-summary">{rep.sections.summary}</div>
      <div className={`c2-decision ${decided}`}>
        <span className={`c2-pill ${d.level === "ok" ? "ok" : d.level === "watch" ? "watch" : "warn"}`}>decision</span>
        <div><div className="big">{d.head}</div><div>{d.text}</div></div>
      </div>

      <section className="stack"><span className="c2-eyebrow">At a glance</span>
        <div className="c2-rtiles">
          <Tile col="var(--c2-orange)" k="Decisions changed" v={`${sg(k.decisions_changed_percent)}%`}
            s={`harmful above 2%${k.detection_step_predicted_points != null ? ` · C1 predicted ${sg(k.detection_step_predicted_points)} pp accuracy drop` : ""}`} />
          <Tile col="var(--c2-red)" k="Decisions at risk" v={(k.decisions_changed_rows ?? 0).toLocaleString()}
            s={k.estimated_accuracy_loss_points != null ? `in this batch · est. accuracy loss ${sg(k.estimated_accuracy_loss_points)} pp` : "in this batch · accuracy not measured (no labels)"} />
          <Tile col="var(--c2-blue)" k="Customers affected" v={quiet ? "0%" : `${k.affected_share_percent}%`}
            s={quiet ? "no group" : `of this batch${k.training_share_percent != null ? ` · ${k.training_share_percent}% normally` : ""}`} />
          <Tile col="var(--c2-green)" k="Confidence" v={`${k.recent_runs_agreeing} of ${k.runs_checked}`} s="recent runs agree on the cause" />
        </div></section>

      <Trend trend={rep.charts.trend} current={rec.run} />
      <WhatChanged rep={rep} quiet={st === "clean"} />
      {!quiet && <Who rep={rep} rec={rec} />}
      <Next rep={rep} rec={rec} quiet={quiet} />
      <Evidence rep={rep} rec={rec} ev={ev} />
    </article>
  );
}

function Tile({ col, k, v, s }) {
  return (<div className="c2-rtile" style={{ "--tile": col }}><span className="k">{k}</span><span className="v">{v}</span><span className="s">{s}</span></div>);
}

const BAR = { clean: "var(--c2-green)", harmless: "var(--c2-green)", suspected: "var(--c2-yellow)", confirmed: "var(--c2-orange)" };

function Trend({ trend, current }) {
  const [mode, setMode] = useSession("c2:rmode", "pct");
  const [help, setHelp] = useState(false);
  const cur = trend.find((t) => t.run === current) || trend[trend.length - 1];
  const hasRows = trend.every((t) => t.decisions_changed_rows != null);
  return (
    <section className="c2-rsec">
      <div className="c2-headrow">
        <SecHead tile="t-orange" icon="bars" title="Decisions changed, run by run">
          <span className="c2-sub2">Green: healthy. Yellow: suspected. Orange: confirmed damage. Diamond: the accuracy drop C1 predicted for the same batch (real C1 only).</span></SecHead>
        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          <div className="c2-seg blue" role="group" aria-label="Show as">
            <button type="button" aria-pressed={mode === "rows"} disabled={!hasRows} title={hasRows ? undefined : "Open the report again with Write again to add row counts"} onClick={() => setMode("rows")}>Number of decisions</button>
            <button type="button" aria-pressed={mode !== "rows"} onClick={() => setMode("pct")}>Percentage</button></div>
          <button type="button" className="c2-btn ghost" aria-label="How to read this chart" aria-expanded={help} onClick={() => setHelp((h) => !h)} style={{ padding: "0 10px" }}><I n="info" size={20} /></button>
        </div>
      </div>
      {help && <div className="c2-msg">Each bar is one run: the share of the model's decisions that flip back when the root cause is put back to normal. Above 2% counts as harmful. A cause is confirmed only when it is found in 2 of the last 3 runs. The first run has nothing to compare with, so it reads 0.</div>}
      <div className="c2-rbarsplit">
        <div className="chart"><RunBars trend={trend} current={current} mode={mode} /></div>
        <div className="rside">
          <div className="c2-rbox"><b style={{ color: "var(--fg)" }}>Legend</b>
            <span><i className="c2-dot" style={{ background: BAR.clean }} />Healthy (no harmful drift)</span>
            <span><i className="c2-dot" style={{ background: BAR.suspected }} />Suspected (seen in 1 run)</span>
            <span><i className="c2-dot" style={{ background: BAR.confirmed }} />Confirmed damage (2 of last 3 runs)</span>
            {mode !== "rows" && <span><b style={{ color: "var(--c2-diamond)", marginRight: 6 }}>◆</b>C1 predicted accuracy drop</span>}
          </div>
          {cur && <div className="c2-rbox blue"><b style={{ color: "var(--fg)" }}>Selected run</b>
            <span><i className="c2-dot" style={{ background: BAR[cur.state] }} /><b>#{cur.run}</b> · <b>{mode === "rows" && hasRows ? cur.decisions_changed_rows.toLocaleString() : `${sg(cur.decisions_changed_percent)}%`}</b> decisions changed</span>
            <span className="c2-sub2">{STATE[cur.state].label}{cur.detection_step_predicted_points != null ? ` · C1 predicted ${sg(cur.detection_step_predicted_points)} pp` : ""}</span></div>}
        </div>
      </div>
    </section>
  );
}

function WhatChanged({ rep, quiet }) {
  const f = rep.charts.factors || [], root = rep.evidence.root_cause;
  const max = Math.max(1, ...f.map((x) => x.decisions_changed_percent));
  return (
    <section className="c2-rsec">
      <SecHead tile="t-blue" icon="trend" title={rep.section_titles.what_changed}>
        <span>{rep.sections.what_changed}</span>
        {!quiet && f.length > 0 && <span className="c2-sub2">Share of the model's decisions each factor changes (what flips back when it returns to normal).</span>}</SecHead>
      {!quiet && f.length > 0 && (
        <div className="c2-rank">{f.map((x, i) => {
          const isRoot = x.factor === root, col = isRoot ? "var(--c2-orange)" : "var(--c2-blue)";
          return [
            <span key={`${x.factor}n`} className="n" style={{ color: isRoot ? "var(--c2-orange-fg)" : "var(--c2-blue-fg)", background: isRoot ? "var(--c2-orange-bg)" : "var(--c2-blue-bg)", borderColor: isRoot ? "var(--c2-orange-line)" : "var(--c2-blue-line)" }}>#{i + 1}</span>,
            <span key={`${x.factor}f`} style={{ fontWeight: isRoot ? 700 : 400 }}>{x.factor}{isRoot ? " (root cause)" : ""}</span>,
            <span key={`${x.factor}b`} style={{ display: "flex", gap: 12, alignItems: "center" }}>
              <span className="c2-track" style={{ flex: 1, height: 14 }}><span style={{ display: "block", height: "100%", borderRadius: 4, width: `${Math.max(1, (x.decisions_changed_percent / max) * 100)}%`, background: col }} /></span>
              <b style={{ minWidth: 52, textAlign: "right", fontWeight: isRoot ? 700 : 500 }}>{sg(x.decisions_changed_percent)}%</b></span>,
          ];
        })}</div>)}
    </section>
  );
}

function Who({ rep, rec }) {
  const c3 = rec.to_c3, n = rec.n_rows || 0, rows = Math.round(c3.slice_support * n), ref = rec.localization.ref_share;
  const max = Math.max(c3.slice_support, ref || 0) * 1.2 || 1;
  const bar = (v, col) => <span className="c2-track" style={{ height: 22 }}><span style={{ display: "block", height: "100%", borderRadius: 4, width: `${(v / max) * 100}%`, background: col }} /></span>;
  return (
    <section className="c2-rsec">
      <SecHead tile="t-green" icon="people" title={rep.section_titles.who_is_affected}><span>{rep.sections.who_is_affected}</span></SecHead>
      <div className="c2-grid2" style={{ gridTemplateColumns: "minmax(0, 1.3fr) minmax(0, 1fr)", alignItems: "center" }}>
        <div className="c2-sharegrid" style={{ gridTemplateColumns: "24px 110px minmax(0, 1fr) 60px auto" }}>
          <span style={{ color: "var(--c2-ink-3)" }}><I n="db" size={20} /></span><span>Training data</span>{bar(ref || 0, "var(--c2-dot)")}<span>{ref != null ? `${(ref * 100).toFixed(1)}%` : "–"}</span><span className="c2-sub2">of training data</span>
          <span style={{ color: "var(--c2-orange)" }}><I n="people" size={20} /></span><span>This batch</span>{bar(c3.slice_support, "var(--c2-orange)")}<b>{(c3.slice_support * 100).toFixed(1)}%</b><span className="c2-sub2">{rows.toLocaleString()} of {n.toLocaleString()} rows</span>
        </div>
        <div className="c2-tool" style={{ background: "var(--c2-green-bg)", borderColor: "var(--c2-green-line)" }}>
          <span className="c2-tile sm t-green" style={{ width: 44, height: 44 }}><I n="target" size={24} /></span>
          <div className="t"><b style={{ color: "var(--c2-green-fg)" }}>Affected group</b><span className="c2-code">{c3.affected_slice_rule}</span>
            <span>{rows.toLocaleString()} of {n.toLocaleString()} records ({(c3.slice_support * 100).toFixed(1)}%)</span>
            {ref ? <span className="c2-sub2">{(c3.slice_support / ref).toFixed(1)}× more common than in training ({(ref * 100).toFixed(1)}%).</span> : null}</div>
        </div>
      </div>
    </section>
  );
}

const OWNER = {
  C3: ["var(--c2-purple-bg)", "var(--c2-purple-line)", "var(--c2-purple)", "wand"],
  C4: ["var(--c2-green-bg)", "var(--c2-green-line)", "var(--c2-green)", "shield"],
  Business: ["var(--c2-amber-bg)", "var(--c2-amber-line)", "var(--c2-amber)", "people"],
  C2: ["var(--c2-blue-bg)", "var(--c2-blue-line)", "var(--c2-blue)", "eye"],
  Platform: ["var(--c2-green-bg)", "var(--c2-green-line)", "var(--c2-green)", "check"],
};
const ROUTE = { C3: "/c3", C4: "/c4", C2: "/c2" };

function actionNote(a, rep, rec) {
  const ev = rep.evidence, rule = rec.to_c3.affected_slice_rule, nf = ev.next_factors?.[0];
  if (a.owner === "C3") return `Create targeted repair data for ${rule}, so the retrained model learns the new pattern.`;
  if (a.owner === "C4" && /retrain/i.test(a.action)) return "Retrain with the repair data; the release check decides whether the new model ships.";
  if (a.owner === "C4") return "No new model is trained until the next run confirms the cause.";
  if (a.owner === "Business") return `Check the data pipeline or business process changes behind ${rec.to_c3.root_cause_feature}.`;
  if (a.owner === "C2" && nf && /watching/i.test(a.action)) return `${nf.factor} changes ${sg(nf.decisions_changed_percent)}% of decisions now; C2 keeps checking it in the next runs.`;
  if (a.owner === "C2") return "If the same cause is found again, it is confirmed and repair is recommended.";
  return "";
}

function Next({ rep, rec, quiet }) {
  const acts = rep.decision.actions;
  return (
    <section className="c2-rsec">
      <SecHead tile="t-purple" icon="arrow" title={rep.section_titles.outlook}><span>{rep.sections.outlook}</span></SecHead>
      <span className="c2-eyebrow">Recommended actions</span>
      <div className="stack" style={{ gap: 10 }}>
        {acts.map((a, i) => {
          const o = OWNER[a.owner] || OWNER[a.owner.split(",")[0].trim()] || OWNER.Platform;
          const to = !quiet && ROUTE[a.owner];
          const body = (<>
            <span className="num" style={{ background: o[2] }}>{i + 1}</span>
            <span style={{ color: o[2] }}><I n={o[3]} size={24} /></span>
            <span className="t"><b>{a.action}</b><span>{actionNote(a, rep, rec)}</span></span>
            <span className="c2-chip" style={{ background: "var(--surface)", border: `1px solid ${o[1]}` }}>Owner: {a.owner}</span>
            {to && <I n="chevron" size={18} />}</>);
          const style = { background: o[0], borderColor: o[1] };
          return to ? <Link key={a.action} className="c2-act" to={to} style={style}>{body}</Link>
            : <div key={a.action} className="c2-act" style={style}>{body}</div>;
        })}
      </div>
    </section>
  );
}

function Row({ k, v, strong }) {
  return <div className="row"><span className="c2-muted">{k}</span>{strong ? <b>{v}</b> : <span className="c2-mono">{v}</span>}</div>;
}

function Evidence({ rep, rec, ev }) {
  const cc = rep.claim_check, show = (v) => (v == null ? "–" : String(v));
  return (
    <section className="c2-rsec">
      <SecHead tile="t-blue" icon="clipboard" title="Evidence behind the text">
        <span>The writer saw only these aggregate facts (never data rows). The claim check compared every number in the text with them: {cc.numbers_checked} checked{cc.problems.length ? `, problems: ${cc.problems.join("; ")}` : ", all matched"}.</span></SecHead>
      <div className="c2-grid2">
        <div className="c2-evcard">
          <div className="h"><I n="doc" size={18} />Run details</div>
          <div className="rows">
            <Row k="Run" v={ev.run} />
            <div className="row"><span className="c2-muted">State</span><StatePill state={ev.state} /></div>
            <Row k="Accuracy measured" v={show(ev.accuracy_measured)} />
            <Row k="Decisions changed rows" v={(ev.decisions_changed_rows ?? 0).toLocaleString()} />
            <Row k="Root cause" v={show(ev.root_cause ?? "none")} />
            {ev.recent_runs_agreeing != null && <Row k="Recent runs agreeing" v={`${ev.recent_runs_agreeing} of ${ev.runs_checked}`} />}
            {ev.affected_group && <Row k="Affected group" v={ev.affected_group} />}
            <Link to="/c2" className="c2-tool" style={{ marginTop: 10, textDecoration: "none", color: "inherit", background: "var(--c2-panel)", borderColor: "var(--c2-line-soft)" }}>
              <span className="c2-tile xs t-red"><I n="code" size={17} /></span>
              <span className="t"><span className="c2-sub2">Repair rule sent to C3</span><span className="c2-code">{rec.to_c3.affected_slice_rule}</span><span className="c2-link">Open the full diagnosis →</span></span></Link>
          </div>
        </div>
        <div className="c2-evcard">
          <div className="h"><I n="chartUp" size={18} />Model impact (UCI Adult, XGBoost)</div>
          <div className="rows">
            <Row k="Rows in batch" v={(ev.rows_in_batch ?? 0).toLocaleString()} />
            <Row k="Decisions changed" v={`${sg(ev.decisions_changed_percent)}%`} strong />
            <Row k="Estimated accuracy loss" v={ev.estimated_accuracy_loss_points == null ? "not estimated" : `${sg(ev.estimated_accuracy_loss_points)} pp`} />
            {ev.detection_step_predicted_points != null && <Row k="C1 predicted drop" v={`${sg(ev.detection_step_predicted_points)} pp`} />}
            {ev.model_reliance_vs_normal != null && <Row k="Model reliance vs normal" v={`${ev.model_reliance_vs_normal}×`} />}
            {ev.affected_share_percent != null && <Row k="Affected share" v={`${ev.affected_share_percent}%`} strong />}
            {ev.training_share_percent != null && <Row k="Share in training data" v={`${ev.training_share_percent}%`} />}
            <div className="c2-note" style={{ marginTop: 10 }}><I n="shield" size={18} /><span>These numbers are aggregated from model outputs and monitoring signals. No individual records are shown.</span></div>
          </div>
        </div>
      </div>
    </section>
  );
}
