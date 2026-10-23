// C2 business report (FR7) for one stored run: /c2/report/:run, in the same tab.
// Back returns to the C2 page; Export saves it as PDF (print), Markdown or JSON.
// The text is written by the configured language model from aggregate evidence only and checked
// number by number (claim check); without a model, or when the check fails, template text is used.
import { useState } from "react";
import { Link, useLocation, useNavigate, useParams } from "react-router-dom";
import { api } from "../../api.js";
import { useApi } from "../../hooks.js";
import { ErrorBox } from "../../ui/parts.jsx";
import { FactorBars, ShareBars, TrendBars } from "./C2Charts.jsx";
import { StatePill } from "./C2Page.jsx";
import { sg, time } from "./c2format.js";
import "./C2Page.css";

const ORDER = ["summary", "what_changed", "who_is_affected", "outlook"];

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
    <>
      <div className="c2-rtop">
        <button className="btn" type="button" onClick={back}>← Back to diagnosis</button>
        <div className="c2-actions">
          <button className="btn" type="button" disabled={rep.loading} onClick={() => setRefresh((r) => r + 1)} title="Ask the writer for new wording (numbers stay the same)">Write again</button>
          <button className="btn primary" type="button" disabled={!rep.data} onClick={savePdf}>Export PDF</button>
          <a className="btn" href={api.c2ReportMdUrl(run)} download={`c2-report-run-${run}.md`}>Markdown</a>
          <button className="btn" type="button" disabled={!rep.data} onClick={saveJson}>JSON</button>
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
    </>
  );
}

function Paper({ rep, rec }) {
  const k = rep.kpis, d = rep.decision, ev = rep.evidence, st = rec.state;
  const decided = d.level === "ok" ? "ok" : d.level === "watch" ? "watch" : "";
  const writer = rep.writer === "template" ? "template text" : `written by ${rep.writer.split(":").pop()}`;
  return (
    <article className="c2-paper">
      <header className="stack tight">
        <span className="eyebrow">Model health report · UCI Adult income model · C2 Root Cause Diagnosis</span>
        <h2>{rep.title}</h2>
        <div className="row small muted"><span className="mono">{time(rec.timestamp)} · {rec.to_c4.diagnosis_version}</span><StatePill state={st} />
          <span className={`badge ${rep.claim_check.passed ? "ok" : ""}`}>{writer} · claim check {rep.claim_check.passed ? "passed" : "failed"}</span></div>
        {rep.fallback_reason && <span className="small muted">Template text used: {rep.fallback_reason}.</span>}
      </header>

      <div className="c2-summary">{rep.sections.summary}</div>
      <div className={`c2-decision ${decided}`}>
        <span className={`c2-pill ${d.level === "ok" ? "ok" : d.level === "watch" ? "watch" : "warn"}`}>decision</span>
        <div><div className="big">{d.head}</div><div>{d.text}</div></div>
      </div>

      <section className="stack"><h3 className="sec">At a glance</h3>
        <div className="c2-tiles">
          <Tile col="var(--chart-2)" k="Accuracy lost" v={`${sg(k.accuracy_lost_points)} pp`}
            s={`harmful above 2 pp${k.detection_step_predicted_points != null ? ` · C1 predicted ${sg(k.detection_step_predicted_points)} pp` : ""}`} />
          <Tile col="var(--chart-5)" k="Extra wrong predictions" v={k.extra_wrong_predictions_per_1000} s="per 1,000 customers scored" />
          <Tile col="var(--chart-1)" k="Customers affected" v={st === "clean" || st === "harmless" ? "0%" : `${k.affected_share_percent}%`}
            s={st === "clean" || st === "harmless" ? "no group" : `of this batch${k.training_share_percent != null ? ` · ${k.training_share_percent}% normally` : ""}`} />
          <Tile col="var(--chart-3)" k="Confidence" v={`${k.recent_runs_agreeing} of ${k.runs_checked}`} s="recent runs agree on the cause" />
        </div></section>

      <section className="stack"><h3 className="sec">Accuracy lost, run by run</h3>
        <p className="c2-sub">Green: healthy. Yellow: suspected. Orange: confirmed damage. Diamond: the drop C1 predicted for the same batch (when C1 ran).</p>
        <TrendBars trend={rep.charts.trend} current={rec.run} /></section>

      {ORDER.slice(1).map((key) => (
        <section key={key} className="stack"><h3 className="sec">{rep.section_titles[key]}</h3><p>{rep.sections[key]}</p>
          {key === "what_changed" && st !== "clean" && (<>
            <p className="c2-sub">Accuracy each factor would give back if it returned to normal.</p><FactorBars factors={rep.charts.factors} root={rep.evidence.root_cause} /></>)}
          {key === "who_is_affected" && st !== "clean" && st !== "harmless" && (
            <ShareBars training={rec.localization.ref_share} batch={rec.to_c3.slice_support} />)}
        </section>))}

      <section className="stack"><h3 className="sec">Recommended actions</h3>
        <div className="c2-actlist">{d.actions.map((a, i) => (
          <div key={a.action} className="c2-act"><span className="n">{i + 1}</span><span>{a.action}</span><span className="who">owner: {a.owner}</span></div>))}</div></section>

      <section className="stack tight"><h3 className="sec">Evidence behind the text</h3>
        <p className="c2-sub">The writer saw only these aggregate facts (never data rows). The claim check compared every number in the text with them: {rep.claim_check.numbers_checked} checked{rep.claim_check.problems.length ? `, problems: ${rep.claim_check.problems.join("; ")}` : ", all matched"}.</p>
        <div className="c2-evid">{Object.entries(ev).filter(([key]) => !["trend", "decision", "next_factors"].includes(key)).map(([key, v]) => (
          <div key={key}><span>{key.replace(/_/g, " ")}</span><span className="mono">{v == null ? "–" : String(v)}</span></div>))}</div>
        <p className="small muted">Repair rule sent to C3: <span className="mono">{rec.to_c3.affected_slice_rule}</span> · <Link to="/c2">open the full diagnosis</Link></p>
      </section>
    </article>
  );
}

function Tile({ col, k, v, s }) {
  return (<div className="c2-tile" style={{ "--tile": col }}><span className="k">{k}</span><span className="v">{v}</span><span className="s">{s}</span></div>);
}
