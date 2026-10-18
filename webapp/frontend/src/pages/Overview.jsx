import { Link, useNavigate } from "react-router-dom";
import { api } from "../api.js";
import { useApi } from "../hooks.js";
import { componentInfo, componentState, usePlatform } from "../platform.jsx";
import { STEPS, TEAM } from "../steps.js";
import { SourceBadge } from "../ui/parts.jsx";
import { stepSummary } from "../ui/PipelineSteps.jsx";
import "./Overview.css";

export default function Overview() {
  const nav = useNavigate();
  const { health, datasets } = usePlatform();
  const latest = useApi(api.latestRun, []);
  const run = latest.data;
  const c1 = componentInfo(health, "c1");
  const engines = c1?.engines ? Object.entries(c1.engines).filter(([, e]) => e.loaded).map(([k]) => k.toUpperCase()) : [];

  return (
    <>
      <section className="hero2">
        <div className="stack">
          <span className="eyebrow">J26-DS-361 · SLIIT final-year research</span>
          <h1>Intelligent MLOps Platform</h1>
          <p className="muted" style={{ fontSize: "1rem", maxWidth: "58ch" }}>
            Production data drifts, but most drift alarms are harmless. This platform finds the drift that actually
            hurts the model, explains why, repairs the data, and decides whether the retrained model is safe to release.
          </p>
          <div className="cta">
            <button className="btn primary" type="button" onClick={() => nav("/lab")}>Open Drift Lab</button>
            <button className="btn" type="button" onClick={() => nav("/c1")}>See C1 Drift Detection</button>
          </div>
        </div>
        <div className="stat3">
          <div><b>4</b><span>components, one pipeline</span></div>
          <div><b>{datasets.length || 2}</b><span>datasets: {datasets.length ? datasets.map((d) => d.label.split(" (")[0]).join(", ") : "UCI Adult, IEEE-CIS"}</span></div>
          <div><b>{run ? `${run.runtime_sec} s` : "-"}</b><span>last full pipeline run</span></div>
        </div>
      </section>

      <section className="stack">
        <div className="head"><h2>How it works</h2><span className="small muted">each step is one component; status is live from the backend</span></div>
        <div className="flow4">
          {STEPS.map((s) => {
            const st = componentState(health, s.component);
            return (
              <button key={s.component} type="button" className="fcard" onClick={() => nav(s.path)}>
                <div className="top"><span className="step">STEP {s.n} · {s.label.toUpperCase()}</span><SourceBadge source={st} /></div>
                <h3>C{s.n} {s.title}</h3>
                <p className="q">{s.question}</p>
                <span className="out">{s.output}</span>
                <span className="who">{s.owner}{s.component === "c1" && engines.length ? ` · engines loaded: ${engines.join(", ")}` : ""}</span>
              </button>
            );
          })}
        </div>
      </section>

      <section className="stack">
        <div className="head">
          <h2>Latest pipeline run</h2>
          {run && (
            <span className="small muted mono">
              run {run.run_id} · {run.origin?.label || run.context?.current_dataset_uri} · {run.context?.dataset?.toUpperCase()} · {(run.origin?.rows ?? "?").toLocaleString()} rows · {run.runtime_sec} s
              {run.origin?.run_ref && <> · <Link to={`/runs/${run.origin.run_ref}`}>open</Link></>}
            </span>
          )}
        </div>
        {latest.loading ? <div className="empty">Loading…</div>
          : !run ? <div className="empty">No pipeline run yet. Open the <b>Drift Lab</b> and press <b>Run full pipeline</b>.</div>
            : (
              <div className="lastrun">
                {STEPS.map((s) => {
                  const step = run.steps.find((x) => x.component === s.component);
                  const sum = step?.status === "ok" ? stepSummary(s.component, run.outputs) : null;
                  return (
                    <div key={s.component}>
                      <span className="k">C{s.n} {s.label}</span>
                      <span className="v">{sum ? sum.text : step?.status === "error" ? "error" : run.status === "no_action" ? "not needed" : "not run"}</span>
                      {sum?.sub && <span className="small muted">{sum.sub}</span>}
                      {step && s.component !== "c1" && <SourceBadge source={step.source} />}
                    </div>
                  );
                })}
              </div>
            )}
      </section>

      <section className="stack">
        <h2>Team</h2>
        <div className="team">
          {TEAM.map((t) => <div key={t.name}><b>{t.name}</b><span className="muted">{t.role}</span></div>)}
        </div>
      </section>
    </>
  );
}
