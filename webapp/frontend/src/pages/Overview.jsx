import { Link, useNavigate } from "react-router-dom";
import { api } from "../api.js";
import { useApi } from "../hooks.js";
import { Icon } from "../layout/icons.jsx";
import { componentInfo, componentState, usePlatform } from "../platform.jsx";
import { STEPS, TEAM } from "../steps.js";
import { stepTile } from "../ui/runTiles.jsx";
import { IconChip, MiniBars, Spark, StepDots, useRecent } from "../ui/viz.jsx";
import "./Overview.css";

const STATE_PILL = { real: ["real", "teal"], fake: ["fake", "purple"], off: ["off", "red"] };
const STEP_SUB = {
  c1: "Did the data drift, and how much will it hurt the model? No labels.",
  c2: "Which feature causes the harm, and which group of records does it hurt?",
  c3: "Can new training data for that group fix the model, safely and provably?",
  c4: "Is the retrained model safe to release: deploy, review, or escalate?",
};

export default function Overview() {
  const nav = useNavigate();
  const { health, datasets, alertsWaiting } = usePlatform();
  const latest = useApi(api.latestRun, []);
  const models = useApi(api.c3Models, []);
  const { runs, s } = useRecent(0);
  const run = latest.data;
  const states = STEPS.map((st) => componentState(health, st.component));
  const nReal = states.filter((x) => x === "real").length;
  const c1 = componentInfo(health, "c1");
  const c4 = componentInfo(health, "c4");
  const engines = c1?.engines ? Object.entries(c1.engines).filter(([, e]) => e.loaded).map(([k]) => k.toUpperCase()) : [];
  const runSummary = run?.origin?.run_ref ? runs.find((r) => r.run_id === run.origin.run_ref) : null;
  const oc = s.outcomes;
  const bad = (oc["false alert"] || 0) + (oc["missed harm"] || 0);

  return (
    <>
      <section className="ov-hero">
        <div className="ov-pitch">
          <span className="eyebrow ov-tag">J26-DS-361 · SLIIT final-year research</span>
          <h1>Find the <span className="hl">drift</span> that hurts.<br />Explain it, repair it, release safely.</h1>
          <p>Production data drifts, but most drift alarms are harmless. This platform measures the harm a drift
            causes without labels, finds its cause, repairs the data, and decides whether the retrained model is safe to release.</p>
          <div className="cta">
            <button className="btn primary" type="button" onClick={() => nav("/lab")}><Icon name="lab" size={18} className="" />Open Drift Lab<Icon name="arrow" size={16} className="" /></button>
            <button className="btn" type="button" onClick={() => nav("/c1")}><Icon name="bars" size={18} className="" />See C1 Drift Detection</button>
          </div>
        </div>
        <div className="ov-stats">
          <div className="card ov-stat">
            <div className="top"><IconChip name="layers" tone="teal" />
              <MiniBars w={70} h={34} caption="C1 C2 C3 C4" items={states.map((x) => ({ v: 1, c: x === "real" ? "#14B8A6" : x === "fake" ? "#DCD5FA" : "#FBC7C2" }))} /></div>
            <span className="num big">{health ? `${nReal} / 4` : "-"}</span><span className="lab">components real in the pipeline</span>
          </div>
          <div className="card ov-stat">
            <div className="top"><IconChip name="db" tone="purple" /></div>
            <span className="num big">{datasets.length || "-"}</span>
            <span className="lab">datasets: {datasets.length ? datasets.map((d) => d.label.split(" (")[0]).join(", ") : "not loaded"}</span>
          </div>
          <div className="card ov-stat">
            <div className="top"><IconChip name="checkCircle" tone="green" /><MiniBars w={100} h={34} items={s.inside} caption={s.inside.length ? `last ${s.inside.length} runs` : null} /></div>
            <span className="num big acc">{s.insideOf ? `${s.insideN} / ${s.insideOf}` : "-"}</span><span className="lab">saved runs: true harm inside C1's 95% interval</span>
          </div>
          <div className="card ov-stat">
            <div className="top"><IconChip name="bolt" tone="blue" /><Spark values={s.runtimes} w={100} h={34} color="#2563EB" caption={s.runtimes.length > 1 ? `last ${s.runtimes.length} pipeline runs` : null} /></div>
            <span className="num big">{run ? `${run.runtime_sec.toFixed(1)} s` : "-"}</span><span className="lab">last full pipeline run</span>
          </div>
        </div>
      </section>

      <section className="card ov-sec">
        <div className="ov-head">
          <div className="row" style={{ gap: 12 }}><Icon name="nodes" size={26} className="" style={{ color: "var(--accent)" }} /><h2>The pipeline</h2>
            <span className="muted small">From detection to safe release, one batch at a time</span></div>
          <span className="muted small">Live status from the backend · click a step to open it</span>
        </div>
        <div className="ov-flow">
          {STEPS.map((st, i) => {
            const state = states[i];
            const waiting = st.component === "c4" && c4 && c4.worker && c4.worker !== "reachable";
            const [label, tone] = waiting ? ["waiting for worker", "purple"] : STATE_PILL[state];
            return [
              i > 0 && <span key={"a" + i} className={`ov-arrow ${state === "real" ? "" : "dim"}`} aria-hidden="true"><Icon name="chevron" size={16} className="" strokeWidth={2.6} /></span>,
              <Link key={st.component} to={st.path} className={`card ov-step ${state === "real" ? "" : "pending"}`}>
                <div className="top"><span className="eyebrow">Step {st.n} · {st.label}</span><span className={`pill tone-${tone}`}>{label}</span></div>
                <div className="body"><IconChip name={st.icon} tone={state === "real" ? "teal" : "purple"} size="sm" iconSize={18} />
                  <div><b>C{st.n} {st.title}</b><p>{STEP_SUB[st.component]}</p></div></div>
                <div className="foot"><span>{st.owner}{st.component === "c1" && engines.length ? ` · engines ${engines.join(", ")}` : ""}</span>
                  <span className="chevbtn"><Icon name="chevron" size={14} className="" strokeWidth={2.4} /></span></div>
              </Link>,
            ];
          })}
        </div>
      </section>

      <section className="card ov-sec">
        <div className="ov-head">
          <div className="row" style={{ gap: 12 }}><span className="ic sm tone-teal"><Icon name="play" size={14} className="" /></span><h2>Latest pipeline run</h2>
            {run && <span className={`pill tone-${run.status === "completed" ? "green" : run.status === "failed" ? "red" : "amber"}`}>{run.status.replaceAll("_", " ")}</span>}</div>
          {run && (
            <span className="small muted mono">
              run {run.run_id} · {run.origin?.label || run.context?.current_dataset_uri} · {run.context?.dataset?.toUpperCase()} · {(run.origin?.rows ?? "?").toLocaleString()} rows · {run.runtime_sec} s
              {run.origin?.run_ref && <> · <Link to={`/runs/${run.origin.run_ref}`}>open</Link></>}
            </span>)}
        </div>
        {latest.loading ? <div className="empty">Loading…</div>
          : !run ? <div className="empty">No pipeline run yet. Open the <b>Drift Lab</b> and press <b>Run full pipeline</b>.</div>
            : (
              <div className="ov-tiles">
                {STEPS.map((st) => {
                  const t = stepTile(st.component, run, runSummary || {});
                  const chart = st.component === "c1" ? <Spark values={s.c1} w={150} h={32} color="#D92D20" caption="C1 harm, last runs" />
                    : st.component === "c2" ? <MiniBars items={s.c2} w={150} h={30} caption={s.c2.length ? `C2 correct, last ${s.c2.length}` : null} />
                      : st.component === "c3" ? <MiniBars items={s.c3} w={150} h={30} caption={s.c3.length ? `C3 results, last ${s.c3.length}` : null} />
                        : t.source === "fake" ? <StepDots caption="no real decision yet" /> : null;
                  return (
                    <div key={st.component} className={`ttile tone-${t.tone}`}>
                      <div className="row" style={{ gap: 10, flexWrap: "nowrap", alignItems: "flex-start" }}>
                        <span className="ic sm"><Icon name={t.icon} size={18} className="" strokeWidth={2.4} /></span>
                        <div className="stack" style={{ gap: 5, minWidth: 0 }}>
                          <span className="eyebrow" style={{ color: "inherit" }}>C{st.n} · {st.label}</span>
                          <span className={`tt ${t.mono ? "mono" : ""}`}>{t.title}</span>
                          <span className="ts">{t.text}</span>
                        </div>
                      </div>
                      {chart && <div style={{ marginTop: "auto", paddingTop: 6 }}>{chart}</div>}
                    </div>);
                })}
              </div>)}
      </section>

      <section className="ov-two">
        <div className="card ov-sec">
          <div className="ov-head"><div className="row" style={{ gap: 12 }}><IconChip name="db" tone="teal" size="sm" iconSize={17} /><h2>Saved runs</h2></div>
            <Link to="/runs" className="small" style={{ fontWeight: 600 }}>Open run history →</Link></div>
          <div className="ov-counts">
            <div><span className="num">{s.total}</span><span>runs</span></div>
            <div><span className="num" style={{ color: "var(--accent)" }}>{oc["correct alert"] || 0}</span><span>correct alerts</span></div>
            <div><span className="num" style={{ color: "#2563EB" }}>{oc["correct quiet"] || 0}</span><span>correct quiet</span></div>
            <div><span className="num" style={{ color: "#B42318" }}>{bad}</span><span>missed or false</span></div>
          </div>
          {s.total > 0 && (
            <div className="ov-share" role="img" aria-label={`Outcomes of ${s.total} runs`}>
              <div style={{ flex: oc["correct alert"] || 0, background: "#14B8A6" }} />
              <div style={{ flex: oc["correct quiet"] || 0, background: "#60A5FA" }} />
              <div style={{ flex: bad, background: "#F04438" }} />
              <div style={{ flex: oc["no truth"] || 0, background: "#E1E5EA" }} />
            </div>)}
        </div>
        <div className="card ov-sec">
          <div className="ov-head"><div className="row" style={{ gap: 12 }}><IconChip name="bell" tone="amber" size="sm" iconSize={17} /><h2>Waiting for you</h2></div></div>
          {alertsWaiting > 0 ? (
            <div className="ov-alert">
              <Icon name="warn" size={20} className="" style={{ color: "var(--t-amber-ic)" }} />
              <span><b>{alertsWaiting} drift alert{alertsWaiting === 1 ? "" : "s"}</b> found by C1 {alertsWaiting === 1 ? "is" : "are"} waiting for a C2 diagnosis.</span>
              <Link to="/c2" className="ov-alertbtn">Diagnose →</Link>
            </div>
          ) : <div className="ov-alert calm"><Icon name="checkCircle" size={20} className="" style={{ color: "var(--t-green-ic)" }} /><span>Nothing waiting. Every C1 alert has been diagnosed or dismissed.</span></div>}
          <span className="small muted">Team: {TEAM.map((t) => `${t.name} (${t.role.split(" ·")[0]})`).join(" · ")}</span>
        </div>
      </section>

      {(models.data?.models?.length > 0 || models.data?.error) && (
        <section className="card ov-sec">
          <div className="ov-head"><div className="row" style={{ gap: 12 }}><IconChip name="repair" tone="teal" size="sm" iconSize={17} /><h2>C3 registered repair models</h2></div>
            <Link to="/c3" className="small" style={{ fontWeight: 600 }}>Open C3 →</Link></div>
          {models.data?.error && <p className="small">{models.data.error}</p>}
          <div className="ov-models">
            {models.data?.models?.map((m) => (
              <div key={m.model_id} className="ov-model">
                <b className="mono">{m.model_id}</b>
                <span className="small muted">{m.dataset_id} · {m.algorithm || "algorithm not loaded"}</span>
                <span className={`pill tone-${m.ready ? "green" : "amber"}`} style={{ alignSelf: "flex-start" }}>{m.ready ? "artifacts available" : "missing artifacts"}</span>
              </div>))}
          </div>
        </section>)}
    </>
  );
}
