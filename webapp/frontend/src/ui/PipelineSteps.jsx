import { STEPS } from "../steps.js";
import { c1View, sg } from "./c1format.js";
import { SourceBadge } from "./parts.jsx";

// One-line summary of each component's output, used by the Drift Lab and the Overview.
// When you change your component's output, update its line here.
export function stepSummary(component, outputs) {
  const o = outputs || {};
  if (component === "c1" && o.c1) {
    const v = c1View(o.c1);
    return { text: `${v.det ? "drift" : "no drift"} · ${sg(v.harm)} pp · ${v.sev} · ${v.ext}`, sub: v.aff.join(", ") };
  }
  if (component === "c2" && o.c2_to_c3) return { text: `${o.c2_to_c3.root_cause_category}: ${o.c2_to_c3.root_cause_feature}` };
  if (component === "c3" && o.c3) return { text: `${o.c3.execution_status || o.c3.status || o.c3.repair_package_status} · ${o.c3.generator} · ${o.c3.n_synthetic_rows} rows`, sub: o.c3.reason };
  if (component === "c4" && o.c4) {
    const ev = o.c4.evaluation_status;          // C4 contract v1.1
    const sub = ev === "UPSTREAM_GATE_FAILED" ? "not evaluated: an upstream gate failed"
      : ev === "FORECAST_UNAVAILABLE" ? "candidate evaluated; live forecast not available" : null;
    return { text: o.c4.decision, sub };
  }
  return null;
}

const SKIPPED = { no_action: "not needed: no meaningful harmful drift", stopped: "not run: C1 → C2 only" };

// The four step cards for one pipeline run (the response of /api/pipeline/run).
// The badge shows what the component did IN THIS RUN: "fake" when it used its contract example
// (not loaded, an error, or a dataset it does not support), with the reason underneath.
export default function PipelineSteps({ run }) {
  const byName = Object.fromEntries((run.steps || []).map((s) => [s.component, s]));
  return (
    <div className="steps4">
      {STEPS.map((s) => {
        const st = byName[s.component];
        if (!st) {
          return (
            <div key={s.component} className="st skip">
              <span className="eyebrow">C{s.n}</span><b>{s.title}</b>
              <span className="small muted">{SKIPPED[run.status] || "not run"}</span>
            </div>
          );
        }
        const done = st.status === "ok" || st.status === "fallback";
        const sum = done ? stepSummary(s.component, run.outputs) : null;
        return (
          <div key={s.component} className="st">
            <div className="top"><span className="eyebrow">C{s.n}</span><SourceBadge source={st.source} /></div>
            <b>{s.title}</b>
            {st.status === "error"
              ? <span className="small" style={{ color: "var(--bad-fg)" }}>{st.error}</span>
              : <span className="small mono">{st.status === "running" ? "Running…" : sum?.text}</span>}
            {sum?.sub && <span className="small muted">{sum.sub}</span>}
            {st.note && <span className="small" style={{ color: "var(--warn-fg)" }}>{st.note}</span>}
            {done && (st.contract_ok
              ? <span className="badge ok">contract ok</span>
              : <span className="badge off" title={JSON.stringify(st.contract_errors)}>contract errors</span>)}
            {st.runtime_sec != null && <span className="small muted mono">{st.runtime_sec} s</span>}
          </div>
        );
      })}
    </div>
  );
}
