import { STEPS } from "../steps.js";
import { c1View, sg } from "./c1format.js";
import { Icon } from "../layout/icons.jsx";
import { SourceBadge } from "./parts.jsx";
import { stepTile } from "./runTiles.jsx";

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

// The four step cards for one pipeline run (v0.4: tinted by result, joined by arrows).
// The badge shows what the component did IN THIS RUN: "fake" when it used its contract example
// (not loaded, an error, or a dataset it does not support), with the reason underneath.
// grade: { c2_ok, true_cause } from the run summary; charts: optional { c1, c2, c3, c4 } nodes.
export default function PipelineSteps({ run, grade = {}, charts = {} }) {
  return (
    <div className="flow4">
      {STEPS.map((s, i) => {
        const t = stepTile(s.component, run, grade);
        const st = (run.steps || []).find((x) => x.component === s.component);
        const done = st && (st.status === "ok" || st.status === "fallback");
        return [
          i > 0 && <span key={"a" + i} className="flow-arrow" aria-hidden="true"><Icon name="arrow" size={16} className="" strokeWidth={2.4} /></span>,
          <div key={s.component} className={`ttile tone-${t.tone} ${t.skipped ? "skipped" : ""} ${t.source === "fake" ? "dashed" : ""}`}>
            <div className="row" style={{ justifyContent: "space-between", flexWrap: "nowrap" }}>
              <span className="row" style={{ gap: 8, flexWrap: "nowrap" }}><Icon name={t.icon} size={17} className="" strokeWidth={2.4} style={{ color: "var(--ti)" }} />
                <span className="eyebrow" style={{ color: "inherit" }}>C{s.n} · {s.label}</span></span>
              {t.source && <SourceBadge source={t.source} />}
            </div>
            <span className={`tt ${t.mono ? "mono" : ""}`}>{t.title}</span>
            {t.text && <span className="ts">{t.text}</span>}
            {t.note && t.source === "fake" && t.text !== t.note && <span className="ts">{t.note}</span>}
            {charts[s.component] && <div style={{ marginTop: "auto" }}>{charts[s.component]}</div>}
            {st && (
              <span className="mono ts" style={{ marginTop: charts[s.component] ? 0 : "auto", opacity: .85 }}>
                {done ? (st.contract_ok ? "contract ok" : <span title={JSON.stringify(st.contract_errors)}>contract errors</span>) : st.status}
                {st.runtime_sec != null ? ` · ${st.runtime_sec} s` : ""}</span>)}
          </div>,
        ];
      })}
    </div>
  );
}
