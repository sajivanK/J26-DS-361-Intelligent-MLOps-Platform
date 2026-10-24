// One tinted tile per pipeline step (v0.4): tone, icon, title and one line, from a pipeline run
// (/api/pipeline/latest or a Drift Lab run). Used by the Overview and the Drift Lab.
import { c1View, sg } from "./c1format.js";

const DECISION_TONE = { AUTO_DEPLOY: "green", HUMAN_REVIEW: "amber", ESCALATE: "red" };

export function stepTile(component, run, extra = {}) {
  const step = (run?.steps || []).find((s) => s.component === component);
  const o = run?.outputs || {};
  const skipped = run?.status === "no_action" ? "not needed: no meaningful harmful drift"
    : run?.status === "stopped" ? "not run: C1 → C2 only" : "not run";
  if (!step) return { tone: "grey", icon: "minus", title: "Not run", text: skipped, source: null, skipped: true };
  const base = { source: step.source, runtime: step.runtime_sec, contractOk: step.contract_ok, note: step.note };
  if (step.status === "error") return { ...base, tone: "red", icon: "warn", title: "Error", text: step.error || "the step failed" };
  if (step.status === "running") return { ...base, tone: "teal", icon: "clock", title: "Running…", text: "" };

  if (component === "c1") {
    const v = c1View(o.c1);
    if (!v) return { ...base, tone: "grey", icon: "minus", title: "No output", text: "" };
    const alert = v.sev !== "none";
    return { ...base, tone: alert ? "red" : "green", icon: alert ? "warn" : "check",
      title: alert ? "Harmful drift" : v.det ? "Drift, harmless" : "No drift",
      text: `${sg(v.harm)} pp · ${v.sev}${v.aff.length ? " · " + v.aff.join(", ") : ""}` };
  }
  if (component === "c2") {
    const c = o.c2_to_c3;
    if (!c) return { ...base, tone: "grey", icon: "minus", title: "No output", text: "" };
    const ok = extra.c2_ok;
    return { ...base, tone: step.source === "fake" ? "purple" : ok === false ? "amber" : "green",
      icon: step.source === "fake" ? "clock" : ok === false ? "warn" : "check", mono: true,
      title: c.root_cause_feature,
      text: `${c.root_cause_category}${ok === true ? " · matches answer key ✓" : ok === false ? ` · answer key: ${extra.true_cause || "?"}` : ""}` };
  }
  if (component === "c3") {
    const c = o.c3;
    if (!c) return { ...base, tone: "grey", icon: "minus", title: "No output", text: "" };
    if (step.source === "fake") return { ...base, tone: "purple", icon: "clock", title: "Fallback", text: step.note || "contract example used" };
    const approved = c.repair_package_status === "APPROVED";
    const strat = c.repair_strategy === "AUTHENTIC_WEIGHTED_ADAPTATION" ? "real labelled rows only"
      : c.generator && c.generator !== "NONE" ? `${c.generator} · ${c.n_synthetic_rows} rows` : "";
    return { ...base, tone: approved ? "green" : "grey", icon: approved ? "check" : "minus",
      title: approved ? "Approved" : (c.execution_status || "").startsWith("BLOCKED") ? "Blocked" : "Rejected",
      text: [strat, c.reason || step.note].filter(Boolean).join(" · ") || (approved ? "repair package ready for C4" : "no repair released") };
  }
  const c = o.c4;
  if (!c) return { ...base, tone: "grey", icon: "minus", title: "No output", text: "" };
  if (step.source === "fake") return { ...base, tone: "purple", icon: "clock", title: "Not connected", text: step.note || "contract example used" };
  return { ...base, tone: DECISION_TONE[c.decision] || "grey", icon: c.decision === "AUTO_DEPLOY" ? "check" : c.decision === "ESCALATE" ? "warn" : "clock",
    title: c.decision.replace("_", " ").toLowerCase().replace(/^./, (x) => x.toUpperCase()),
    text: c.evaluation_status === "UPSTREAM_GATE_FAILED" ? "not evaluated: an upstream gate failed" : (c.reason || "") };
}
