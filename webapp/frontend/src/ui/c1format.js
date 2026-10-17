// Turning a C1 output (contract v1.1) into the words and numbers the pages show.
// Shared by the C1 page, the Drift Lab and the Overview.

// signed number with a real minus sign: -1.5 -> "−1.50"
export const sg = (v, d = 2) => (v == null || Number.isNaN(v) ? "-" : (v < 0 ? "−" : "") + Math.abs(v).toFixed(d));

export const MIN_DETECTORS_FOR_ALARM = 3;   // a classic monitor alarms when 3 or more of the 6 detectors fire

// C1 output -> flat numbers in percentage points (pp)
export function c1View(o) {
  if (!o) return null;
  return {
    harm: o.predicted_harm * 100,
    lo: o.harm_interval[0] * 100,
    hi: o.harm_interval[1] * 100,
    sev: o.harm_severity,
    ext: o.marginal_extent,
    pint: o.interaction_suspected,
    aff: o.affected_features || [],
    fired: o.detectors_fired_list || [],
    det: o.drift_detected,
    bs: o.batch_size,
    metric: o.harm_metric === "auc_drop" ? "AUC" : "accuracy",
    warnings: o.warnings || [],
  };
}

export const isClassicAlarm = (v) => v.fired.length >= MIN_DETECTORS_FOR_ALARM;
export const isC1Alert = (v) => v.sev !== "none";

// [title, sentence, "ok"|"warn"]
export function verdict(v, nextStep = "Passed to C2 for diagnosis.") {
  if (!v.det && v.sev === "none") return ["No drift", "Nothing needs attention.", "ok"];
  if (v.sev === "none") return ["Drift found, low priority", "Real change, but the predicted cost is under 1 pp.", "ok"];
  return [`Drift found · severity ${v.sev}`, `Predicted cost about ${v.harm.toFixed(1)} pp. ${nextStep}`, "warn"];
}

export function driftKind(v) {
  if (v.ext === "none") return v.pint >= 0.5 ? "possible interaction" : "none";
  return v.ext === "single" ? "one feature" : "several features";
}

export function classicText(v) {
  const n = v.fired.length;
  if (isClassicAlarm(v)) return `ALARM: ${n} of 6 detectors fired`;
  return n ? `${n} detector${n > 1 ? "s" : ""} fired, no alarm` : "Nothing";
}
