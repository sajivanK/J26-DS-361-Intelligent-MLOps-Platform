// Formatting helpers for the C2 page and report (Tharmika N).

// A run's state, as decided by the C2 engine (c2/src/history.py run_state).
export const STATE = {
  clean: { pill: "ok", label: "no harmful drift", head: "No harmful drift", tone: "ok" },
  harmless: { pill: "ok", label: "harmless drift", head: "Harmless drift", tone: "ok" },
  suspected: { pill: "watch", label: "suspected, not confirmed", head: "Possible root cause", tone: "warn" },
  confirmed: { pill: "warn", label: "root cause confirmed", head: "Root cause", tone: "warn" },
};

export const NICE = {
  age: "age", workclass: "type of employer", education_num: "education level", marital_status: "marital status",
  occupation: "occupation", relationship: "family role", race: "race", sex: "sex", capital_gain: "capital gain",
  capital_loss: "capital loss", hours_per_week: "hours per week", native_country: "country of origin",
};
export const nice = (f) => NICE[f] || String(f).replace(/_/g, " ");

export const sg = (v, d = 1) => (v == null ? "–" : (v < 0 ? "−" : "") + Math.abs(v).toFixed(d));
export const pct = (v, d = 0) => (v == null ? "–" : `${(v * 100).toFixed(d)}%`);
// A batch source for display: a file path is shortened to its file name.
export const shortSource = (src) => String(src || "").replace(/(?:[A-Za-z]:)?[\\/]?(?:[^\s\\/·]+[\\/])+([^\s\\/·]+)/g, "$1");
export const time = (iso) => (iso ? new Date(iso).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }) : "");
export const clock = (iso) => (iso ? new Date(iso).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) : "");

// No labels in production, so accuracy is never measured. What C2 measures is the share of the
// model's decisions that flip when the root feature is put back to normal (c2/src/drift_impact_free.py).
export const decOf = (rec) => {
  const d = rec?.evidence?.decision_change;
  return d == null ? null : Math.max(d, 0) * 100;
};
// Estimated accuracy loss: only from a harm predictor or a real C1; null otherwise.
export const harmOf = (rec) => {
  const ev = rec?.evidence || {};
  const h = ev.estimated_harm_pp ?? ev.c1_estimated_harm_pp;
  return h == null ? null : Math.max(h, 0);
};
export const c1PredOf = (rec) => {
  if (rec?.c1_source === "fake") return null;     // the fake C1 copies the true drop: not a prediction
  const p = rec?.c1?.predicted_harm;
  return p == null ? null : p * 100;           // C1 contract v1.1: a proportion
};
export const relianceOf = (rec, f) => {
  const ts = rec.temporal_shap, a = ts.mean_abs_shap_ref[f];
  return a ? ts.mean_abs_shap_now[f] / a : null;
};
export const runsAgreeing = (rec) => Math.round((rec.to_c3?.diagnosis_stability || 0) * 3);
export const topZ = (rec) => Object.entries(rec.temporal_shap.z).sort((a, b) => b[1] - a[1])[0];

// The page's plain-language answer for one run.
export function answer(rec) {
  const st = rec.state, c3 = rec.to_c3, root = c3.root_cause_feature, plain = rec.plain || {};
  const name = plain.root_cause || nice(root);
  const harm = harmOf(rec), dec = decOf(rec), rel = relianceOf(rec, root);
  const n = rec.n_rows || 1000, rows = Math.round(c3.slice_support * n);
  if (st === "clean") {
    const [zf, zv] = topZ(rec);
    return { name, harm, dec, rows, sentence: ["The model is reading this batch the way it normally does. The biggest change was in ",
      { b: nice(zf) }, `, at ${zv.toFixed(1)} times normal noise, below the alarm line of ${rec.temporal_shap.z_thresh.toFixed(2)}.`] };
  }
  if (st === "harmless") {
    return { name, harm, dec, rows, sentence: [{ b: name }, " changed, but putting it back to normal changes only ", { b: `${sg(dec)}%` },
      " of the model's decisions, below the 2% line. No repair needed."] };
  }
  return { name, harm, dec, rows, sentence: ["The model now leans on ", { b: name }, " about ", { b: `${rel?.toFixed(1)} times` },
    " more than normal, and the change flips about ", { b: `${sg(dec)}% of its decisions` },
    harm != null ? ` (estimated accuracy loss ${sg(harm)} points)` : "", ". The affected decisions sit in a group of about ",
    { b: `${rows.toLocaleString()} of ${n.toLocaleString()} records` }, "."] };
}

// What simpler tools would have said about the same run (from the run's own stored evidence).
export function baselines(rec) {
  const top3 = (obj) => Object.entries(obj).sort((a, b) => b[1] - a[1]).slice(0, 3).map((x) => x[0]);
  const raw = rec.evidence.signals_raw;
  return {
    drift: top3(Object.fromEntries(Object.entries(raw).map(([f, s]) => [f, s.D]))),
    snapshot: top3(rec.temporal_shap.z),
  };
}
