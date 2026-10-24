import {useEffect, useRef, useState} from "react";

const targets = [
  ["future_global_margin", "Global", "#2563eb"],
  ["future_slice_margin", "Affected slice", "#9333ea"],
  ["future_nonaffected_margin", "Complement", "#0d9488"],
];
const pp = value => `${value >= 0 ? "+" : ""}${(value * 100).toFixed(3)} pp`;

function MarginChart({target, label, color, result}) {
  const dimension = result.safety.dimensions[target];
  const median = result.forecasts[target + "__median"];
  const {lower, upper, boundary, status} = dimension;
  const minimum = Math.min(lower, median, boundary, 0);
  const maximum = Math.max(upper, median, boundary, 0);
  const padding = Math.max((maximum - minimum) * 0.18, 0.001);
  const lo = minimum - padding;
  const hi = maximum + padding;
  const x = value => 32 + (value - lo) / (hi - lo) * 456;

  return (
    <article style={{
      background: "#f8fafc", border: "1px solid #dbe4ef",
      borderTop: `4px solid ${color}`, borderRadius: 12, padding: 18,
    }}>
      <div className="c4-between">
        <h3>{label}</h3>
        <strong style={{color: status === "SAFE" ? "#047857" : "#a16207"}}>
          {status}
        </strong>
      </div>
      <p>Forecast: <strong>{pp(median)}</strong></p>
      <svg viewBox="0 0 520 110" role="img"
        aria-label={`${label}: forecast ${pp(median)}, interval ${pp(lower)} to ${pp(upper)}`}>
        <line x1="32" x2="488" y1="55" y2="55" stroke="#cbd5e1" />
        <line x1={x(boundary)} x2={x(boundary)} y1="18" y2="82"
          stroke="#64748b" strokeDasharray="5 4">
          <title>Safety boundary: {pp(boundary)}</title>
        </line>
        <line x1={x(lower)} x2={x(upper)} y1="55" y2="55"
          stroke={color} strokeWidth="12" strokeLinecap="round">
          <title>Transferred interval: {pp(lower)} to {pp(upper)}</title>
        </line>
        <circle cx={x(median)} cy="55" r="9" fill={color}
          stroke="white" strokeWidth="3">
          <title>Forecast median: {pp(median)}</title>
        </circle>
        <text x="32" y="103" fontSize="12" fill="#475569">{pp(lo)}</text>
        <text x="488" y="103" textAnchor="end" fontSize="12" fill="#475569">{pp(hi)}</text>
      </svg>
      <p style={{fontSize: 13}}>
        Interval: {pp(lower)} to {pp(upper)}<br />
        Safety boundary: {pp(boundary)}
      </p>
    </article>
  );
}

export default function C4SandboxPanel({record}) {
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [elapsed, setElapsed] = useState(null);
  const request = useRef(null);

  useEffect(() => {
    request.current?.abort();
    setResult(null);
    setError("");
    setBusy(false);
    setElapsed(null);
    return () => request.current?.abort();
  }, [record]);

  const pipeline = record?.pipeline;
  const outputs = pipeline?.outputs || {};
  const genuine = ["c1", "c2", "c3"].every(
    name => pipeline?.component_sources?.[name] === "real"
      && pipeline?.steps?.some(step =>
        step.component === name && step.source === "real"
        && step.status === "ok" && step.contract_ok === true)
  );
  const ready = genuine && outputs.c1 && outputs.c2_to_c4 && outputs.c3;

  async function evaluate() {
    request.current?.abort();
    const controller = new AbortController();
    request.current = controller;
    setBusy(true);
    setError("");
    setResult(null);
    setElapsed(null);
    const started = performance.now();

    try {
      const response = await fetch("/api/c4/sandbox/evaluate", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        signal: controller.signal,
        body: JSON.stringify({
          context: pipeline.context,
          c1: outputs.c1,
          c2_to_c4: outputs.c2_to_c4,
          c3: outputs.c3,
        }),
      });
      const body = await response.json();
      if (!response.ok) {
        throw new Error(typeof body.detail === "string"
          ? body.detail : `Evaluation failed (${response.status}).`);
      }
      if (body.scope !== "experimental_transfer"
          || body.production_release_authorized !== false) {
        throw new Error("Unexpected sandbox response.");
      }
      if (body.N1_executed &&
          body.pipeline_run_id !== pipeline.context?.run_id) {
        throw new Error("Assessment belongs to another execution.");
      }
      if (!controller.signal.aborted) {
        setResult(body);
        setElapsed((performance.now() - started) / 1000);
      }
    } catch (failure) {
      if (!controller.signal.aborted) setError(failure.message);
    } finally {
      if (!controller.signal.aborted) setBusy(false);
    }
  }

  if (!record) return null;

  return (
    <section className="c4-card" style={{borderTop: "4px solid #2563eb"}}>
      <div className="c4-between">
        <div>
          <h2>Fresh C4 assessment</h2>
          <p>Evaluate this run's actual C1, C2 and C3 package here.</p>
        </div>
        <button className="btn primary" disabled={!ready || busy}
          onClick={evaluate}>
          {busy ? "Evaluating C4?" : "Evaluate C4"}
        </button>
      </div>

      <p className="small muted">Experimental assessment</p>
      {!ready && <p>Choose a run with verified real C1, C2 and C3 outputs.</p>}
      {error && <p role="alert">{error}</p>}
      {busy && <p role="status">Verifying artifacts and computing assessment?</p>}

      {result && (
        <>
          <div style={{
            background: "#fffbeb", border: "1px solid #fde68a",
            borderRadius: 12, padding: 18, margin: "18px 0",
          }}>
            <h2>{result.experimental_decision.replaceAll("_", " ")}</h2>
            <p>
              {result.N1_executed
                ? "Fresh forecast and transferred interval calculation completed."
                : result.pipeline_assessment.reason}
            </p>
            <p>Request time: {elapsed?.toFixed(3)} seconds</p>
            {result.N1_executed && <p>
              Package: {result.package_id}<br />
              Execution: {result.pipeline_run_id}
            </p>}
          </div>

          {result.N1_executed && (
            <>
              <div style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fit, minmax(230px, 1fr))",
                gap: 16,
              }}>
                {targets.map(([target, label, color]) =>
                  <MarginChart key={target} target={target}
                    label={label} color={color} result={result} />
                )}
              </div>

              <div className="c4-columns" style={{marginTop: 18}}>
                <article className="c4-card">
                  <h3>N1 - Forecasting</h3>
                  <p>Global: Ridge. Slice and complement: Quantile LightGBM.</p>
                  <p>75 features assembled from the selected upstream package.</p>
                </article>
                <article className="c4-card">
                  <h3>N2 - Transferred uncertainty</h3>
                  <p>Frozen benchmark adjustment: {pp(result.transferred_adjustment)}.</p>
                  <p>Coverage on authentic live repairs remains unvalidated.</p>
                </article>
                <article className="c4-card">
                  <h3>N3 - Population safety</h3>
                  <p>{targets.map(([target, label]) =>
                    `${label}: ${result.safety.dimensions[target].status}`
                  ).join(" | ")}</p>
                  <p>Benchmark population definitions retained.</p>
                </article>
                <article className="c4-card">
                  <h3>N4 - Governance</h3>
                  <p>Decision: {result.experimental_decision}</p>
                  <p>Policy: {result.selected_policy.policy_id}</p>
                  <p>{result.governance.reason.replaceAll("_", " ")}</p>
                </article>
              </div>

              <details style={{marginTop: 18}}>
                <summary>Input substitutions and technical evidence</summary>
                <p>Original C3 null values remain unchanged.</p>
                <pre style={{whiteSpace: "pre-wrap", overflowWrap: "anywhere"}}>
                  {JSON.stringify(result.substitutions, null, 2)}
                </pre>
                <p>{result.note}</p>
              </details>
            </>
          )}
        </>
      )}
    </section>
  );
}
