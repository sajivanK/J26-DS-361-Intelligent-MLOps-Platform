import {useEffect, useRef, useState} from "react";
import {Link} from "react-router-dom";
import "./C4LiveAssessment.css";

const numeric = value =>
  typeof value === "number" && Number.isFinite(value);

const display = value => {
  if (value == null) return "Unavailable";
  if (typeof value === "boolean") return value ? "Yes" : "No";
  if (Array.isArray(value)) return value.join(", ") || "None";
  return String(value);
};

const percent = value =>
  numeric(value) ? `${(value * 100).toFixed(2)}%` : "Unavailable";

const change = value =>
  numeric(value)
    ? `${value > 0 ? "+" : ""}${(value * 100).toFixed(3)} pp`
    : "Unavailable";

const seconds = value =>
  numeric(value) ? `${value.toFixed(3)} s` : "Unavailable";

function Fields({rows}) {
  return (
    <dl className="c4live-fields">
      {rows.map(([label, value]) => (
        <div key={label}>
          <dt>{label}</dt>
          <dd>{display(value)}</dd>
        </div>
      ))}
    </dl>
  );
}

async function getJSON(path, signal) {
  const response = await fetch(path, {signal});
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    throw new Error(
      typeof body?.detail === "string"
        ? body.detail
        : `Request failed (${response.status}).`
    );
  }
  return body;
}

function GainChart({evidence}) {
  const rows = [
    ["Global", evidence?.validation_global_f1_gain],
    ["Diagnosed slice", evidence?.validation_affected_slice_recovery],
    ["Complement", evidence?.validation_non_affected_margin],
  ];

  const available = rows.filter(([, value]) => numeric(value));
  if (!available.length) {
    return <p>No measured pre-release F1 gains were saved for this decision.</p>;
  }

  const maximum = Math.max(
    0.01, ...available.map(([, value]) => Math.abs(value))
  );

  return (
    <div className="c4live-gains">
      {rows.map(([name, value]) => (
        <div className="c4live-gain-row" key={name}>
          <span>{name}</span>
          <svg
            viewBox="0 0 400 34"
            role="img"
            aria-label={`${name}: ${change(value)}`}
          >
            <title>{`${name}: ${change(value)}`}</title>
            <rect x="0" y="8" width="400" height="18"
              rx="5" fill="#edf2f8" />
            <line x1="200" y1="3" x2="200" y2="31"
              stroke="#72839b" />
            {numeric(value) && (
              <rect
                x={value < 0 ? 200 - Math.abs(value) / maximum * 190 : 200}
                y="8"
                width={Math.abs(value) / maximum * 190}
                height="18"
                rx="4"
                fill={value < 0 ? "#dc654d" : "#149c88"}
              />
            )}
          </svg>
          <strong>{change(value)}</strong>
        </div>
      ))}
      <p className="small muted">
        Positive-class F1 change versus champion. These are observed
        changes on C3's approval holdout, not future forecasts.
      </p>
    </div>
  );
}

export default function C4LiveAssessment({health}) {
  const [runs, setRuns] = useState([]);
  const [selected, setSelected] = useState("");
  const [record, setRecord] = useState(null);
  const [listBusy, setListBusy] = useState(false);
  const [detailBusy, setDetailBusy] = useState(false);
  const [listError, setListError] = useState("");
  const [detailError, setDetailError] = useState("");
  const [refresh, setRefresh] = useState(0);
  const selection = useRef("");
  selection.current = selected;

  useEffect(() => {
    const controller = new AbortController();
    setListBusy(true);
    setListError("");

    getJSON("/api/runs", controller.signal)
      .then(body => {
        const entries = Array.isArray(body) ? body : body?.value;
        if (!Array.isArray(entries)) {
          throw new Error("Unexpected run-history response.");
        }

        const eligible = entries.filter(
          row => row.mode === "pipeline" && row.dataset === "uci"
        );
        setRuns(eligible);

        const previous = selection.current;
        setSelected(
          eligible.some(row => row.run_id === previous)
            ? previous
            : eligible[0]?.run_id || ""
        );
      })
      .catch(error => {
        if (error.name !== "AbortError") {
          setListError(error.message);
          setRuns([]);
          setSelected("");
          setRecord(null);
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) setListBusy(false);
      });

    return () => controller.abort();
  }, [refresh]);

  useEffect(() => {
    setRecord(null);
    setDetailError("");

    if (!selected) {
      setDetailBusy(false);
      return;
    }

    const controller = new AbortController();
    setDetailBusy(true);

    getJSON(`/api/runs/${encodeURIComponent(selected)}`, controller.signal)
      .then(body => {
        if (body?.run_id !== selected || !body.pipeline) {
          throw new Error("Saved run identity or pipeline result is missing.");
        }
        setRecord(body);
      })
      .catch(error => {
        if (error.name !== "AbortError") setDetailError(error.message);
      })
      .finally(() => {
        if (!controller.signal.aborted) setDetailBusy(false);
      });

    return () => controller.abort();
  }, [selected, refresh]);

  const pipeline = record?.pipeline;
  const outputs = pipeline?.outputs || {};
  const steps = Array.isArray(pipeline?.steps) ? pipeline.steps : [];
  const c4Step = steps.find(step => step.component === "c4");
  const result = outputs.c4;

  const genuineC4 = (
    pipeline?.component_sources?.c4 === "real" &&
    c4Step?.source === "real" &&
    c4Step?.status === "ok" &&
    c4Step?.contract_ok === true &&
    result &&
    ["HUMAN_REVIEW", "ESCALATE", "AUTO_DEPLOY"].includes(result.decision)
  );

  const upstream = result?.upstream_evidence_summary || {};
  const c1 = upstream.c1 || outputs.c1 || {};
  const c2 = upstream.c2 || outputs.c2_to_c4 || {};
  const c3 = upstream.c3 || outputs.c3 || {};
  const services = health?.data?.components || [];
  const gateStatus = result?.hard_gate_status || {};
  const decisionClass = result?.decision === "ESCALATE"
    ? "is-escalate" : "is-review";

  return (
    <div className="c4live">
      <section className="c4-card c4live-intro">
        <div className="c4-between">
          <div>
            <h2>Integrated Adult assessments</h2>
            <p>
              Actual C1 → C2 → C3 → C4 results saved by the platform.
              Opening a run does not retrain or reevaluate its candidate.
            </p>
          </div>
          <Link className="btn primary" to="/lab">Run a new pipeline</Link>
        </div>

        <div className="c4live-services">
          {["c1", "c2", "c3", "c4"].map(name => {
            const service = services.find(item => item.name === name);
            const ready = service?.loaded === true && service?.source === "real";
            return (
              <span key={name} className={ready ? "is-ready" : ""}>
                {name.toUpperCase()}: {ready ? "real · loaded" : "check status"}
              </span>
            );
          })}
        </div>
        <p className="small muted">
          Service availability is separate from each saved run's execution source.
        </p>
      </section>

      <section className="c4-card">
        <div className="c4-between">
          <h2>Select a saved pipeline run</h2>
          <button
            className="btn"
            disabled={listBusy || detailBusy}
            onClick={() => {
              setRecord(null);
              setRefresh(value => value + 1);
              health?.reload?.();
            }}
          >
            Refresh runs
          </button>
        </div>

        <label className="c4live-select">
          Adult pipeline history
          <select
            value={selected}
            disabled={listBusy}
            onChange={event => {
              setRecord(null);
              setSelected(event.target.value);
            }}
          >
            <option value="">Select a saved run</option>
            {runs.map(run => (
              <option key={run.run_id} value={run.run_id}>
                {run.run_id} · {run.saved_at} · {run.pipeline_status}
              </option>
            ))}
          </select>
        </label>

        {listError && <p role="alert">{listError}</p>}
        {detailError && <p role="alert">{detailError}</p>}
        {(listBusy || detailBusy) && <p role="status">Loading saved evidence…</p>}
        {!listBusy && !runs.length && !listError && (
          <p>No saved Adult pipeline runs. Start one in Drift Lab.</p>
        )}
      </section>

      {record && (
        <section className="c4-card">
          <div className="c4-between">
            <h2>Execution provenance</h2>
            <Link className="btn" to={`/runs/${encodeURIComponent(record.run_id)}`}>
              Open original run
            </Link>
          </div>
          <Fields rows={[
            ["Saved-history ID", record.run_id],
            ["Pipeline execution ID", pipeline.run_id],
            ["Batch ID", pipeline.context?.batch_id],
            ["Pipeline status", pipeline.status],
            ["Pipeline runtime", seconds(pipeline.runtime_sec)],
            ["C4 runtime", seconds(c4Step?.runtime_sec)],
          ]} />
          <div className="c4live-services">
            {steps.map(step => (
              <span
                key={step.component}
                className={step.source === "real" && step.contract_ok === true
                  ? "is-ready" : ""}
              >
                {step.component.toUpperCase()} · {step.source} ·
                {" "}{step.contract_ok === true ? "contract passed" : "check contract"}
              </span>
            ))}
          </div>
          <p className="small muted">
            Real component execution on a Drift Lab batch. The batch contains
            injected drift; this is not evidence of a production deployment.
          </p>
        </section>
      )}

      {record && !genuineC4 && (
        <section className="c4-card">
          <h2>No verified real C4 result for this run</h2>
          <p>
            This run may predate the live adapter or contain a fallback.
            Its example output is not presented as an actual C4 decision.
          </p>
          <Link className="btn" to={`/runs/${encodeURIComponent(record.run_id)}`}>
            Inspect original run
          </Link>
        </section>
      )}

      {record && genuineC4 && (
        <>
          <section className={`c4live-verdict ${decisionClass}`}>
            <div className="c4-between">
              <h2>{result.decision.replaceAll("_", " ")}</h2>
              <span className="c4-source">Real C4 execution · saved result</span>
            </div>
            <p><strong>Evaluation:</strong> {result.evaluation_status}</p>
            <p>{result.reason}</p>
            <p><strong>Required action:</strong> {result.required_action}</p>
          </section>

          <section className="c4-card">
            <h2>Candidate and assessment identity</h2>
            <Fields rows={[
              ["Champion", result.champion_model_id],
              ["Candidate", result.candidate_model_id],
              ["Repair strategy", c3.repair_strategy],
              ["Decision timestamp", result.decision_timestamp],
              ["Audit ID", result.audit_id],
              ["Policy version", result.policy_version],
            ]} />
          </section>

          <section className="c4live-upstream">
            <article className="c4-card c4live-blue">
              <h3>C1 · Drift evidence</h3>
              <Fields rows={[
                ["Predicted accuracy harm", change(c1.predicted_harm)],
                ["Severity", c1.harm_severity],
                ["Affected features", c1.affected_features],
                ["Batch rows", c1.batch_size],
              ]} />
            </article>
            <article className="c4-card c4live-purple">
              <h3>C2 · Diagnosis</h3>
              <Fields rows={[
                ["Root cause", c2.root_cause_feature],
                ["Diagnosed slice", c2.affected_slice_rule],
                ["Slice support", percent(c2.slice_support)],
                ["Diagnosis stability", display(c2.diagnosis_stability)],
              ]} />
            </article>
            <article className="c4-card c4live-teal">
              <h3>C3 · Repair package</h3>
              <Fields rows={[
                ["Package status", c3.repair_package_status],
                ["Repair strategy", c3.repair_strategy],
                ["Generator", c3.generator],
                ["Synthetic rows", c3.n_synthetic_rows],
                ["Final package privacy status", c3.privacy_status],
              ]} />
              <p className="small muted">
                NONE with zero synthetic rows can represent approved authentic
                adaptation. Synthetic fidelity and coverage are not applicable
                to that strategy.
              </p>
            </article>
          </section>

          <section className="c4-card">
            <h2>Observed pre-release evidence</h2>
            <GainChart evidence={result.pre_release_evidence} />
          </section>

          <section className="c4-card">
            <h2>Mandatory checks</h2>
            {Object.entries(gateStatus).map(([name, value]) => {
              const status = value === true || value === "PASS" ? "PASS"
                : value === false || value === "FAIL" ? "FAIL" : "UNKNOWN";
              return (
                <div className="c4-gate-label" key={name}>
                  <span>{name.replaceAll("_", " ")}</span>
                  <strong className={status === "PASS" ? "c4-pass"
                    : status === "FAIL" ? "c4-fail" : ""}>
                    {status}
                  </strong>
                </div>
              );
            })}
            <p className="small muted">
              Rollback readiness here means champion recovery artifact
              availability. Serving rollback was not tested.
            </p>
          </section>

          <section className="c4-card c4live-future">
            <h2>Future forecasting status</h2>
            <p>
              {result.evaluation_status === "UPSTREAM_GATE_FAILED"
                ? "An upstream gate stopped evaluation. Forecasting was not run."
                : result.evaluation_status === "FORECAST_UNAVAILABLE"
                ? "The candidate was evaluated, but applicable live N1 forecasts and N2 intervals were not computed."
                : "Inspect the saved forecast and calibration details."}
            </p>
            <Fields rows={[
              ["Forecast model version", result.forecast_model_version],
              ["Conformal configuration", result.conformal_config_version],
            ]} />
            <p className="small muted">
              Empty forecasts are unavailable measurements, not zero risk
              or zero performance change. No deployment is executed by this viewer.
            </p>
          </section>
        </>
      )}
    </div>
  );
}