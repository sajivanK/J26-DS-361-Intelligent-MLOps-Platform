"""
Pipeline orchestrator: runs C1 -> C2 -> C3 -> C4 on one batch, passing everything produced so far
as `upstream`, and checks every output against its contract.
J26-DS-361 | Intelligent MLOps Platform | owner: Sajivan K

Repair gate (after C1): continue only when C1 reports detected drift, a severity other than "none"
and at least 1 percentage point of predicted harm (C1 contract v1.1 consumer_guidance). This keeps
clean batches whose severity is "low" only from sampling noise out of C2-C4. The same rule decides
whether a C1-only run raises a drift alert (runs.alert_state).

Honest real/fake per run (v0.3):
  A step's badge shows what happened IN THIS RUN, not only the component's general status. A component
  that falls back to its contract example for one run (e.g. C2 on IEEE-CIS, which it does not support)
  marks the output with "_source": "fake" (fakes.py does this) and the step shows "fake".
  If C2, C3 or C4 is not loaded or raises an error, the step is marked "fallback" with the reason and the
  contract example is used, so the later steps still run and the demo never stops half way.
  C1 has no fallback: without C1 there is nothing to diagnose, so the run fails.

Contract failures (v0.3): recorded on the step ("contract errors", red) and the run continues;
the run status becomes "completed_with_errors". Whether to stop instead is an open team decision.

stop_after (v0.3): "c2" runs C1 and C2 only (Drift Lab "Run C1 -> C2").

Live progress (C3 PR): one pipeline runs at a time (C2's temporal memory belongs to one batch at a
time), and a full pipeline publishes its state after every step to runs.save_latest, so the C3 page
can show "running" while a long repair is in progress. C1 -> C2 runs do not replace the latest run.
"""
import threading
import time
import uuid

from . import contracts, runs
from .components import PIPELINE, fakes

FALLBACKS = {"c2": fakes.fake_c2, "c3": fakes.fake_c3, "c4": fakes.fake_c4}
MIN_HARM_FOR_REPAIR = 0.01          # 1 pp, as a proportion like C1's predicted_harm

_PIPELINE_LOCK = threading.Lock()


def repair_gate(c1):
    # drift_detected is not required: interaction drift can be harmful while the classic
    # detectors stay quiet (C1 reports it through interaction_suspected and predicted harm).
    return (c1.get("harm_severity", "none") != "none"
            and c1.get("predicted_harm", 0) >= MIN_HARM_FOR_REPAIR)


def _fallback(comp, upstream, reason):
    """Contract example for C2-C4 so the pipeline can continue; None for C1."""
    fb = FALLBACKS.get(comp.name)
    if fb is None:
        return None
    try:
        out = fb(upstream)
    except Exception:  # noqa: BLE001  the example itself could not be built (e.g. no C1 output)
        return None
    out.pop("_source", None)
    return out


def run_pipeline(batch, context, stop_after=None):
    with _PIPELINE_LOCK:
        return _run_pipeline(batch, context, stop_after)


def _run_pipeline(batch, context, stop_after):
    run_id = uuid.uuid4().hex[:12]
    upstream = {"context": {**context, "run_id": run_id}, "component_sources": {}}
    steps, status, stopped_reason = [], "running", None
    t_all = time.time()

    def result():
        return {"run_id": run_id, "status": status, "stopped_reason": stopped_reason,
                "runtime_sec": round(time.time() - t_all, 3), "context": upstream["context"],
                "steps": [dict(s) for s in steps],
                "outputs": {k: v for k, v in upstream.items() if k not in ("context", "component_sources")},
                "component_sources": dict(upstream["component_sources"])}

    def publish():
        if stop_after is None:              # only full pipeline runs are shown as the latest run
            runs.save_latest(result())

    publish()
    for comp in PIPELINE:
        step = {"component": comp.name, "title": comp.title, "source": comp.source, "status": "running"}
        steps.append(step)
        publish()
        t0 = time.time()
        out, note = None, None
        if not comp.loaded:
            note = f"not loaded ({comp.error}); the contract example was used so the pipeline could continue"
            out = _fallback(comp, upstream, note)
        else:
            try:
                out = comp.run(batch, upstream)
            except Exception as e:  # noqa: BLE001
                note = (f"{type(e).__name__}: {e}. The contract example was used so the pipeline could continue"
                        if comp.name in FALLBACKS else f"{type(e).__name__}: {e}")
                out = _fallback(comp, upstream, note)
        step["runtime_sec"] = round(time.time() - t0, 3)

        if out is None:                       # C1 failed, or no example could be built: stop here
            step.update(status="error", error=note)
            status, stopped_reason = "failed", f"{comp.name}: {note}"
            break

        if note:                              # not loaded or raised: example used
            step.update(status="fallback", source="fake", note=note)
            status = "completed_with_errors"
        else:
            step["status"] = "ok"
        if out.pop("_source", None) == "fake":    # the component returned its example for this run
            step["source"] = "fake"
        if "_note" in out:
            step["note"] = out.pop("_note")

        problems = {}
        for name in comp.produces:
            if name not in out:
                problems[name] = [f"{name}: output missing from {comp.name}.run()"]
            else:
                errs = contracts.validate(name, out[name])
                if errs:
                    problems[name] = errs
        if problems:
            status = "completed_with_errors"
        upstream.update(out)
        upstream["component_sources"][comp.name] = step["source"]
        step.update(outputs=list(out), contract_ok=not problems, contract_errors=problems)
        publish()

        if comp.name == "c1":
            if not repair_gate(out["c1"]):
                status, stopped_reason = "no_action", "C1 did not establish meaningful harmful drift: no repair needed"
                break
            upstream["context"]["drift_event_id"] = out["c1"].get("evidence_version", run_id)
        if stop_after == comp.name:
            if status == "running":
                status = "stopped"
            stopped_reason = f"stopped after {comp.name.upper()} as requested"
            break

    if status == "running":
        status = "completed"
    final = result()
    if stop_after is None:
        runs.save_latest(final)
    return final
