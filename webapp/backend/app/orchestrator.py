"""
Pipeline orchestrator: runs C1 -> C2 -> C3 -> C4 on one batch, passing everything produced so far
as `upstream`, and checks every output against its contract.
J26-DS-361 | Intelligent MLOps Platform

Repair gate: diagnosis/repair proceeds only when C1 reports detected drift,
non-none severity and at least one percentage point of predicted harm.
"""
import time
import uuid
import threading

from . import contracts, runs
from .components import PIPELINE


def repair_gate(c1):
    return (c1.get("drift_detected") is True and c1.get("harm_severity", "none") != "none"
            and c1.get("predicted_harm", 0) >= 0.01)


_PIPELINE_LOCK = threading.Lock()


def run_pipeline(batch, context):
    # C2's temporal diagnosis and the latest-run view belong to one batch at a time.
    with _PIPELINE_LOCK:
        return _run_pipeline(batch, context)


def _run_pipeline(batch, context):
    run_id = uuid.uuid4().hex[:12]
    upstream = {"context": {**context, "run_id": run_id}, "component_sources": {}}
    steps, status, stopped_reason = [], "completed", None
    t_all = time.time()

    def publish():
        result = {"run_id": run_id, "status": status, "stopped_reason": stopped_reason,
                  "runtime_sec": round(time.time() - t_all, 3), "context": upstream["context"],
                  "steps": [dict(step) for step in steps],
                  "outputs": {k: v for k, v in upstream.items() if k not in ("context", "component_sources")},
                  "component_sources": dict(upstream["component_sources"])}
        runs.save_latest(result)
        return result

    status = "running"
    publish()

    for comp in PIPELINE:
        step = {"component": comp.name, "title": comp.title, "source": comp.source}
        if not comp.loaded:
            step.update(status="error", error=f"not loaded: {comp.error}")
            steps.append(step)
            status, stopped_reason = "failed", f"{comp.name} not loaded"
            break
        t0 = time.time()
        step["status"] = "running"
        steps.append(step)
        publish()
        try:
            out = comp.run(batch, upstream)
        except Exception as e:  # noqa: BLE001
            step.update(status="error", error=f"{type(e).__name__}: {e}", runtime_sec=round(time.time() - t0, 3))
            status, stopped_reason = "failed", f"{comp.name} raised an error"
            break
        problems = {}
        for name in comp.produces:
            if name not in out:
                problems[name] = [f"{name}: output missing from {comp.name}.run()"]
            else:
                errs = contracts.validate(name, out[name])
                if errs:
                    problems[name] = errs
        upstream.update(out)
        upstream["component_sources"][comp.name] = comp.source
        step.update(status="ok", runtime_sec=round(time.time() - t0, 3), outputs=list(out),
                    contract_ok=not problems, contract_errors=problems)
        publish()

        if problems:
            step["status"] = "error"
            status, stopped_reason = "failed", f"{comp.name} output contract failed"
            break

        if comp.name == "c1" and not repair_gate(out["c1"]):
            status, stopped_reason = "no_action", "C1 did not establish meaningful harmful drift: no repair needed"
            break

        if comp.name == "c1":
            upstream["context"]["drift_event_id"] = out["c1"].get("evidence_version", run_id)

    if status == "running":
        status = "completed"

    return publish()
