"""
Pipeline orchestrator: runs C1 -> C2 -> C3 -> C4 on one batch, passing everything produced so far
as `upstream`, and checks every output against its contract.
J26-DS-361 | Intelligent MLOps Platform

Repair gate (after C1): continue only if C1 harm_severity != "none" (C1 contract v1.1
consumer_guidance). OPEN TEAM DECISION: a clean batch can get harm_severity "low" from sampling noise
(drift_detected False, interval includes 0); the gate rule may be refined after the team discussion.
"""
import time
import uuid

from . import contracts
from .components import PIPELINE


def repair_gate(c1):
    return c1.get("harm_severity", "none") != "none"


def run_pipeline(batch, context):
    run_id = uuid.uuid4().hex[:12]
    upstream = {"context": {**context, "run_id": run_id}}
    steps, status, stopped_reason = [], "completed", None
    t_all = time.time()

    for comp in PIPELINE:
        step = {"component": comp.name, "title": comp.title, "source": comp.source}
        if not comp.loaded:
            step.update(status="error", error=f"not loaded: {comp.error}")
            steps.append(step)
            status, stopped_reason = "failed", f"{comp.name} not loaded"
            break
        t0 = time.time()
        try:
            out = comp.run(batch, upstream)
        except Exception as e:  # noqa: BLE001
            step.update(status="error", error=f"{type(e).__name__}: {e}", runtime_sec=round(time.time() - t0, 3))
            steps.append(step)
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
        step.update(status="ok", runtime_sec=round(time.time() - t0, 3), outputs=list(out),
                    contract_ok=not problems, contract_errors=problems)
        steps.append(step)

        if comp.name == "c1" and not repair_gate(out["c1"]):
            status, stopped_reason = "no_action", "C1 harm_severity is 'none': nothing to repair"
            break

    return {"run_id": run_id, "status": status, "stopped_reason": stopped_reason,
            "runtime_sec": round(time.time() - t_all, 3), "context": upstream["context"],
            "steps": steps, "outputs": {k: v for k, v in upstream.items() if k != "context"}}
