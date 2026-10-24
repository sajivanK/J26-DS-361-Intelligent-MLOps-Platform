"""
Fake outputs for C2, C3 and C4, used until each member plugs in their real code.
J26-DS-361 | Intelligent MLOps Platform

Each fake starts from the member's OWN contract example (shared/contracts/) and only adapts the
fields that obviously depend on upstream results (e.g. C2's root-cause feature follows C1's
affected_features). Fake values are placeholders: never report them as research results.
Every fake carries "_source": "fake"; the orchestrator removes it and shows the step as fake for that
run, even when a real component returned the example (e.g. C2 on a dataset it does not support).
"""
from datetime import datetime, timezone

from .. import contracts


def _now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def root_cause_category(c1):
    """Rule from the C1 contract v1.1 consumer_guidance (0.5 cut-off is a convention)."""
    marginal = c1.get("marginal_extent", "none") != "none"
    inter = float(c1.get("interaction_suspected", 0.0)) >= 0.5
    if marginal and inter:
        return "combined"
    if marginal:
        return "marginal"
    if inter:
        return "interaction"
    return "unknown"


def fake_c2(upstream):
    c1 = upstream["c1"]
    feats = list(c1.get("affected_features") or [])
    to_c3, to_c4 = contracts.example("c2_to_c3"), contracts.example("c2_to_c4")
    feature = feats[0] if feats else to_c3["root_cause_feature"]
    common = {"root_cause_feature": feature,
              "root_cause_category": root_cause_category(c1),
              "affected_slice_rule": f"FAKE slice on {feature}",
              "affected_features_ranked": feats or to_c3["affected_features_ranked"]}
    to_c3.update(common)
    to_c3["affected_value_ranges"] = {}
    to_c4.update(common)
    to_c4.update({"diagnosis_version": "c2-fake", "diagnosis_timestamp": _now()})
    return {"c2_to_c3": to_c3, "c2_to_c4": to_c4, "_source": "fake"}


def fake_c3(upstream):
    out = contracts.example("c3")
    run_id = upstream["context"].get("run_id", "run")
    out.update({"repair_dataset_uri": f"fake://repairs/{run_id}.csv",
                "repair_version": "c3-fake", "repair_timestamp": _now()})
    return {"c3": out, "_source": "fake"}


def fake_c4(upstream):
    c1, c2, c3 = upstream["c1"], upstream["c2_to_c4"], upstream["c3"]
    sev = c1.get("harm_severity", "none")
    key = "escalate" if (sev in ("high", "critical") or c2.get("diagnosis_stability", 1) < 0.5) \
        else "human_review" if sev == "moderate" else "auto_deploy"
    out = contracts.example("c4", key)
    spec = contracts.fields("c4")
    out["pre_release_evidence"] = {k: v.get("example") for k, v in
                                   spec["pre_release_evidence"].get("fields", {}).items()}
    summary_spec = spec["upstream_evidence_summary"].get("fields", {})
    src = {"c1": c1, "c2": c2, "c3": c3}
    out["upstream_evidence_summary"] = {
        part: {k: src[part].get(k) for k in summary_spec.get(part, {}).get("fields", {})}
        for part in ("c1", "c2", "c3")}
    out["hard_gate_status"]["repair_approved"] = c3.get("repair_package_status") == "APPROVED"
    out.update({"audit_id": f"fake-{upstream['context'].get('run_id', '')}", "decision_timestamp": _now()})
    out["reason"] = "FAKE C4 decision (rule of thumb on C1 severity). " + out.get("reason", "")
    return {"c4": out, "_source": "fake"}
