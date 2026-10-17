"""
C2 Diagnosis Engine: full pipeline and the two contract outputs.
J26-DS-361 | C2 XAI Root Cause Diagnosis Engine | Tharmika N (IT22247940)

Pipeline per monitoring cycle:
    Novelty 1  TemporalSHAPMonitor.update()   explanation evidence over cycles
    Novelty 2  DriftImpactScorer.score()      fused root cause + estimated harm
    Novelty 3  HierarchicalLocalizer.localize() subgroup rule for C3

Outputs (contracts v1.1, shared/contracts/):
    to_c3   Repair Specification   c2_output_to_c3_schema.json
    to_c4   Diagnostic Report      c2_output_to_c4_schema.json

Usage:
    engine = C2DiagnosisEngine()                 # loads shared artifacts
    engine.warmup(clean_batches)                 # first clean cycles after deployment
    result = engine.diagnose(batch, c1_output)   # once per monitoring cycle
    result["to_c3"], result["to_c4"], result["details"]
"""
import json
import uuid
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from c2.src.config import CONTRACTS_DIR, load_artifacts
from c2.src.drift_impact import DriftImpactScorer
from c2.src.localization import HierarchicalLocalizer
from c2.src.temporal_shap import TemporalSHAPMonitor

C3_SCHEMA = CONTRACTS_DIR / "c2_output_to_c3_schema.json"
C4_SCHEMA = CONTRACTS_DIR / "c2_output_to_c4_schema.json"


# ── ROOT CAUSE CATEGORY ────────────────────────────────────────────────────
def root_cause_category(c1_output, dis_result, shap_result, use_c1=True):
    """
    Contract v1.1: there is no C1 drift_type any more. C2 derives the category
    from its own evidence and, optionally, C1's marginal_extent and
    interaction_suspected (c1_output_schema.json consumer_guidance):

      marginal evidence    some feature's data distribution changed significantly
                           (normalised drift signal >= 0.5), or C1 reports a
                           marginal extent
      interaction evidence the ROOT feature shows a behaviour shift (its SHAP
                           attribution for the same values changed, z above the
                           Novelty 1 threshold and larger than its distribution
                           change) while its own data did not change, or C1
                           interaction_suspected >= 0.5

    Checked on the scenario suite: C2 evidence alone labels 92.2% of streams
    correctly; with C1 hints 98.0% (the fake C1 is built from the answer key,
    so the C2-only figure is the honest one until real C1 output is connected).
    """
    norm = dis_result["signals_normalised"]
    root = dis_result["root_cause_feature"]
    zb, zd = shap_result["z_behaviour"][root], shap_result["z_distribution"][root]

    marginal = max(v["D"] for v in norm.values()) >= 0.5
    interaction = (zb > shap_result["z_thresh"] and zb > 1.5 * max(zd, 0)
                   and norm[root]["D"] < 0.5)
    if use_c1:
        marginal = marginal or c1_output.get("marginal_extent", "none") != "none"
        interaction = interaction or float(c1_output.get("interaction_suspected", 0)) >= 0.5
    if marginal and interaction:
        return "combined"
    if marginal:
        return "marginal"
    if interaction:
        return "interaction"
    return "unknown"


def _fallback_rule(cur_df, ref_df, root, feature_types):
    """Rule used when no concentrated drift harm is found: where the root shifted."""
    if feature_types.get(root) == "categorical":
        cur = cur_df[root].value_counts(normalize=True)
        ref = ref_df[root].value_counts(normalize=True)
        gain = cur.sub(ref, fill_value=0)
        c = int(gain.idxmax())
        return f"{root} == {c}", {root: [c]}
    med = float(ref_df[root].median())
    if cur_df[root].mean() >= ref_df[root].mean():
        return f"{root} > {med:.4g}", {root: [med, float(cur_df[root].max())]}
    return f"{root} <= {med:.4g}", {root: [float(cur_df[root].min()), med]}


def _json_safe(v):
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, float) and not np.isfinite(v):
        return None
    if isinstance(v, dict):
        return {k: _json_safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_json_safe(x) for x in v]
    return v


# ── ENGINE ─────────────────────────────────────────────────────────────────
class C2DiagnosisEngine:

    def __init__(self, artifacts=None, window=3):
        self.a = artifacts or load_artifacts()
        a = self.a
        self.monitor = TemporalSHAPMonitor(a["model"], a["ref_df"], a["feature_types"],
                                           window=window)
        self.scorer = DriftImpactScorer(a["model"], a["ref_df"], a["feature_types"],
                                        self.monitor.z_thresh)
        self.localizer = HierarchicalLocalizer(a["model"], a["ref_df"], a["feature_types"])
        self.window = window
        self.root_history = []
        self.n_diagnoses = 0

    def warmup(self, clean_batches):
        """Feed the first clean production cycles: SHAP history + performance noise band."""
        self.monitor.reset()
        self.root_history = []
        for b in clean_batches:
            self.monitor.update(b)
        self.scorer.calibrate_performance(clean_batches)
        return self

    def _stability(self, root):
        self.root_history.append(root)
        recent = self.root_history[-self.window:]
        stability = sum(r == root for r in recent) / len(recent)
        run = 0
        for r in reversed(self.root_history):
            if r != root:
                break
            run += 1
        return round(stability, 3), run

    def diagnose(self, cur_df, c1_output):
        if self.scorer.mu_P is None:
            raise RuntimeError("call warmup(clean_batches) before diagnose()")
        a = self.a
        shap_r = self.monitor.update(cur_df)
        dis = self.scorer.score(cur_df, shap_r)
        ranked = list(dis["drift_impact_scores"])
        root = ranked[0]
        loc = self.localizer.localize(cur_df, root, ranked)

        rule = loc["affected_slice_rule"]
        ranges = loc["affected_value_ranges"]
        support = loc["slice_support"]
        if not rule:
            rule, ranges = _fallback_rule(cur_df, a["ref_df"], root, a["feature_types"])
            support = round(float(len(cur_df.query(rule)) / len(cur_df)), 4)
        ranges = {f: [None if (isinstance(x, float) and not np.isfinite(x)) else x
                      for x in v] for f, v in ranges.items()}

        stability, run = self._stability(root)
        category = root_cause_category(c1_output, dis, shap_r)
        self.last_evidence = {"shap": shap_r, "dis": dis, "localization": loc}
        self.n_diagnoses += 1
        version = f"c2-{self.n_diagnoses:03d}-{uuid.uuid4().hex[:6]}"
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        to_c3 = {
            "root_cause_feature": root,
            "root_cause_category": category,
            "affected_slice_rule": rule,
            "affected_features_ranked": ranked[:3],
            "affected_value_ranges": ranges,
            "slice_support": support,
            "diagnosis_stability": stability,
            "drift_impact_score": dis["top_score"],
            "temporal_window": run,
        }
        to_c4 = {
            "root_cause_category": category,
            "root_cause_feature": root,
            "affected_slice_rule": rule,
            "slice_support": support,
            "diagnosis_stability": stability,
            "affected_features_ranked": ranked[:3],
            "diagnosis_version": version,
            "diagnosis_timestamp": now,
        }
        details = {
            "c1_evidence_version": c1_output.get("evidence_version"),
            "estimated_harm_pp": dis["estimated_harm_pp"],
            "is_malignant": dis["is_malignant"],
            "explanation_alarm": shap_r["alarm"],
            "shift_pattern": shap_r["shift_pattern"],
            "localization_levels": loc["levels_used"],
            "localization_note": loc.get("note", ""),
            "drift_impact_scores": dis["drift_impact_scores"],
        }
        return {"to_c3": _json_safe(to_c3), "to_c4": _json_safe(to_c4),
                "details": _json_safe(details)}


# ── CONTRACT VALIDATION ────────────────────────────────────────────────────
_TYPES = {
    "string": str, "float": (int, float), "integer": int,
    "array[string]": list, "object": dict, "boolean": bool,
}


def validate_output(output, schema_path, ref_df=None):
    """
    Check an output against a team contract: every required field present, no
    unknown fields, types and allowed values correct, 0-1 ranges respected, and
    the slice rule runs as a pandas query on the reference data (C3 applies it there).
    Returns a list of problems (empty list = valid).
    """
    with open(schema_path) as f:
        fields = json.load(f)["fields"]
    problems = []
    for name, spec in fields.items():
        if spec.get("required") and name not in output:
            problems.append(f"missing required field '{name}'")
    for name, value in output.items():
        if name not in fields:
            problems.append(f"unknown field '{name}'")
            continue
        spec = fields[name]
        expected = _TYPES.get(spec["type"])
        if expected and (not isinstance(value, expected) or isinstance(value, bool)
                         and spec["type"] != "boolean"):
            problems.append(f"'{name}' should be {spec['type']}, got {type(value).__name__}")
            continue
        if "allowed_values" in spec and value not in spec["allowed_values"]:
            problems.append(f"'{name}'={value!r} not in {spec['allowed_values']}")
        if spec.get("range") == "0.0 to 1.0" and not (0.0 <= float(value) <= 1.0):
            problems.append(f"'{name}'={value} outside 0.0 to 1.0")
        if spec["type"] == "array[string]" and not all(isinstance(x, str) for x in value):
            problems.append(f"'{name}' must contain only strings")
    rule = output.get("affected_slice_rule")
    if ref_df is not None and isinstance(rule, str):
        try:
            ref_df.query(rule)
        except Exception as e:  # noqa: BLE001
            problems.append(f"affected_slice_rule is not a valid pandas query: {e}")
    return problems


if __name__ == "__main__":
    # Quick self-check: python3.11 -m c2.src.outputs
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    from contracts.c1_fake_output import build_fake_c1_output
    from c2.src.config import RESULTS_DIR
    from c2.src.scenarios import make_temporal_stream

    engine = C2DiagnosisEngine()
    stream = make_temporal_stream("marginal_single", ["capital_gain"], 0.6, seed=22)
    engine.warmup([s["drifted_df"] for s in stream[:2]])
    for s in stream[2:]:
        c1 = build_fake_c1_output(s["ground_truth"], seed=s["spec"]["seed"])
        result = engine.diagnose(s["drifted_df"], c1)

    print("C2 -> C3 Repair Specification:")
    print(json.dumps(result["to_c3"], indent=2))
    print("\nC2 -> C4 Diagnostic Report:")
    print(json.dumps(result["to_c4"], indent=2))
    ref = engine.a["ref_df"]
    p3 = validate_output(result["to_c3"], C3_SCHEMA, ref)
    p4 = validate_output(result["to_c4"], C4_SCHEMA, ref)
    print(f"\nC3 contract: {'VALID' if not p3 else p3}")
    print(f"C4 contract: {'VALID' if not p4 else p4}")

    out_dir = RESULTS_DIR / "example_outputs"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "c2_to_c3_example.json").write_text(json.dumps(result["to_c3"], indent=2))
    (out_dir / "c2_to_c4_example.json").write_text(json.dumps(result["to_c4"], indent=2))
    print(f"examples saved to {out_dir}")
    print("OUTPUTS OK" if not (p3 or p4) else "OUTPUTS INVALID")