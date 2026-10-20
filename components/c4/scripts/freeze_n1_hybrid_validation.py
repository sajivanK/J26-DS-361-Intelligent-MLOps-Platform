"""Plan/freeze fresh-group hybrid validation. NO generation, fitting or evaluation."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform

from _benchmark_manifest import build_manifest

VERSION = "adult-hybrid-fresh-validation-v1"
NAMESPACES = [51001, 51002, 51003, 51004]
COUNTS = {"conformal_calibration": 24, "policy_calibration": 12, "outer_test": 16}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(content):
    return hashlib.sha256(content).hexdigest()


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n").encode()


def checked(path, digest):
    contents = Path(path).read_bytes()
    require(sha(contents) == digest, f"Checksum mismatch: {path}")
    return contents


def checked_json(directory, stem="metadata"):
    directory = Path(directory)
    digest = (directory / (stem+".sha256")).read_text(encoding="utf-8").strip()
    return json.loads(checked(directory / (stem+".json"), digest)), digest


def child_seeds(manifest):
    return {seed for s in manifest["scenarios"]
            for seed in [s["generator_seed"], s["c1_mock_seed"], s["c2_bootstrap_seed"],
                         s["corruption_seed"], *s["drift_seeds"].values()]}


def create_design(original):
    """Reuse the exact factorial/seed builder; predeclare all new roles."""
    require(not set(NAMESPACES) & set(original["replication"]["seed_namespaces"]),
            "New namespaces overlap the original design.")
    source = copy.deepcopy(original)
    source["protocol_version"] = VERSION
    source["replication"]["seed_namespaces"] = list(NAMESPACES)
    source["replication"]["replications_per_namespace"] = 13
    # Existing builder requires four positive blocks. Merge the first two
    # deterministic blocks into calibration BEFORE any outcome is generated.
    # No new forecast-training cases are generated or fitted in this design.
    source["scenario_split"]["fold_seed"] = 29147
    source["scenario_split"]["per_namespace"] = {
        "forecast_train": 1, "conformal_calibration": 5,
        "policy_calibration": 3, "outer_test": 4}
    manifest = build_manifest(source)
    for row in [*manifest["groups"], *manifest["scenarios"]]:
        if row["split_role"] == "forecast_train":
            row["split_role"] = "conformal_calibration"
    require(dict(Counter(g["split_role"] for g in manifest["groups"])) == COUNTS,
            "Unexpected group assignment.")
    mapping = {g["group_id"]: g["split_role"] for g in manifest["groups"]}
    require(len(mapping) == 52 and len(manifest["scenarios"]) == 1924, "Unexpected design size.")
    require(all(s["split_role"] == mapping[s["group_id"]] for s in manifest["scenarios"]),
            "Related scenarios cross roles.")
    for group in mapping:
        rows = [s for s in manifest["scenarios"] if s["group_id"] == group]
        require(len(rows) == 37 and sum(s["is_no_drift_control"] for s in rows) == 1,
                "Factorial/control inventory differs.")
    manifest["assignment_note"] = (
        "SHA256 ordering with fold seed 29147; builder blocks 1+5 merged into "
        "six calibration groups per namespace, then three policy and four outer groups. "
        "All role mapping fixed before outcomes; no new forecast-training role.")
    return manifest


def freeze_bundle(output, documents):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for name, document in documents.items():
        content = encoded(document)
        (output / (name+".json")).write_bytes(content)
        (output / (name+".sha256")).write_text(sha(content)+"\n", encoding="utf-8")
        hashes[name] = sha(content)
    return hashes


def prepare(root, c4):
    pins = {}
    def read(directory, stem="metadata"):
        doc, digest = checked_json(directory, stem)
        pins[str(Path(directory) / (stem+".json"))] = digest
        return doc, digest
    old_dir = root / "c4_artifacts/benchmark_protocols/adult/adult-simulation-pp1-v2"
    frozen, _ = read(old_dir, "freeze")
    content = checked(old_dir / "protocol.json", frozen["protocol_sha256"])
    pins[str(old_dir / "protocol.json")] = sha(content)
    original = json.loads(content)
    old_manifest, digest = read(old_dir, "scenario_manifest")
    require(digest == frozen["scenario_manifest_sha256"], "Original manifest binding differs.")
    manifest = create_design(original)
    fresh_ids = {g["group_id"] for g in manifest["groups"]}
    fresh_seeds = child_seeds(manifest)
    # Read planning metadata only, never old case outcomes or target tables.
    manifests = root / "c4_artifacts/benchmark_protocols/adult"
    for path in sorted(manifests.glob("*/scenario_manifest.json")):
        previous, _ = read(path.parent, "scenario_manifest")
        require(not fresh_ids & {g["group_id"] for g in previous["groups"]},
                f"Group identity already present: {path}")
        require(not fresh_seeds & child_seeds(previous), f"Child seed overlaps previous design: {path}")
    for group in fresh_ids:
        require(not any((root / "c4_artifacts/benchmark_runs/adult").glob("*/"+group+"-*")),
                "Fresh group already has run directories; stop and inspect provenance.")
    hybrid_dir = root / "c4_artifacts/forecasting_experiments/adult/adult-hybrid-development-v1"
    hybrid_spec, hybrid_spec_hash = read(hybrid_dir, "specification")
    model_meta, model_meta_hash = read(hybrid_dir / "fit-v1")
    require(model_meta["specification_sha256"] == hybrid_spec_hash
            and model_meta["production_release_authorized"] is False
            and model_meta["conformal_calibrated"] is False
            and model_meta["calibration_policy_outer_cases_accessed"] is False,
            "Hybrid model scope/binding differs.")
    for name, digest in model_meta["files"].items():
        path = hybrid_dir / "fit-v1" / name
        require(Path(name).name == name, "Unexpected artifact filename.")
        checked(path, digest)
        pins[str(path)] = digest
    policy_bytes = (c4 / "configs/policy_benchmark_v1.json").read_bytes()
    policy = json.loads(policy_bytes)
    pins[str(c4 / "configs/policy_benchmark_v1.json")] = sha(policy_bytes)
    require(policy["safety_boundaries"] == {
        "future_global_margin": 0., "future_slice_margin": 0., "future_nonaffected_margin": -.02},
        "Original safety boundaries differ.")
    require(original["N2"]["alpha"] == .05, "Unexpected N2 alpha.")
    for folder in (c4 / "scripts", c4 / "src"):
        for path in sorted(folder.rglob("*.py")):
            pins[str(path)] = sha(path.read_bytes())
    return dict(version=VERSION, purpose="fresh_group_validation_of_fixed_development_hybrid",
        group_counts=COUNTS, factorial_source_protocol=original,
        fixed_hybrid_specification_sha256=hybrid_spec_hash, fixed_hybrid_metadata_sha256=model_meta_hash,
        primary_method="hybrid_ridge_lightgbm_lightgbm",
        comparison_methods=["same_external_training_lightgbm", "same_external_training_linear", "same_external_training_constant_quantiles"],
        comparison_fit_rule="Use exactly the verified external-development training inputs and fixed screened parameters; freeze fitted baseline hashes before any new case.",
        original_frozen_controller_comparison="secondary; original training-label cohort differs, so not a model-only comparison",
        N2=original["N2"], policy_rules=policy,
        model_methods_override_policy_source=["hybrid_ridge_lightgbm_lightgbm", "same_external_training_lightgbm", "same_external_training_linear", "same_external_training_constant_quantiles"],
        fixed_external_cohort="The same 12899-row external DEVELOPMENT cohort; scenario-group validation conditional on shared data, not an independent-cohort test.",
        cohort_binding="Executor must verify the original completed external experiment/cohort hashes, disjointness and preprocessing before execution.",
        targets="candidate-minus-champion accuracy margins: global, true injected slice, true complement",
        pre_release_feature_populations="global, diagnosed slice, diagnosed-slice complement; no true-slice or future targets in features",
        inference_order="Freeze candidate/pre-release evidence and all model forecasts before reading the new case future outcomes.",
        execution_order=["freeze_fitted_baselines_and_executor_implementation", "conformal_groups", "freeze_N2", "policy_groups", "freeze_policy", "outer_groups_once"],
        empty_or_blocked_groups="Retain without outcome-based exclusion or replacement. Report assigned and nonempty counts separately.",
        insufficient_N2_support="Keep original full-range fallback; fewer than 19 nonempty groups cannot supply the finite 95% order statistic.",
        stop_rule="No seed replacement, outcome-driven threshold changes, optional final-test stopping, or model tuning after validation outcomes.",
        final_metrics=["equal-group MAE/RMSE/R2", "quantile-specific and mean pinball", "scenario and whole-group joint coverage", "interval width", "decision counts", "group-averaged automation", "conditional unsafe rate undefined when no deployment"],
        adoption_rule="Report all paired results; no automatic adoption. Assess forecast error, quantile loss, population regressions and calibrated safety together; do not select on R2 alone.",
        workload=dict(groups=52, cases=1924, maximum_TVAE_fits=988, candidate_fits="admitted cases only; unknown before admission"),
        inputs=pins, environment={"python": platform.python_version(), **{p: importlib.metadata.version(p)
            for p in ("numpy", "pandas", "scikit-learn", "lightgbm", "scipy")}},
        validation_executor_implemented=False, execution_authorized=False,
        experiments_executed=0, production_release_authorized=False), manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path)
    p.add_argument("--freeze", action="store_true")
    args = p.parse_args()
    root = args.data_root or os.environ.get("C4_DATA_ROOT")
    require(bool(root), "Set C4_DATA_ROOT.")
    root = Path(root)
    c4 = Path(__file__).resolve().parents[1]
    protocol, manifest = prepare(root, c4)
    print("GROUPS:", COUNTS)
    print("PLANNED CASES: 1924; MAXIMUM NEW TVAE FITS: 988")
    print("No candidate fitting, target evaluation, calibration or old outcome reads.")
    if not args.freeze:
        print("PLAN ONLY. --freeze writes the specification; it does not execute experiments.")
        return
    for path, digest in protocol["inputs"].items():
        checked(path, digest)
    directory = root / "c4_artifacts/forecasting_experiments/adult" / VERSION
    hashes = freeze_bundle(directory, {"validation_protocol": protocol, "scenario_manifest": manifest})
    print("FROZEN:", directory)
    print("PROTOCOL SHA256:", hashes["validation_protocol"])
    print("MANIFEST SHA256:", hashes["scenario_manifest"])
    print("EXECUTION DISABLED. Executor and fitted baseline bindings remain pending; original C4 unchanged.")


if __name__ == "__main__":
    main()
