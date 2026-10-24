"""Admission-only bindings for the new experiment. No old artifact rewrites."""
from __future__ import annotations

from collections import Counter
import copy
import importlib.metadata
import json
from pathlib import Path

import fit_n1_validation_baselines as b

d, e, v = b.d, b.e, b.v
VERSION = v.VERSION


def fresh_group_cases(manifest, group_id):
    groups = [g for g in manifest["groups"] if g["group_id"] == group_id]
    d.require(len(groups) == 1 and groups[0]["split_role"] == "conformal_calibration",
              "This admission runner permits only a frozen conformal-calibration group.")
    rows = [s for s in manifest["scenarios"] if s["group_id"] == group_id]
    d.require(len(rows) == 37 and len({r["scenario_id"] for r in rows}) == 37
              and all(r["split_role"] == "conformal_calibration" for r in rows),
              "Missing/duplicated cases or role mismatch.")
    expected = {(m, strength, diagnosis, repair) for m in ("marginal_multi", "interaction", "combined")
                for strength in (.2, .5, .8) for diagnosis in ("exact_slice", "broadened_slice")
                for repair in ("targeted_repair", "label_corrupted_repair")}
    d.require({(r["mechanism"], r["magnitude"], r["diagnosis_condition"], r["repair_condition"])
               for r in rows if not r["is_no_drift_control"]} == expected, "Factorial grid differs.")
    d.require(sum(r["is_no_drift_control"] for r in rows) == 1, "Exactly one control required.")
    return sorted(rows, key=lambda r: (r["shared_base_package_id"],
                                      r["repair_condition"] != "targeted_repair", r["scenario_id"]))


def expected_admission_source(original):
    """Only scope/path/description edits; generation and admission code unchanged."""
    text = original.replace('"""Plan or execute ONE selected training-fold scenario through admission.',
        '"""Plan or execute ONE explicitly bound fresh conformal scenario through admission.')
    text = text.replace("from _training_group import training_group_cases, corrupt_labels",
        "from _training_group import corrupt_labels\nfrom _hybrid_fresh_execution import fresh_group_cases as training_group_cases, verify_execution_binding")
    text = text.replace("root = resolve_data_root(args.data_root)",
        "root = resolve_data_root(args.data_root)\n    verify_execution_binding(root, args.group_id)")
    text = text.replace("adult-simulation-pp1-v2", VERSION)
    text = text.replace("forecasting-training group", "fresh conformal-calibration group")
    text = text.replace("Forecasting-training group", "Fresh conformal-calibration group")
    text = text.replace("one_selected_training_scenario_claim", "one_selected_fresh_conformal_admission_claim")
    return text.replace("training_group_scenario_simulation_admission", "fresh_conformal_scenario_simulation_admission")


def bundle_at(root):
    return Path(root) / "c4_artifacts/forecasting_experiments/adult" / VERSION


def build_execution(root, c4):
    root, c4 = Path(root), Path(c4)
    directory, _, _, _, pins, _ = b.prepare(root, c4)
    protocol, protocol_hash = d.checked_json(directory, "validation_protocol")
    manifest, manifest_hash = d.checked_json(directory, "scenario_manifest")
    baseline_dir = directory / "baseline-fit-v1"
    baselines, baseline_hash = d.checked_json(baseline_dir)
    d.require(baselines["new_validation_cases_accessed"] is False
              and baselines["production_release_authorized"] is False
              and baselines["n2_calibrated"] is False and baselines["policy_calibrated"] is False,
              "Baseline fitting scope differs.")
    d.require(set(baselines["models"]) == {"hybrid_ridge_lightgbm_lightgbm", *b.METHODS},
              "Incomplete fitted comparator inventory.")
    primary = baselines["models"]["hybrid_ridge_lightgbm_lightgbm"]
    d.require(primary["metadata_sha256"] == protocol["fixed_hybrid_metadata_sha256"],
              "Primary model link differs.")
    claim, claim_hash = d.checked_json(baseline_dir, "claim")
    d.require(claim_hash == baselines["claim_sha256"] and claim["methods"] == b.METHODS,
              "Baseline fit claim differs.")
    for path, digest in claim["inputs"].items():
        d.checked(Path(path), digest)
        pins[path] = digest
    pins[str(baseline_dir / "metadata.json")] = baseline_hash
    for relative, digest in baselines["files"].items():
        path = baseline_dir / relative
        d.checked(path, digest)
        pins[str(path)] = digest
    original_case = (c4 / "scripts/run_training_group_case_admission.py").read_text(encoding="utf-8-sig")
    new_case = (c4 / "scripts/run_hybrid_fresh_case_admission.py").read_text(encoding="utf-8-sig")
    d.require(new_case == expected_admission_source(original_case),
              "Admission implementation differs beyond the declared scope adapter.")
    environment = {**e.environment(), **{p: importlib.metadata.version(p) for p in ("ctgan", "torch", "xgboost")}}
    d.require(environment["ctgan"] == protocol["factorial_source_protocol"]["repair_generation"]["ctgan_version"],
              "ctgan differs from frozen generation configuration.")
    for name in ("_hybrid_fresh_execution.py", "run_hybrid_fresh_case_admission.py", "run_hybrid_fresh_admissions.py",
                 "fit_n1_validation_baselines.py"):
        path = c4 / "scripts" / name
        pins[str(path)] = d.sha(path.read_bytes())
    original_dir = root / "c4_artifacts/benchmark_protocols/adult/adult-simulation-pp1-v2"
    old_freeze, old_freeze_hash = d.checked_json(original_dir, "freeze")
    original_bytes = d.checked(original_dir / "protocol.json", old_freeze["protocol_sha256"])
    scientific = json.loads(original_bytes)
    scientific["protocol_version"] = VERSION
    scientific["replication"]["seed_namespaces"] = list(v.NAMESPACES)
    scientific["replication"]["replications_per_namespace"] = 13
    scientific["scenario_split"]["fold_seed"] = 29147
    scientific["scenario_split"]["per_namespace"] = {
        "forecast_train": 0, "conformal_calibration": 6, "policy_calibration": 3, "outer_test": 4}
    scientific["fresh_validation_binding"] = dict(validation_protocol_sha256=protocol_hash,
        manifest_sha256=manifest_hash, assignment=manifest["assignment_note"],
        scope="admission_only_new_conformal_groups_no_candidate_or_future_execution")
    frozen = {k: copy.deepcopy(old_freeze[k]) for k in
        ("source_implementation_sha256", "champion_model_sha256", "split_manifest_sha256")}
    frozen.update(purpose="fresh_conformal_admission_compatibility_binding",
        status="frozen_specification_execution_disabled", full_execution_enabled=False,
        production_approval_export=False, production_release_authorized=False,
        protocol_sha256=d.sha(e.encoded(scientific)), scenario_manifest_sha256=manifest_hash,
        original_protocol_freeze_sha256=old_freeze_hash, validation_protocol_sha256=protocol_hash,
        baseline_metadata_sha256=baseline_hash)
    pins[str(original_dir / "freeze.json")] = old_freeze_hash
    execution = dict(purpose="fresh_conformal_admission_executor_binding_v1",
        validation_protocol_sha256=protocol_hash, manifest_sha256=manifest_hash,
        baseline_metadata_sha256=baseline_hash, inputs=pins, environment=environment,
        allowed_role="conformal_calibration", allowed_stages=["repair_generation", "simulation_admission"],
        candidate_training_authorized=False, future_evaluation_authorized=False,
        original_outcomes_accessed=False, production_release_authorized=False,
        compatibility_directory=str(root / "c4_artifacts/benchmark_protocols/adult" / VERSION),
        compatibility_hashes={"protocol": d.sha(e.encoded(scientific)),
            "freeze": d.sha(e.encoded(frozen)), "scenario_manifest": manifest_hash})
    return execution, scientific, frozen, manifest


def ensure_documents(directory, documents):
    directory = Path(directory)
    if directory.exists():
        for stem, document in documents.items():
            saved, _ = d.checked_json(directory, stem)
            d.require(e.encoded(saved) == e.encoded(document), "Existing immutable binding differs: "+stem)
        return
    v.freeze_bundle(directory, documents)


def materialize(root, documents):
    execution, scientific, frozen, manifest = documents
    for path, digest in execution["inputs"].items():
        d.checked(Path(path), digest)
    ensure_documents(execution["compatibility_directory"], {
        "protocol": scientific, "freeze": frozen, "scenario_manifest": manifest})
    ensure_documents(bundle_at(root) / "admission-executor-v1", {"claim": execution})


def verify_execution_binding(root, group_id):
    root = Path(root)
    execution, _ = d.checked_json(bundle_at(root) / "admission-executor-v1", "claim")
    d.require(execution["allowed_role"] == "conformal_calibration"
              and execution["candidate_training_authorized"] is False
              and execution["future_evaluation_authorized"] is False
              and execution["production_release_authorized"] is False,
              "Invalid admission execution scope.")
    current = {**e.environment(), **{p: importlib.metadata.version(p) for p in ("ctgan", "torch", "xgboost")}}
    d.require(current == execution["environment"], "Admission environment changed.")
    for path, digest in execution["inputs"].items():
        d.checked(Path(path), digest)
    directory = Path(execution["compatibility_directory"])
    for stem, digest in execution["compatibility_hashes"].items():
        d.checked(directory / (stem+".json"), digest)
    manifest, _ = d.checked_json(directory, "scenario_manifest")
    fresh_group_cases(manifest, group_id)
    return manifest
