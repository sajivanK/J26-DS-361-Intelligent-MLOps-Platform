"""Fresh policy-role admission authorization after immutable four-method N2.

Repair/admission science is reused unchanged. No policy fitting or future reads.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

import _hybrid_fresh_execution as base

d, e, v = base.d, base.e, base.v
bundle_at, ensure_documents = base.bundle_at, base.ensure_documents
C4 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(C4 / "src"))
from c4.conformal.benchmark_n2 import JointCalibration, apply_joint_calibration


def policy_group_cases(manifest, group_id):
    """Use the exact original factorial validator/order with a role-only adapter."""
    groups = [g for g in manifest["groups"] if g["group_id"] == group_id]
    d.require(len(groups) == 1 and groups[0]["split_role"] == "policy_calibration",
              "This runner permits only a frozen policy-calibration group.")
    originals = [s for s in manifest["scenarios"] if s["group_id"] == group_id]
    d.require(all(s["split_role"] == "policy_calibration" for s in originals), "Policy case role mismatch.")
    # No mutation of the frozen manifest or returned scenario factors/seeds.
    view = {"groups": [{**groups[0], "split_role": "conformal_calibration"}],
            "scenarios": [{**s, "split_role": "conformal_calibration"} for s in originals]}
    ordered = base.fresh_group_cases(view, group_id)
    by_id = {s["scenario_id"]: s for s in originals}
    return [by_id[s["scenario_id"]] for s in ordered]


def expected_case_source(source):
    text = source.replace("fresh conformal", "fresh policy").replace("Fresh conformal", "Fresh policy")
    text = text.replace("from _hybrid_fresh_execution import fresh_group_cases as training_group_cases, verify_execution_binding",
        "from _hybrid_policy_execution import policy_group_cases as training_group_cases, verify_execution_binding")
    return text.replace("fresh_conformal", "fresh_policy")


def expected_group_function(source):
    start = source.index("def run_group(")
    end = source.index("\n\ndef main():", start)
    text = source[start:end]
    text = text.replace("f.fresh_group_cases", "f.policy_group_cases")
    text = text.replace("admission-executor-v1", "policy-admission-executor-v1")
    text = text.replace("conformal_calibration", "policy_calibration")
    text = text.replace("fresh_conformal", "fresh_policy")
    text = text.replace("run_hybrid_fresh_case_admission.py", "run_hybrid_policy_case_admission.py")
    return text


def verify_n2(root, full_inputs=False):
    directory = bundle_at(root) / "n2-calibration-v1"
    claim, claim_hash = d.checked_json(directory, "claim")
    output = directory / "completion-v1"
    metadata, metadata_hash = d.checked_json(output)
    d.require(claim["purpose"] == "fresh_four_method_N2_group_max_calibration_v1"
              and metadata["purpose"] == "verified_fresh_N2_calibration"
              and metadata["claim_sha256"] == claim_hash
              and claim["production_release_authorized"] is False
              and metadata["production_release_authorized"] is False
              and metadata["independent_coverage_validated"] is False
              and metadata["outer_test_evaluated"] is False,
              "Completed fresh four-method N2 calibration is required before policy admissions.")
    for key in ("assigned_groups", "nonempty_groups", "all_blocked_groups", "cases", "blocked_cases_retained"):
        d.require(metadata[key] == claim[key], "N2 inventory bindings differ.")
    d.require(metadata["assigned_groups"] == 24
              and metadata["nonempty_groups"]+metadata["all_blocked_groups"] == 24,
              "N2 assigned/nonempty group accounting differs.")
    for name, checksum in metadata["files"].items():
        d.checked(output / name, checksum)
    calibration, calibration_hash = d.checked_json(output, "calibration")
    d.require(metadata["files"]["calibration.json"] == calibration_hash
              and calibration["claim_sha256"] == claim_hash
              and calibration["definition"] == claim["definition"]
              and calibration["model_bindings"] == claim["model_bindings"]
              and calibration["production_release_authorized"] is False
              and calibration["policy_calibrated"] is False
              and calibration["independent_coverage_validated"] is False,
              "N2 model/rule/provenance bindings differ.")
    expected_methods = {"hybrid_ridge_lightgbm_lightgbm", *base.b.METHODS}
    d.require(set(calibration["methods"]) == set(claim["model_bindings"]) == expected_methods,
              "All four fixed N2 calibrations must be present.")
    columns = {t+"__"+label: [0.] for t in d.TARGETS for _, label in d.QUANTILES}
    for payload in calibration["methods"].values():
        state = JointCalibration.from_dict(payload)
        d.require(state.nonempty_groups == metadata["nonempty_groups"], "N2 numerical support differs.")
        apply_joint_calibration(state, pd.DataFrame(columns, index=["technical_support_check"]))
    if full_inputs:
        for path, checksum in claim["inputs"].items():
            d.checked(Path(path), checksum)
    pins = {str(directory / "claim.json"): claim_hash, str(output / "metadata.json"): metadata_hash}
    pins.update({str(output / name): checksum for name, checksum in metadata["files"].items()})
    return claim, calibration, pins


def build_execution(root, c4=C4):
    root, c4 = Path(root), Path(c4)
    n2_claim, calibration, pins = verify_n2(root, full_inputs=True)
    from _hybrid_fresh_candidate import environment
    current = environment()
    d.require(n2_claim["environment"] == current, "N2/admission environment changed.")
    old_execution, old_hash = d.checked_json(bundle_at(root) / "admission-executor-v1", "claim")
    frozen_dir = root / "c4_artifacts/benchmark_protocols/adult" / v.VERSION
    for stem, checksum in old_execution["compatibility_hashes"].items():
        path = frozen_dir / (stem+".json")
        d.checked(path, checksum)
        pins[str(path)] = checksum
    pins[str(bundle_at(root) / "admission-executor-v1/claim.json")] = old_hash
    frozen, freeze_hash = d.checked_json(frozen_dir, "freeze")
    protocol = json.loads(d.checked(frozen_dir / "protocol.json", frozen["protocol_sha256"]))
    manifest, manifest_hash = d.checked_json(frozen_dir, "scenario_manifest")
    d.require(manifest_hash == frozen["scenario_manifest_sha256"]
              and calibration["definition"] == protocol["N2"], "Policy protocol/N2 rule binding differs.")
    groups = [g["group_id"] for g in manifest["groups"] if g["split_role"] == "policy_calibration"]
    d.require(len(groups) == len(set(groups)) == 12, "Exactly 12 fixed policy groups are required.")
    for gid in groups:
        policy_group_cases(manifest, gid)
    original = (c4 / "scripts/run_hybrid_fresh_case_admission.py").read_text(encoding="utf-8-sig")
    adapted = (c4 / "scripts/run_hybrid_policy_case_admission.py").read_text(encoding="utf-8-sig")
    d.require(adapted == expected_case_source(original), "Policy repair/admission science changed beyond the role adapter.")
    group_source = (c4 / "scripts/run_hybrid_fresh_admissions.py").read_text(encoding="utf-8-sig")
    new_source = (c4 / "scripts/run_hybrid_policy_admissions.py").read_text(encoding="utf-8-sig")
    start = new_source.index("def run_group(")
    end = new_source.index("\n\ndef completed_group(", start)
    d.require(new_source[start:end] == expected_group_function(group_source), "Policy admission checkpoint/order implementation differs.")
    # Admission runtime consumes N2 artifacts, source code, and the original
    # reference/champion/split inputs. Old calibration outcome CSVs are not inputs
    # to repair generation; their transitive provenance remains in the N2 claim.
    repo = c4.parent.parent
    for path, checksum in n2_claim["inputs"].items():
        source = Path(path)
        if source.is_relative_to(repo) and source.suffix in {".py", ".json"}:
            pins[path] = checksum
    champion_dir = root / "c4_artifacts/champion/adult" / protocol["source_versions"]["champion"]
    for path in (root / "uci_adult_data/reference.csv", root / "uci_adult_data/test.csv",
                 champion_dir / "metadata.json"):
        checksum = n2_claim["inputs"].get(str(path))
        d.require(bool(checksum), "Required admission input is not bound by N2 provenance: "+str(path))
        d.checked(path, checksum)
        pins[str(path)] = checksum
    pins[str(champion_dir / "champion.ubj")] = frozen["champion_model_sha256"]
    split_dir = root / "c4_artifacts/splits/adult" / protocol["source_versions"]["split_manifest"]
    pins[str(split_dir / "manifest.json")] = frozen["split_manifest_sha256"]
    for path, checksum in pins.items():
        d.checked(Path(path), checksum)
    for path in (c4 / "scripts/_hybrid_policy_execution.py", c4 / "scripts/run_hybrid_policy_case_admission.py",
                 c4 / "scripts/run_hybrid_policy_admissions.py"):
        pins[str(path)] = d.sha(path.read_bytes())
    claim = dict(purpose="fresh_policy_admission_executor_v1", inputs=pins, environment=current,
        allowed_role="policy_calibration", allowed_group_ids=groups, planned_groups=12, planned_cases=444,
        frozen_protocol_sha256=freeze_hash, manifest_sha256=manifest_hash,
        N2_model_bindings=calibration["model_bindings"], N2_fixed_before_policy_admission=True,
        allowed_stages=["repair_generation", "simulation_admission"],
        candidate_training_authorized=False, future_evaluation_authorized=False,
        N1_refitting_authorized=False, N2_refitting_authorized=False, policy_fitting_authorized=False,
        outer_evaluation_authorized=False, production_release_authorized=False)
    return claim, manifest


def verify_execution_binding(root, group_id):
    from _hybrid_fresh_candidate import environment
    root = Path(root)
    claim, _ = d.checked_json(bundle_at(root) / "policy-admission-executor-v1", "claim")
    d.require(claim["purpose"] == "fresh_policy_admission_executor_v1"
              and claim["allowed_role"] == "policy_calibration"
              and group_id in claim["allowed_group_ids"]
              and claim["candidate_training_authorized"] is False
              and claim["future_evaluation_authorized"] is False
              and claim["N1_refitting_authorized"] is False
              and claim["N2_refitting_authorized"] is False
              and claim["policy_fitting_authorized"] is False
              and claim["outer_evaluation_authorized"] is False
              and claim["production_release_authorized"] is False, "Policy admission scope is invalid.")
    d.require(claim["environment"] == environment(), "Policy admission environment changed.")
    for path, checksum in claim["inputs"].items():
        d.checked(Path(path), checksum)
    frozen_dir = root / "c4_artifacts/benchmark_protocols/adult" / v.VERSION
    frozen, freeze_hash = d.checked_json(frozen_dir, "freeze")
    manifest, manifest_hash = d.checked_json(frozen_dir, "scenario_manifest")
    d.require(freeze_hash == claim["frozen_protocol_sha256"]
              and manifest_hash == claim["manifest_sha256"] == frozen["scenario_manifest_sha256"],
              "Policy frozen manifest/protocol changed.")
    policy_group_cases(manifest, group_id)
    return manifest
