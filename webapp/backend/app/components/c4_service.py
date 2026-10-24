"""C4-only recorded-case inference. No raw-upload or production deployment adapter."""
import hashlib
import importlib
import json
import os
import sys
import threading
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
SCRIPTS = REPO / "components/c4/scripts"
EXPECTED_CONTROLLER = "de09b0220f25b3c82f8849dd2becef4f3301806dfcb1e701d5108b0d252fb4a7"
_lock = threading.RLock()
_loaded = None

class C4Unavailable(RuntimeError):
    pass

def _json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))

def recorded_results():
    return _json(Path(__file__).with_name("c4_recorded_results.json"))

def _runtime():
    global _loaded
    if _loaded is None:
        # Existing research helpers have short module names. Reject a collision
        # rather than replacing another component's already imported module.
        for name in ("_common", "_controller_artifacts", "verify_c4_controller"):
            module = sys.modules.get(name)
            if module and Path(module.__file__).resolve().parent != SCRIPTS.resolve():
                raise C4Unavailable("Research helper import collision: " + name)
        if not SCRIPTS.is_dir():
            raise C4Unavailable("Missing components/c4/scripts research source.")
        scripts = str(SCRIPTS)
        if scripts not in sys.path:
            sys.path.insert(0, scripts)
        try:
            common = importlib.import_module("_common")
            artifacts = importlib.import_module("_controller_artifacts")
            controller, binding, digest = artifacts.load_controller(common.resolve_data_root(None))
            if digest != EXPECTED_CONTROLLER:
                raise C4Unavailable("Unexpected frozen controller version.")
            _loaded = (controller, binding, digest)
        except C4Unavailable:
            raise
        except Exception as exc:
            raise C4Unavailable("C4 artifacts could not be verified or loaded. Check C4_DATA_ROOT and the research Python dependencies.") from exc
    return _loaded

def _demo():
    path = REPO / "components/c4/demo/real_outer_case_input.json"
    if not path.is_file():
        raise C4Unavailable("Missing committed real_outer_case_input.json demo evidence.")
    payload = _json(path)
    if set(payload) != {"evidence_id", "features", "hard_gates"}:
        raise C4Unavailable("Recorded demo input schema differs.")
    if payload["evidence_id"] != "adult-ns42-r002-interaction-medium-broadened_slice-targeted_repair":
        raise C4Unavailable("Unexpected recorded demo case.")
    common = importlib.import_module("_common")
    root = common.resolve_data_root(None)
    version = "adult-simulation-pp1-v2"
    registry, registry_hash = common.checked_json(root / "c4_artifacts/outer_pre_release/adult" / version / "outer-all56-decision-freeze-v1", "registry")
    if registry_hash != "8630e0ad7e39b2ca2b9b4921fe531f6964e1ef48f2dc14de9c1350a5629b2493":
        raise C4Unavailable("Outer pre-release registry differs.")
    directory = root / "c4_artifacts/benchmark_runs/adult" / version / payload["evidence_id"] / "pre_release_freeze-v1"
    frozen, frozen_hash = common.checked_json(directory, "freeze")
    saved, saved_hash = common.checked_json(directory, "comparison_forecasts")
    definition = registry["definition"]
    if (definition["scenario_freeze_sha256"][payload["evidence_id"]] != frozen_hash
        or definition["comparison_forecast_sha256"][payload["evidence_id"]] != saved_hash
        or frozen["comparison_forecasts_sha256"] != saved_hash
        or saved["future_targets_included"] is not False
        or saved["decision_time_features"] != payload["features"]
        or saved["explicit_hard_gates"] != payload["hard_gates"]):
        raise C4Unavailable("Demo input differs from the frozen pre-release evidence.")
    return payload, hashlib.sha256(path.read_bytes()).hexdigest()

def _case_details(payload):
    """Read only pinned pre-release records. No model fitting or future outcome reads."""
    common = importlib.import_module("_common")
    root = common.resolve_data_root(None)
    version = "adult-simulation-pp1-v2"
    run = root / "c4_artifacts/benchmark_runs/adult" / version / payload["evidence_id"]
    frozen, _ = common.checked_json(run / "pre_release_freeze-v1", "freeze")
    upstream, upstream_hash = common.checked_json(run / "pre_release_freeze-v1", "upstream_simulation")
    if upstream_hash != frozen["upstream_simulation_sha256"]:
        raise C4Unavailable("Saved upstream evidence differs from the pre-release freeze.")
    evidence, evidence_hash = common.checked_json(root / "c4_artifacts/pre_release/adult" / payload["evidence_id"], "evidence")
    if evidence_hash != frozen["pre_release_evidence_sha256"]:
        raise C4Unavailable("Saved candidate evaluation differs from the pre-release freeze.")
    # Exact feature reconstruction checks that displayed values belong to the same input.
    helper_path = SCRIPTS / "_scenario_meta.py"
    helper_links = [digest for path,digest in frozen["dependencies"].items()
                    if path.replace("\\", "/").endswith("/scripts/_scenario_meta.py")]
    if len(helper_links) != 1:
        raise C4Unavailable("Missing pinned feature reconstruction implementation.")
    common.checked_bytes(helper_path, helper_links[0])
    feature_module = importlib.import_module("_scenario_meta")
    reconstructed = feature_module.feature_row(upstream["upstream"], evidence)
    if reconstructed != payload["features"]:
        raise C4Unavailable("Displayed evidence does not reconstruct the decision vector.")
    metadata, metadata_hash = common.checked_json(run / "scoped_candidate_validation-v1")
    linked = [value for name,value in frozen["dependencies"].items()
              if name.replace("\\", "/").endswith("/scoped_candidate_validation-v1/metadata.json")]
    if linked != [metadata_hash]:
        raise C4Unavailable("Candidate metadata is not bound to the case freeze.")
    return {
        "scenario":frozen["scenario"],
        "champion_model_id":metadata.get("champion_version"),
        "candidate_model_id":metadata.get("artifact_version"),
        "pre_release_recorded_at":frozen.get("created_at_utc"),
        "candidate_evaluation":evidence["report"]["comparisons"],
        "upstream_evidence":upstream["upstream"],
        "source_declarations":upstream.get("source_declarations",{}),
        "privacy_interpretation":upstream.get("privacy_interpretation"),
        "production_package_approved":upstream.get("production_package_approved",False),
        "rollback_scope":frozen.get("runtime_checks",{}).get("rollback_scope"),
        "pre_release_evidence_sha256":evidence_hash,
        "upstream_simulation_sha256":upstream_hash,
        "candidate_metadata_sha256":metadata_hash,
        "displayed_evidence_reconstructed":True,
    }

def catalog():
    return {
        "scope":"Adult accuracy-margin pilot", "live_upstream_integration":False,
        "production_release_authorized":False,
        "cases":[
            {"id":"revised-outer-case", "label":"Revised Adult case · hybrid", "mode":"verified_revised_recorded_replay"},
            {"id":"real-outer-case", "label":"Original PP1 Adult case", "mode":"benchmark_replay"}],
        "default_case":"revised-outer-case", "revised_methods":["hybrid","lightgbm","linear","constant"],
        "methods":["lightgbm","linear"], "families":["adaptive","global"],
        "controller_freeze_sha256":EXPECTED_CONTROLLER,
        "note":"Pipeline C4 placeholder remains fake until actual upstream integration is implemented."
    }

def evaluate(case_id, method="lightgbm", family="adaptive"):
    if case_id != "real-outer-case":
        raise KeyError(case_id)
    if method not in ("lightgbm","linear") or family not in ("adaptive","global"):
        raise ValueError("Unknown forecasting method or policy family.")
    with _lock:
        controller, binding, digest = _runtime()
        payload, input_hash = _demo()
        details = _case_details(payload)
        result = controller.decide(payload["features"],payload["hard_gates"],
            evidence_id=payload["evidence_id"],method=method,family=family)
        return {"mode":"benchmark_replay", "upstream_source":"simulated",
            "inference_executed":True, "candidate_retrained":False,
            "controller_freeze_sha256":digest, "input_file_sha256":input_hash,
            "hard_gates":payload["hard_gates"], "decision_time_features":payload["features"],
            "case_details":details,
            "n2_calibration":controller.calibrations[method].to_dict(),
            "candidate_evaluation":details["candidate_evaluation"],
            "candidate_evaluation_note":"Observed pre-release evaluation on the diagnosed slice and its complement.",
            "result":result}

def fixtures():
    with _lock:
        _, binding, digest = _runtime()
        # Execute the existing acceptance helper and actual controller class.
        helper = importlib.import_module("verify_c4_controller")
        definition = _json(REPO / "components/c4/configs/controller_acceptance_fixtures_v1.json")
        results = helper.acceptance_results(definition,binding["selected_policies"]["lightgbm"]["adaptive"])
        return {"mode":"synthetic_acceptance", "controller_freeze_sha256":digest,
            "research_observations":False, "empirical_auto_deployment_evidence":False,
            "production_release_authorized":False, "fixtures":results}


def hybrid_results():
    """Read the separate revised research summary; never replace case inference."""
    from .c4_hybrid_research import read_results, ResearchUnavailable
    try:
        return read_results()
    except ResearchUnavailable as exc:
        raise C4Unavailable(str(exc)) from exc


def revised_evaluate(case_id="revised-outer-case", method="hybrid", family="adaptive"):
    """Replay matched revised evidence; keep original inference entry point separate."""
    from .c4_revised_case import replay, RevisedCaseUnavailable
    try:
        return replay(case_id, method, family)
    except RevisedCaseUnavailable as exc:
        raise C4Unavailable(str(exc)) from exc
