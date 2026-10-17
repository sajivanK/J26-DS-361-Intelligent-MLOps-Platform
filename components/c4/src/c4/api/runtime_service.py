"""Standalone operational preparation service; not a release-controller endpoint.

Do not expose execute_training/configuration/data-root directly as untrusted HTTP
fields. Shared platform integration and diagnosed-slice forecasting are pending.
"""
from pathlib import Path
import re

from c4.contracts.platform_adapter import adapt_platform_upstream
from c4.contracts.runtime_configuration import load_runtime_configuration


def _validate_output(data_root, version):
    if (not isinstance(version, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}", version)
            or version.endswith(".")
            or version.split(".")[0].upper() in {
                "CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
                *(f"LPT{i}" for i in range(1, 10))}):
        raise ValueError("Provide a valid new operational artifact version.")
    if data_root is None or not str(data_root).strip():
        raise ValueError("Explicit output data root required for execution.")
    output = Path(data_root).expanduser().resolve() / "c4_artifacts/operational_candidates/adult" / version
    if output.exists():
        raise FileExistsError("Operational artifact version already exists; preserve it.")


def _evaluate(adapted, bindings):
    from c4.models.runtime_candidate import evaluate_runtime_candidate
    return evaluate_runtime_candidate(adapted, bindings, execute_training=True)


def _persist(result, data_root, version):
    from c4.models.runtime_artifacts import save_runtime_candidate
    return save_runtime_candidate(result, data_root=data_root, version=version)


def prepare_runtime_candidate(*, config_path, upstream, execute_training=False,
                              data_root=None, version=None):
    """Plan by default; execution returns candidate evidence, never release approval.

    Exceptions propagate as operational failures, never fabricated decisions.
    The successful manifest hash must be kept in the caller's audit record.
    """
    if type(execute_training) is not bool:
        raise ValueError("execute_training must be a boolean.")
    adapted = adapt_platform_upstream(upstream)
    config = load_runtime_configuration(config_path)
    report = {
        "status": "PLAN_ONLY", "run_id": adapted.context.run_id,
        "scope": "operational_candidate_preparation_only",
        "missing_bindings": list(config.missing),
        "prerequisite_failures": list(adapted.prerequisite_failures),
        "candidate_training_executed": False,
        "artifact_persistence_completed": False,
        "forecast_applicable": False,
        "N1_executed": False, "N2_executed": False,
        "release_controller_executed": False,
        "production_release_authorized": False,
        "release_decision": None,
    }
    if adapted.prerequisite_failures:
        report["status"] = "UPSTREAM_PREREQUISITES_FAILED"
        return report
    if config.missing:
        report["status"] = "CONFIGURATION_INCOMPLETE"
        return report
    if not execute_training:
        report["note"] = "Configuration parsed; artifact compatibility/model checks have not run."
        return report
    _validate_output(data_root, version)
    result = _evaluate(adapted, config.bindings)
    artifact = _persist(result, data_root, version)
    report.update(
        status="CANDIDATE_SAVED_RELEASE_PENDING",
        candidate_training_executed=True,
        artifact_persistence_completed=True,
        artifact_directory=str(artifact.directory),
        manifest_sha256=artifact.manifest_sha256,
        candidate_model_sha256=result.report["candidate_model_sha256"],
        pre_release_evaluation=result.report["pre_release_evaluation"],
        forecast_reason=result.report["forecast_reason"],
        next_action="Review candidate evidence. Operational forecasting and release integration remain pending.",
    )
    return report
