"""Platform adapter for artifact-bound C3 repair; no dataset substitution."""
from typing import Any

from component3.src.integration_fallback import rejected_output

from .. import contracts
from ..settings import c3_data_context
from .base import Component


class C3Component(Component):
    name = "c3"
    title = "C3 Synthetic Data Repair"
    owner = "Mathusan K"
    produces = ("c3",)
    source = "real"

    def load(self):
        self.fallback_runs = 0
        self.last_reason = None
        self.dependency_error = None
        try:
            from component3.src.live_repair import execute_live_repair
            self.source = "real"
        except Exception as exc:  # Import-time dependency and binary compatibility failures.
            self.source = "fake"
            self.dependency_error = f"{type(exc).__name__}: {exc}"
        self.loaded = True

    def reject(self, reason, context=None, upstream=None):
        self.fallback_runs = getattr(self, "fallback_runs", 0) + 1
        self.last_reason = reason
        output = rejected_output(reason)
        context = context or {}
        output.update(execution_status="BLOCKED", model_id=context.get("model_id"),
                      dataset_id=context.get("dataset_id", context.get("dataset")),
                      pipeline_run_id=context.get("run_id"), batch_id=context.get("batch_id"))
        output.update(run_id=output["repair_version"], preview_rows=[], preview_columns=[], download_url=None,
                      readiness_passed=False, upstream_evidence={"c1": (upstream or {}).get("c1"),
                      "c2": (upstream or {}).get("c2_to_c3")},
                      provenance={"data_context": context, "pipeline_run_id": context.get("run_id"),
                                  "batch_id": context.get("batch_id")})
        from component3.src.integration_storage import run_root
        from component3.src.execution_progress import Progress
        import json
        directory = run_root() / output["run_id"]
        progress = Progress(directory, output["run_id"], context)
        progress.stage(1, "Blocked", reason)
        progress.stage(13, detail="BLOCKED")
        progress.stage(14)
        progress.finish("BLOCKED", reason)
        (directory / "result.json").write_text(json.dumps(output, default=str), encoding="utf-8")
        errors = contracts.validate("c3", output)
        if errors:
            raise ValueError(f"C3 fallback output contract failure: {errors}")
        return {"c3": output}

    def details(self):
        return {"fallback_runs": getattr(self, "fallback_runs", 0),
                "last_reason": getattr(self, "last_reason", None),
                "dependency_error": getattr(self, "dependency_error", None),
                "supported_generators": ["TVAE", "CTGAN", "TabDDPM"],
                "supported_tasks": ["tabular_binary_classification"],
                "decision_modes": ["RULE_BASED"], "version": "2.0.0", "contract_version": "1.1"}

    def run(self, batch, upstream: dict[str, Any]) -> dict[str, Any]:
        context = {**upstream.get("context", {}), **upstream.get("data_context", {})}
        def blocked(reason, *_):
            return self.reject(reason, context, upstream)
        if getattr(self, "dependency_error", None):
            return blocked("C3 dependencies unavailable: " + self.dependency_error)
        if upstream.get("component_sources", {}).get("c1") == "fake":
            return blocked("Real C1 drift evidence unavailable; repair was not attempted.")
        context_hint = {**upstream.get("context", {}), **upstream.get("data_context", {})}
        if (upstream.get("component_sources", {}).get("c2") == "fake"
                or context_hint.get("dataset", context_hint.get("dataset_id")) == "ieee"):
            return blocked("C2 diagnosis not available for this dataset.")
        import pandas as pd
        from component3.src.live_repair import execute_live_repair, response_for_package, artifact_path
        from component3.src.contracts.external_contracts import ContractValidationError, DependencyResolutionError

        context = {**upstream.get("context", {}), **upstream.get("data_context", {})}
        try:
            context = {**c3_data_context(context.get("model_id", "")), **context}
        except (ValueError, OSError) as exc:
            return blocked(f"C3 onboarding context unavailable: {exc}")
        context.setdefault("dataset_id", context.get("dataset"))
        if upstream.get("component_sources", {}).get("c2") == "fake":
            return blocked("C2 diagnosis not available for this dataset.")
        current_uri = context.get("current_dataset_uri")
        if not current_uri or not context.get("reference_dataset_uri") or not context.get("model_uri"):
            return blocked("C3 requires onboarding context with the exact monitored model, reference dataset and current batch.")
        if context.get("label_status", "UNAVAILABLE") != "AVAILABLE":
            return blocked("Authentic current labels unavailable; waiting for labelled adaptation and independent evaluation rows.", context)
        try:
            path = artifact_path(current_uri, "datasets")
        except DependencyResolutionError as exc:
            return blocked(f"C3 current batch unavailable: {exc}")
        actual = pd.read_csv(path)
        target = context.get("target_column")
        sampled_labels = context.get("adaptation_dataset_uri") and context.get("evaluation_dataset_uri")
        if not sampled_labels and (target not in actual or actual[target].isna().any()):
            return blocked("Authentic current labels unavailable; waiting for labelled adaptation and independent evaluation rows.")
        if len(actual) < 100:
            return blocked("Insufficient current rows: need at least 100 labelled rows.")
        columns = [c for c in batch.columns if c != context.get("target_column")]
        if not set(columns).issubset(actual.columns):
            raise ContractValidationError("Current dataset does not contain the monitored batch's columns.")
        try:
            pd.testing.assert_frame_equal(batch[columns].reset_index(drop=True),
                actual[columns].reset_index(drop=True), check_dtype=False, check_exact=False)
        except AssertionError as exc:
            raise ContractValidationError("current_dataset_uri does not represent the actual monitored batch.") from exc
        # Verify fitted model state, not just a model name. This leaves C1/C2 algorithms untouched.
        from . import BY_NAME
        from component3.src.core.model_loader import load_trusted_pickle
        import joblib
        import json

        def fingerprint(model):
            if hasattr(model, "get_booster"):
                booster = model.get_booster()
                learner = json.loads(booster.save_config())["learner"]
                return joblib.hash((booster.get_dump(dump_format="json"), booster.feature_names,
                                    list(model.classes_), learner["learner_model_param"], learner["objective"]))
            return joblib.hash(model)

        try:
            bound_model = load_trusted_pickle(artifact_path(context["model_uri"], "models"), trusted=True)
            expected = fingerprint(bound_model)
            c1_component, c2_component = BY_NAME["c1"], BY_NAME["c2"]
            if c1_component.engines and fingerprint(c1_component.engine(context["dataset"]).base_model) != expected:
                return blocked("C1 monitored model differs from C3's supplied model artifact.", context)
            if c2_component.engine is not None and fingerprint(c2_component.engine.a["model"]) != expected:
                return blocked("C2 diagnosed model differs from C3's supplied model artifact.", context)
        except (OSError, ValueError, DependencyResolutionError) as exc:
            return blocked(f"Monitored model binding failed: {exc}", context)

        payload = {"c1_evidence": upstream.get("c1"),
                   "c2_diagnosis": upstream.get("c2_to_c3", upstream.get("c2")),
                   "data_context": context}
        try:
            package, log = execute_live_repair(payload)
        except DependencyResolutionError as exc:
            return blocked(f"C3 artifacts unavailable: {exc}")
        except ContractValidationError as exc:
            # Invalid integration inputs are visible rejections, never fake repairs.
            return blocked(f"C3 input rejected: {exc}")
        output = response_for_package(package, log)
        if output["repair_package_status"] == "REJECTED":
            self.last_reason = output.get("reason")
            if output["generator"] == "NONE":
                self.fallback_runs = getattr(self, "fallback_runs", 0) + 1
        errors = contracts.validate("c3", output)
        if errors:
            raise ValueError(f"C3 output contract failure: {errors}")
        return {"c3": output}
