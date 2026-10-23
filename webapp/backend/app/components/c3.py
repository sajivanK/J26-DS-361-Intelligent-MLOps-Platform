"""Platform adapter for artifact-bound C3 repair; no dataset substitution."""
from typing import Any

import pandas as pd

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
        from component3.src.live_repair import execute_live_repair
        self.loaded = True

    def details(self):
        return {"supported_generators": ["TVAE", "CTGAN", "TabDDPM"],
                "supported_tasks": ["tabular_binary_classification"],
                "decision_modes": ["RULE_BASED"], "version": "2.0.0", "contract_version": "1.1"}

    def run(self, batch: pd.DataFrame, upstream: dict[str, Any]) -> dict[str, Any]:
        from component3.src.live_repair import execute_live_repair, response_for_package, artifact_path
        from component3.src.contracts.external_contracts import ContractValidationError

        context = {**upstream.get("context", {}), **upstream.get("data_context", {})}
        context = {**c3_data_context(context.get("model_id", "")), **context}
        context.setdefault("dataset_id", context.get("dataset"))
        if upstream.get("component_sources", {}).get("c2") == "fake":
            raise ContractValidationError("C2 is still a placeholder. Connect real C2 or submit actual C1/C2 evidence to /api/c3/repair.")
        current_uri = context.get("current_dataset_uri")
        if not current_uri:
            raise ContractValidationError("C3 requires the actual persisted current_dataset_uri and complete data_context.")
        path = artifact_path(current_uri, "datasets")
        actual = pd.read_csv(path)
        columns = [c for c in batch.columns if c != context.get("target_column")]
        if not set(columns).issubset(actual.columns):
            raise ContractValidationError("Current dataset does not contain the monitored batch's columns.")
        try:
            pd.testing.assert_frame_equal(batch[columns].reset_index(drop=True),
                actual[columns].reset_index(drop=True), check_dtype=False, check_exact=False)
        except AssertionError as exc:
            raise ContractValidationError("current_dataset_uri does not represent the actual monitored batch.") from exc
        payload = {"c1_evidence": upstream.get("c1"),
                   "c2_diagnosis": upstream.get("c2_to_c3", upstream.get("c2")),
                   "data_context": context}
        package, log = execute_live_repair(payload)
        output = response_for_package(package, log)
        errors = contracts.validate("c3", output)
        if errors:
            raise ValueError(f"C3 output contract failure: {errors}")
        return {"c3": output}
