"""Persist actual C3 stage events for polling, including rejected/blocked runs."""
import json
import os
import threading
from datetime import datetime, timezone

STAGES = (
    "Input Validation", "Repair Readiness", "Slice Extraction", "Generator Prescription",
    "Repair Quantity Calculation", "Generator Training", "Synthetic Data Generation",
    "Structural Validation", "Target Coverage Validation", "Fidelity Validation",
    "Privacy Validation", "Utility Evaluation", "Final Decision", "C4 Package Preparation",
)


class Progress:
    def __init__(self, directory, run_id, context):
        self.directory = directory
        self.data = {"run_id": run_id, "pipeline_run_id": context.get("run_id"),
                     "model_id": context.get("model_id"), "dataset_id": context.get("dataset_id"),
                     "batch_id": context.get("batch_id"), "status": "RUNNING",
                     "stages": [{"name": name, "status": "Waiting"} for name in STAGES]}

    def save(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        self.data["updated_at"] = datetime.now(timezone.utc).isoformat()
        temporary = self.directory / f"progress-{threading.get_ident()}.tmp"
        temporary.write_text(json.dumps(self.data, default=str), encoding="utf-8")
        os.replace(temporary, self.directory / "progress.json")

    def stage(self, number, status="Completed", detail=None):
        item = self.data["stages"][number - 1]
        item.update(status=status, timestamp=datetime.now(timezone.utc).isoformat())
        if detail is not None:
            item["detail"] = detail
        self.save()

    def finish(self, status, reason):
        self.data.update(status=status, reason=reason)
        for item in self.data["stages"]:
            if item["status"] == "Running":
                item["status"] = "Failed" if status in ("FAILED", "REJECTED") else "Blocked"
        self.save()
