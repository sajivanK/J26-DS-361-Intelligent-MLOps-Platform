"""Repair Knowledge Base for Component 3.

Provides auditable querying of empirical evidence across Batches 5I-5L,
including baseline metrics, matched-budget authentic controls, and additive records.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EVIDENCE_PATH = REPO_ROOT / "component3" / "results" / "repair_evidence_registry.json"


class RepairKnowledgeBase:
    """Interface to historical empirical repair evidence and benchmark performance."""

    def __init__(self, registry_path: Path | str | None = None) -> None:
        self.registry_path = Path(registry_path or DEFAULT_EVIDENCE_PATH)
        self._data: dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        if self.registry_path.is_file():
            try:
                with open(self.registry_path, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
            except Exception as exc:
                logger.error("Failed to load repair evidence registry from %s: %s", self.registry_path, exc)
                self._data = {"registry_metadata": {}, "experiments": {}}
        else:
            self._data = {"registry_metadata": {}, "experiments": {}}

    @property
    def metadata(self) -> dict[str, Any]:
        return self._data.get("registry_metadata", {})

    @property
    def experiments(self) -> dict[str, Any]:
        return self._data.get("experiments", {})

    def get_source_eval_count(self) -> int:
        """Audited source evaluation set size (493 rows for Telco Churn)."""
        return int(self.metadata.get("source_eval_count", 493))

    def get_dev_val_count(self) -> int:
        """Audited development validation set size (292 rows for Telco Churn)."""
        return int(self.metadata.get("dev_val_count", 292))

    def get_experiment(self, exp_id: str) -> dict[str, Any] | None:
        """Lookup an experiment record by ID."""
        return self.experiments.get(exp_id)

    def lookup_baseline_f1(self, dataset_id: str = "telco_churn") -> float:
        """Lookup baseline positive F1 on dev_val (0.4055 for telco_churn)."""
        b0 = self.experiments.get("EXP_5I_B0")
        if b0 and "target_dev_performance" in b0:
            return float(b0["target_dev_performance"].get("positive_f1", 0.4055))
        return 0.4055

    def lookup_matched_authentic_control(self, budget_size: int) -> dict[str, Any] | None:
        """Retrieve matched authentic control performance for a given budget."""
        arm_mapping = {
            50: "EXP_5I_B1_B50",
            100: "EXP_5I_B1_B100",
            200: "EXP_5I_B1_B200",
        }
        exp_id = arm_mapping.get(budget_size)
        return self.get_experiment(exp_id) if exp_id else None

    def add_evidence_record(self, record_id: str, record_data: dict[str, Any]) -> None:
        """Additively append a new experiment record without overwriting historical entries."""
        if "experiments" not in self._data:
            self._data["experiments"] = {}
        if record_id in self._data["experiments"]:
            raise ValueError(f"Cannot overwrite historical experiment record '{record_id}'. Use a unique record_id.")
        self._data["experiments"][record_id] = record_data
        self.metadata["total_experiments_recorded"] = len(self._data["experiments"])

    def save(self, target_path: Path | str | None = None) -> Path:
        """Safely save evidence registry to disk."""
        out_path = Path(target_path or self.registry_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2)
        tmp_path.replace(out_path)
        return out_path


_GLOBAL_KNOWLEDGE_BASE: RepairKnowledgeBase | None = None


def get_repair_knowledge_base() -> RepairKnowledgeBase:
    """Retrieve global RepairKnowledgeBase singleton."""
    global _GLOBAL_KNOWLEDGE_BASE
    if _GLOBAL_KNOWLEDGE_BASE is None:
        _GLOBAL_KNOWLEDGE_BASE = RepairKnowledgeBase()
    return _GLOBAL_KNOWLEDGE_BASE
