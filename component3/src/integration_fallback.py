"""Dependency-free, honest no-repair output for the platform adapter."""
from datetime import datetime, timezone
import uuid


def rejected_output(reason):
    return {
        "repair_dataset_uri": "NONE", "repair_package_status": "REJECTED",
        "generator": "NONE", "n_synthetic_rows": 0, "augmentation_ratio": 0.0,
        "fidelity_score": 0.0, "utility_score": 0.0, "privacy_status": "FAIL",
        "target_coverage": 0.0, "prescription_confidence": 0.0,
        "repair_version": "c3-" + uuid.uuid4().hex,
        "repair_timestamp": datetime.now(timezone.utc).isoformat(),
        "reason": reason, "status": "BLOCKED", "fallback": True,
    }
