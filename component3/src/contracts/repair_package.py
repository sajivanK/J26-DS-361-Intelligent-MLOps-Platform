"""Versioned, machine-readable RepairPackage contract for Component 3.

Provides complete audit trails, cryptographic artifact hashes, validation reports,
fidelity/privacy risk indicators, and handoff metadata for Component 4 review.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import Any
import uuid

import joblib
import pandas as pd


class RepairPackageStatus(str, Enum):
    """Lifecycle status of the generated repair package."""
    APPROVED_FOR_REVIEW = "APPROVED_FOR_REVIEW"
    REPAIR_APPROVED_FOR_REVIEW = "APPROVED_FOR_REVIEW"
    REJECTED = "REJECTED"
    REPAIR_REJECTED = "REJECTED"
    ABSTAINED_NO_REPAIR = "ABSTAINED_NO_REPAIR"
    NO_REPAIR = "ABSTAINED_NO_REPAIR"
    SAFE_ABSTENTION = "ABSTAINED_NO_REPAIR"
    BLOCKED = "BLOCKED"
    FAILED = "FAILED"
    EXPLICIT_ERROR = "FAILED"


def compute_sha256_bytes(data: bytes) -> str:
    """Compute SHA-256 digest of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def compute_file_sha256(filepath: Path | str) -> str:
    """Compute SHA-256 hash of a file on disk."""
    p = Path(filepath)
    if not p.is_file():
        return "MISSING"
    return hashlib.sha256(p.read_bytes()).hexdigest()


@dataclass(frozen=True)
class RepairPackage:
    """Versioned, auditable repair package contract."""
    # Identification
    package_id: str = field(default_factory=lambda: f"PKG-{uuid.uuid4().hex[:12].upper()}")
    schema_version: str = "1.0.0"
    request_id: str = ""
    run_id: str = ""
    created_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Governance & Inputs
    dataset_id: str = "telco_churn"
    model_id: str = "telco_rf_experimental_source"
    drift_summary: dict[str, Any] = field(default_factory=dict)
    diagnostic_summary: dict[str, Any] = field(default_factory=dict)

    # Strategy & Generation
    chosen_strategy: str = "NO_REPAIR"
    generator_info: dict[str, Any] = field(default_factory=dict)
    quantity_selected: int = 0

    # Artifact References & Hashes
    candidate_dataset_path: str | None = None
    candidate_dataset_sha256: str | None = None
    candidate_model_path: str | None = None
    candidate_model_sha256: str | None = None

    # Evaluation & Validation
    validation_report: dict[str, Any] = field(default_factory=dict)
    fidelity_indicators: dict[str, Any] = field(default_factory=dict)
    privacy_risk_indicators: dict[str, Any] = field(default_factory=dict)
    utility_metrics: dict[str, Any] = field(default_factory=dict)
    source_retention_metrics: dict[str, Any] = field(default_factory=dict)

    # Decisions & Limitations
    execution_status: str = "COMPLETED"
    package_status: RepairPackageStatus = RepairPackageStatus.ABSTAINED_NO_REPAIR
    rejection_reasons: tuple[str, ...] = ()
    evidence_limitations: tuple[str, ...] = (
        "Exploratory pilot evidence only; does not claim verified real-world drift recovery.",
        "Candidate labels reflect generative distribution modeling, not observational ground truth.",
        "Candidate model requires explicit human sign-off prior to production deployment.",
    )
    requires_human_approval: bool = True
    downstream_handoff_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert repair package to serializable dictionary."""
        d = asdict(self)
        d["package_status"] = self.package_status.value
        d["rejection_reasons"] = list(self.rejection_reasons)
        d["evidence_limitations"] = list(self.evidence_limitations)
        return d

    def verify_artifact_integrity(self) -> tuple[bool, list[str]]:
        """Verify SHA-256 hashes of referenced artifacts on disk."""
        errors: list[str] = []
        if self.candidate_dataset_path:
            p = Path(self.candidate_dataset_path)
            if not p.is_file():
                errors.append(f"Candidate dataset file missing: {self.candidate_dataset_path}")
            elif self.candidate_dataset_sha256:
                actual_hash = compute_file_sha256(p)
                if actual_hash != self.candidate_dataset_sha256:
                    errors.append(
                        f"Dataset hash mismatch: expected {self.candidate_dataset_sha256}, got {actual_hash}"
                    )

        if self.candidate_model_path:
            p = Path(self.candidate_model_path)
            if not p.is_file():
                errors.append(f"Candidate model file missing: {self.candidate_model_path}")
            elif self.candidate_model_sha256:
                actual_hash = compute_file_sha256(p)
                if actual_hash != self.candidate_model_sha256:
                    errors.append(
                        f"Model hash mismatch: expected {self.candidate_model_sha256}, got {actual_hash}"
                    )

        return len(errors) == 0, errors

    def compute_package_hmac(self, secret_key: bytes) -> str:
        """Compute genuine HMAC-SHA256 signature over manifest fields using secret key."""
        import hmac
        manifest_bytes = json.dumps(self.to_dict(), sort_keys=True).encode("utf-8")
        return hmac.new(secret_key, manifest_bytes, hashlib.sha256).hexdigest()

    def verify_package_hmac(self, secret_key: bytes, expected_signature: str) -> bool:
        """Verify HMAC-SHA256 signature of manifest."""
        import hmac
        actual = self.compute_package_hmac(secret_key)
        return hmac.compare_digest(actual, expected_signature)

    def save_package(
        self,
        output_directory: Path | str,
        synthetic_df: pd.DataFrame | None = None,
        candidate_model: Any = None,
        allow_overwrite: bool = True,
    ) -> Path:
        """Persist repair package JSON manifest and associated artifacts atomically."""
        out_dir = Path(output_directory)
        out_dir.mkdir(parents=True, exist_ok=True)

        dataset_path_str = self.candidate_dataset_path
        dataset_hash = self.candidate_dataset_sha256
        model_path_str = self.candidate_model_path
        model_hash = self.candidate_model_sha256

        # Save synthetic dataframe artifact if provided (atomic write)
        if synthetic_df is not None and len(synthetic_df) > 0:
            csv_path = out_dir / f"{self.package_id.lower()}_candidates_k{len(synthetic_df)}.csv"
            tmp_csv = csv_path.with_suffix(".tmp")
            synthetic_df.to_csv(tmp_csv, index=False)
            tmp_csv.replace(csv_path)
            dataset_path_str = str(csv_path)
            dataset_hash = compute_file_sha256(csv_path)
        elif dataset_path_str and Path(dataset_path_str).is_file() and not dataset_hash:
            dataset_hash = compute_file_sha256(Path(dataset_path_str))

        # Save candidate model artifact if provided (atomic write)
        if candidate_model is not None:
            m_path = out_dir / f"{self.package_id.lower()}_candidate_model.joblib"
            tmp_m = m_path.with_suffix(".tmp")
            joblib.dump(candidate_model, tmp_m)
            tmp_m.replace(m_path)
            model_path_str = str(m_path)
            model_hash = compute_file_sha256(m_path)
        elif model_path_str and Path(model_path_str).is_file() and not model_hash:
            model_hash = compute_file_sha256(Path(model_path_str))

        manifest_path = out_dir / f"{self.package_id.lower()}_manifest.json"
        if not allow_overwrite and manifest_path.exists():
            raise FileExistsError(f"Package manifest already exists at {manifest_path} and overwrite is False.")

        # Construct updated package dict with verified hashes
        pkg_dict = self.to_dict()
        pkg_dict["candidate_dataset_path"] = dataset_path_str
        pkg_dict["candidate_dataset_sha256"] = dataset_hash
        pkg_dict["candidate_model_path"] = model_path_str
        pkg_dict["candidate_model_sha256"] = model_hash

        tmp_manifest = manifest_path.with_suffix(".tmp")
        with open(tmp_manifest, "w", encoding="utf-8") as f:
            json.dump(pkg_dict, f, indent=2)
        tmp_manifest.replace(manifest_path)

        return manifest_path

