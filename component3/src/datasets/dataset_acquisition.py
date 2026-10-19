"""Dataset Acquisition and Provenance Manager for Component 3.

Provides automated download for verified open-access benchmarks (Telco Churn, Bank Marketing),
and structured manual instructions for competition datasets requiring credentials (Home Credit, IEEE-CIS).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
import urllib.request
import pandas as pd

from component3.src.datasets.dataset_registry import DatasetMetadata, DatasetRegistry, DatasetStatus, get_dataset_registry
from component3.src.datasets.dataset_verifier import DatasetVerificationReport, DatasetVerifier

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RAW_DIR = ROOT / "component3" / "data" / "raw"

# Authoritative public open-access URLs for non-credentialed datasets
PUBLIC_DATASET_URLS: dict[str, str] = {
    "telco_churn": "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv",
    "bank_marketing": "https://raw.githubusercontent.com/madmashup/targeted-marketing-predictive-engine/master/banking.csv",
}


class DatasetAcquisitionManager:
    """Manages downloading, caching, and verifying candidate datasets."""

    def __init__(self, raw_data_dir: Path | str | None = None, registry: DatasetRegistry | None = None) -> None:
        self.raw_data_dir = Path(raw_data_dir) if raw_data_dir else DEFAULT_RAW_DIR
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.registry = registry or get_dataset_registry()
        self.verifier = DatasetVerifier(registry=self.registry)

    def acquire_dataset(
        self,
        dataset_id: str,
        force_download: bool = False,
    ) -> tuple[Path, DatasetVerificationReport]:
        """Acquire dataset if available, verify schema and integrity, and update registry."""
        meta = self.registry.get(dataset_id)

        # 1. Check if already verified local asset (like UCI Adult)
        if meta.dataset_id == "uci_adult":
            p = meta.get_absolute_path()
            if p and p.is_file():
                report = self.verifier.verify(dataset_id, filepath=p)
                return p, report

        # Target local file path
        target_path = self.raw_data_dir / f"{dataset_id}.csv"

        # 2. Download from public authoritative URL if available
        if dataset_id in PUBLIC_DATASET_URLS:
            if not target_path.is_file() or force_download:
                url = PUBLIC_DATASET_URLS[dataset_id]
                logger.info("Downloading %s from %s...", dataset_id, url)
                urllib.request.urlretrieve(url, target_path)

            # Verify downloaded asset
            report = self.verifier.verify(dataset_id, filepath=target_path)
            if report.is_valid:
                # Dynamically update registry metadata status to VERIFIED
                meta.status = DatasetStatus.VERIFIED
                meta.local_path = str(target_path.relative_to(ROOT)).replace("\\", "/")
            return target_path, report

        # 3. Datasets requiring manual credentials (Home Credit, IEEE-CIS)
        if target_path.is_file():
            report = self.verifier.verify(dataset_id, filepath=target_path)
            if report.is_valid:
                meta.status = DatasetStatus.VERIFIED
                meta.local_path = str(target_path.relative_to(ROOT)).replace("\\", "/")
            return target_path, report

        # If file is missing, provide explicit acquisition instructions
        raise FileNotFoundError(
            f"Dataset '{dataset_id}' ({meta.display_name}) requires authenticated access or manual acquisition.\n"
            f"Source: {meta.source}\n"
            f"Expected file location: '{target_path}'.\n"
            f"Please consult 'component3/docs/dataset_acquisition_guide.md' for download instructions."
        )

    def get_dataset_status(self, dataset_id: str) -> dict[str, Any]:
        """Check status of dataset in local repository without downloading."""
        meta = self.registry.get(dataset_id)
        target_path = self.raw_data_dir / f"{dataset_id}.csv"
        if meta.dataset_id == "uci_adult":
            p = meta.get_absolute_path()
            if p and p.is_file():
                return {"dataset_id": dataset_id, "status": "verified", "available": True, "path": str(p)}
        if target_path.is_file():
            return {"dataset_id": dataset_id, "status": "downloaded", "available": True, "path": str(target_path)}
        if meta.status == DatasetStatus.PENDING_DOWNLOAD:
            return {
                "dataset_id": dataset_id,
                "status": "pending",
                "available": False,
                "message": f"Requires manual download or Kaggle API token ({meta.source})",
            }
        return {"dataset_id": dataset_id, "status": "missing", "available": False, "message": "File not found locally"}

