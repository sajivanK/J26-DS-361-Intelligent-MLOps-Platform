"""Google Colab Execution Runner and Utility Library for Component 3.

Provides path-agnostic environment configuration, Google Drive mounting,
device detection (GPU/CPU fallback), checkpointing, and artifact export helpers.
Zero dependence on local Windows drive paths.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import sys
from typing import Any
import torch


def detect_execution_environment() -> dict[str, Any]:
    """Detect runtime platform, GPU availability, and Python environment."""
    is_colab = "google.colab" in sys.modules or os.path.exists("/content")
    has_gpu = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if has_gpu else "CPU"

    env_info = {
        "is_colab": is_colab,
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "has_gpu": has_gpu,
        "device_name": device_name,
        "device": "cuda" if has_gpu else "cpu",
    }
    return env_info


def setup_colab_paths(
    drive_mount_root: str = "/content/drive/MyDrive/J26-DS-361-MLOps",
    local_fallback_root: str = "./colab_workspace",
) -> dict[str, Path]:
    """Resolve workspace, dataset, and artifact export paths for Colab execution."""
    env = detect_execution_environment()

    if env["is_colab"]:
        try:
            from google.colab import drive
            if not os.path.exists("/content/drive"):
                drive.mount("/content/drive")
            base_dir = Path(drive_mount_root)
        except Exception as exc:
            print(f"Warning: Could not mount Google Drive ({exc}). Using /content/workspace.")
            base_dir = Path("/content/workspace")
    else:
        # Non-Colab fallback (Linux/macOS/Windows agnostic)
        base_dir = Path(local_fallback_root).resolve()

    paths = {
        "base_dir": base_dir,
        "data_dir": base_dir / "data",
        "models_dir": base_dir / "models",
        "checkpoints_dir": base_dir / "checkpoints",
        "results_dir": base_dir / "results",
        "exports_dir": base_dir / "exports",
    }

    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)

    return paths


class ColabCheckpointManager:
    """Manages experiment checkpointing, progress tracking, and resumption."""

    def __init__(self, checkpoint_dir: Path | str, experiment_batch_id: str) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_batch_id = experiment_batch_id
        self.state_file = self.checkpoint_dir / f"checkpoint_{experiment_batch_id}.json"
        self._completed_ids: set[str] = set()
        self._load_state()

    def _load_state(self) -> None:
        """Load completed experiment IDs from existing state file."""
        if self.state_file.is_file():
            try:
                with self.state_file.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                    self._completed_ids = set(data.get("completed_ids", []))
                    print(f"Loaded checkpoint '{self.experiment_batch_id}': {len(self._completed_ids)} experiments already completed.")
            except Exception as exc:
                print(f"Warning: Failed to parse checkpoint state file ({exc}). Starting fresh.")
                self._completed_ids = set()

    def is_completed(self, experiment_id: str) -> bool:
        """Check if an experiment was previously executed successfully."""
        return experiment_id in self._completed_ids

    def record_completed(self, experiment_id: str, summary_data: dict[str, Any] | None = None) -> None:
        """Atomically record an experiment completion to disk."""
        self._completed_ids.add(experiment_id)
        payload = {
            "batch_id": self.experiment_batch_id,
            "total_completed": len(self._completed_ids),
            "completed_ids": sorted(list(self._completed_ids)),
            "last_updated_utc": detect_execution_environment()["python_version"],
        }
        temp_file = self.state_file.with_suffix(".tmp")
        with temp_file.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        temp_file.replace(self.state_file)


def export_experiment_bundle(
    results_dir: Path | str,
    export_archive_path: Path | str,
) -> Path:
    """Package JSON evidence files and CSV datasets into a distributable zip archive."""
    import shutil
    res_path = Path(results_dir)
    out_path = Path(export_archive_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    base_name = str(out_path.with_suffix(""))
    archive_format = "zip"
    created = shutil.make_archive(base_name, archive_format, root_dir=res_path)
    print(f"Export bundle successfully created: {created}")
    return Path(created)
