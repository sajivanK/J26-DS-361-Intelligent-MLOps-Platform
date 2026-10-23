"""Reading an uploaded CSV batch (shared by the C1 and pipeline endpoints)."""
import io
import uuid
from pathlib import Path

import pandas as pd
from fastapi import HTTPException, UploadFile

from .settings import DATASETS, LABEL_COLUMNS, MAX_UPLOAD_MB, REPO_ROOT, c3_data_context


def read_batch(file: UploadFile, dataset: str):
    """Return (batch without labels, context dict, removed label columns)."""
    if dataset not in DATASETS:
        raise HTTPException(404, f"unknown dataset '{dataset}'; use one of {list(DATASETS)}")
    raw = file.file.read()
    if len(raw) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"file is larger than {MAX_UPLOAD_MB} MB")
    try:
        batch = pd.read_csv(io.BytesIO(raw))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"could not read CSV: {e}")
    onboarding = c3_data_context(DATASETS[dataset]["model_id"])
    target = onboarding.get("target_column", "target")
    removed = [c for c in set(LABEL_COLUMNS + (target,)) if c in batch.columns]
    # Persist the labeled artifact for C3; C1/C2 receive only the feature view.
    directory = REPO_ROOT / "component3/data/integration_uploads"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"batch_{uuid.uuid4().hex}.csv"
    path.write_bytes(raw)
    batch = batch.drop(columns=removed)
    context = {**onboarding, "dataset": dataset, "dataset_id": onboarding.get("dataset_id", dataset),
               "model_id": DATASETS[dataset]["model_id"], "current_dataset_uri": str(path),
               "label_status": "AVAILABLE" if target in removed else "UNAVAILABLE"}
    return batch, context, removed
