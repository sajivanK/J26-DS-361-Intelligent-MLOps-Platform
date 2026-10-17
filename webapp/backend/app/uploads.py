"""Reading an uploaded CSV batch (shared by the C1 and pipeline endpoints)."""
import io

import pandas as pd
from fastapi import HTTPException, UploadFile

from .settings import DATASETS, LABEL_COLUMNS, MAX_UPLOAD_MB


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
    removed = [c for c in LABEL_COLUMNS if c in batch.columns]
    batch = batch.drop(columns=removed)
    context = {"dataset": dataset, "model_id": DATASETS[dataset]["model_id"],
               "current_dataset_uri": file.filename or ""}
    return batch, context, removed
