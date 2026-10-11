"""Paired pre-release evaluation of champion and candidate."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)

from c4.data import FEATURE_COLUMNS, TARGET_COLUMN


def _model_outputs(model, features):
    if list(model.classes_) != [0, 1]:
        raise ValueError("Model classes must be ordered [0, 1].")

    predictions = np.asarray(model.predict(features))
    probabilities = np.asarray(
        model.predict_proba(features), dtype=float
    )

    if (
        predictions.shape != (len(features),)
        or not np.isin(predictions, [0, 1]).all()
    ):
        raise ValueError("Model produced invalid binary predictions.")

    if probabilities.shape != (len(features), 2):
        raise ValueError("Model probabilities have an invalid shape.")

    if (
        not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or (probabilities > 1).any()
        or not np.allclose(probabilities.sum(axis=1), 1.0)
    ):
        raise ValueError("Model produced invalid probabilities.")

    return predictions, probabilities[:, 1]


def _metrics(target, predictions, positive_probabilities):
    auc = None

    if len(np.unique(target)) == 2:
        auc = float(roc_auc_score(target, positive_probabilities))

    return {
        "accuracy": float(accuracy_score(target, predictions)),
        "macro_f1": float(
            f1_score(
                target,
                predictions,
                labels=[0, 1],
                average="macro",
                zero_division=0,
            )
        ),
        # AUC is undefined when the population has one target class.
        "auc": auc,
    }


def evaluate_pre_release(
    champion,
    candidate,
    release_validation: pd.DataFrame,
    affected_mask: pd.Series,
) -> dict[str, object]:
    """Compare both models on identical release-validation rows.

    affected_mask must be computed on clean rows before injection
    and preserved through the scenario transformation.
    """

    expected_columns = [*FEATURE_COLUMNS, TARGET_COLUMN]

    if release_validation.columns.tolist() != expected_columns:
        raise ValueError("Release-validation schema is invalid.")

    if release_validation.empty:
        raise ValueError("Release-validation dataset is empty.")

    if not release_validation.index.is_unique:
        raise ValueError("Release-validation indices must be unique.")

    if not all(
        pd.api.types.is_numeric_dtype(release_validation[column])
        for column in expected_columns
    ):
        raise ValueError("Release-validation columns must be numeric.")

    if not np.isfinite(
        release_validation[expected_columns].to_numpy(dtype=float)
    ).all():
        raise ValueError("Release-validation contains nonfinite values.")

    if not release_validation[TARGET_COLUMN].isin([0, 1]).all():
        raise ValueError("Release-validation targets must be binary.")

    if not isinstance(affected_mask, pd.Series):
        raise ValueError("Affected mask must be a pandas Series.")

    if not affected_mask.index.equals(release_validation.index):
        raise ValueError("Affected mask indices must match evaluation rows.")

    if (
        not pd.api.types.is_bool_dtype(affected_mask.dtype)
        or affected_mask.isna().any()
    ):
        raise ValueError("Affected mask must contain nonmissing booleans.")

    affected = affected_mask.to_numpy(dtype=bool)

    if not affected.any() or affected.all():
        raise ValueError(
            "Both affected and non-affected populations require rows."
        )

    features = release_validation.loc[:, list(FEATURE_COLUMNS)]
    target = release_validation[TARGET_COLUMN].to_numpy()

    champion_predictions, champion_probabilities = _model_outputs(
        champion, features
    )
    candidate_predictions, candidate_probabilities = _model_outputs(
        candidate, features
    )

    populations = {
        "global": np.ones(len(target), dtype=bool),
        "affected_slice": affected,
        "non_affected": ~affected,
    }

    comparisons = {}

    for name, mask in populations.items():
        champion_metrics = _metrics(
            target[mask],
            champion_predictions[mask],
            champion_probabilities[mask],
        )
        candidate_metrics = _metrics(
            target[mask],
            candidate_predictions[mask],
            candidate_probabilities[mask],
        )

        margins = {}

        for metric in ("accuracy", "macro_f1", "auc"):
            before = champion_metrics[metric]
            after = candidate_metrics[metric]
            margins[metric] = (
                None
                if before is None or after is None
                else after - before
            )

        comparisons[name] = {
            "rows": int(mask.sum()),
            "target_counts": {
                "0": int((target[mask] == 0).sum()),
                "1": int((target[mask] == 1).sum()),
            },
            "champion": champion_metrics,
            "candidate": candidate_metrics,
            "candidate_minus_champion": margins,
        }

    return {
        "evidence_schema_version": "1.0",
        "evaluation_role": "release_validation",
        "primary_margin_metric": "accuracy",
        "margin_units": "fraction",
        "comparisons": comparisons,
    }
