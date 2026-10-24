import numpy as np
import pandas as pd
import pytest

from c4.data import FEATURE_COLUMNS
from c4.models import evaluate_pre_release


class FixedModel:
    """Deterministic predictions for testing evaluation arithmetic."""

    classes_ = np.array([0, 1])

    def __init__(self, predictions):
        self.predictions = np.asarray(predictions)

    def predict(self, features):
        return self.predictions[
            features["age"].to_numpy(dtype=int)
        ]

    def predict_proba(self, features):
        predictions = self.predict(features)
        positive = np.where(predictions == 1, 0.8, 0.2)
        return np.column_stack([1 - positive, positive])


def example():
    frame = pd.DataFrame({
        feature: np.arange(6, dtype=float)
        for feature in FEATURE_COLUMNS
    })
    frame["target"] = [1, 0, 1, 0, 1, 0]

    champion = FixedModel([1, 1, 0, 0, 1, 0])
    candidate = FixedModel([0, 0, 1, 0, 1, 0])
    mask = pd.Series(
        [True, False, False, False, False, False],
        index=frame.index,
    )
    return frame, champion, candidate, mask


def test_global_improvement_does_not_hide_slice_harm():
    frame, champion, candidate, mask = example()
    result = evaluate_pre_release(
        champion, candidate, frame, mask
    )
    groups = result["comparisons"]

    assert groups["global"]["rows"] == 6
    assert groups["affected_slice"]["rows"] == 1
    assert groups["non_affected"]["rows"] == 5

    assert groups["global"]["candidate_minus_champion"][
        "accuracy"
    ] == pytest.approx(1 / 6)
    assert groups["affected_slice"]["candidate_minus_champion"][
        "accuracy"
    ] == pytest.approx(-1.0)
    assert groups["non_affected"]["candidate_minus_champion"][
        "accuracy"
    ] == pytest.approx(0.4)


def test_one_class_slice_has_undefined_auc():
    frame, champion, candidate, mask = example()
    result = evaluate_pre_release(
        champion, candidate, frame, mask
    )
    affected = result["comparisons"]["affected_slice"]

    assert affected["champion"]["auc"] is None
    assert affected["candidate"]["auc"] is None
    assert affected["candidate_minus_champion"]["auc"] is None


def test_misaligned_slice_mask_is_rejected():
    frame, champion, candidate, mask = example()
    mask.index = range(10, 16)

    with pytest.raises(ValueError, match="indices"):
        evaluate_pre_release(champion, candidate, frame, mask)


@pytest.mark.parametrize("membership", [False, True])
def test_empty_population_is_rejected(membership):
    frame, champion, candidate, mask = example()
    mask[:] = membership

    with pytest.raises(ValueError, match="require rows"):
        evaluate_pre_release(champion, candidate, frame, mask)


def test_invalid_model_probabilities_are_rejected():
    frame, champion, candidate, mask = example()

    class InvalidProbabilityModel(FixedModel):
        def predict_proba(self, features):
            return np.full((len(features), 2), 0.9)

    invalid_candidate = InvalidProbabilityModel(
        candidate.predictions
    )

    with pytest.raises(ValueError, match="probabilities"):
        evaluate_pre_release(
            champion, invalid_candidate, frame, mask
        )