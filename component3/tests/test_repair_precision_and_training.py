"""Boundary precision and real generator configuration regression checks."""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from c2.src.localization import HierarchicalLocalizer, _fmt
from component3.src.contracts.safe_slice_evaluator import evaluate_slice_rule
from component3.src.generators.ctgan_generator import CTGANGenerator
from component3.src.generators.tvae_generator import TVAEGenerator
from component3.tests.test_live_repair import supplied, IndependentSampler
from component3.src import live_repair as live


def test_original_rounding_failure_preserves_the_exact_boundary():
    boundary = -.4179650267872039
    data = pd.DataFrame({"relationship": [2] * 2000,
                         "education_num": [boundary] * 535 + [1.] * 877 + [-1.] * 588})
    exact = f"relationship == 2 and education_num > {_fmt(boundary)}"
    rounded = "relationship == 2 and education_num > -0.418"
    assert int(evaluate_slice_rule(data, exact).sum()) == 877
    assert int(evaluate_slice_rule(data, rounded).sum()) == 1412


def test_all_generated_localization_conditions_reproduce_their_masks():
    values = np.repeat([-.4179650267872039, -.0000123456789, .3562875791049908, 1.8181234567], 40)
    data = pd.DataFrame({"x": values, "group": np.tile([0, 1], len(values) // 2)})
    localizer = HierarchicalLocalizer(None, data, {"x": "continuous", "group": "categorical"})
    for mask, rule, _ in localizer._range_candidates(values, "x"):
        assert np.array_equal(evaluate_slice_rule(data, rule).to_numpy(), mask)
    for mask, rule, _ in localizer._refine_candidates(data, np.ones(len(data), dtype=bool), "group"):
        assert np.array_equal(evaluate_slice_rule(data, rule).to_numpy(), mask)


@pytest.mark.parametrize("generator,module,batch", [(TVAEGenerator, "tvae_generator.TVAE", 64),
                                                   (CTGANGenerator, "ctgan_generator.CTGAN", 100)])
def test_requested_batch_and_epochs_reach_actual_generator(generator, module, batch):
    with patch("component3.src.generators." + module) as implementation:
        wrapper = generator(epochs=200, batch_size=batch)
        assert implementation.call_args.kwargs["epochs"] == 200
        assert implementation.call_args.kwargs["batch_size"] == batch
        assert wrapper.batch_size == batch


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_tvae_batch_rejected(value):
    with pytest.raises(ValueError, match="batch_size"):
        TVAEGenerator(batch_size=value)


@pytest.mark.parametrize("value", [0, 64, True, 1.5])
def test_invalid_ctgan_packing_rejected(value):
    with pytest.raises(ValueError, match="batch_size"):
        CTGANGenerator(batch_size=value)


def test_bounded_pool_uses_authentic_fallback_when_only_copies_are_generated(supplied):
    payload, root = supplied
    payload["data_context"]["privacy_candidate_pool_multiplier"] = 2
    package, _ = live.execute_live_repair(payload, root / "out", generator_selector=IndependentSampler("copies"))
    assert package.package_status.value == "APPROVED_FOR_REVIEW"
    assert package.chosen_strategy == "AUTHENTIC_WEIGHTED_ADAPTATION"
    assert package.downstream_handoff_metadata["privacy_sampling"]["selected_rows"] == 0
    assert package.candidate_dataset_path.endswith("authentic_adaptation.csv")
    assert package.utility_metrics["utility_gain"] >= .01


def test_bounded_pool_discards_copies_then_revalidates_candidate(supplied):
    payload, root = supplied
    payload["data_context"]["privacy_candidate_pool_multiplier"] = 3
    class MixedSampler(IndependentSampler):
        def fit_and_sample(self, **kwargs):
            pool, metadata = super().fit_and_sample(**kwargs)
            copied = len(pool) // 3
            pool.iloc[:copied] = kwargs["training_df"].head(copied).to_numpy()
            return pool, metadata
    package, _ = live.execute_live_repair(payload, root / "out", generator_selector=MixedSampler())
    sampling = package.downstream_handoff_metadata["privacy_sampling"]
    assert sampling["raw_pool_privacy_report"]["status"] == "FAIL"
    assert sampling["discarded_too_close_rows"] > 0
    assert sampling["selected_rows"] == package.downstream_handoff_metadata["generator_configuration"]["requested_rows"]
    assert package.validation_report["privacy_report"]["status"] == "PASS"
    assert package.validation_report["fidelity_report"]
