"""
C1 predicted harm only: Novelty 1's label-free harm prediction for a batch, without the interval,
drift description or any other output field.
J26-DS-361 | C1 Drift Detection Engine | Sajivan K (IT23172296)

Why: C2's label-free Drift Impact Score asks C1 a counterfactual question per feature,
    harm(batch) - harm(batch with feature f restored to its reference distribution),
which needs many fast harm predictions (agreed C2 label-free plan, 23 Oct 2026).

It uses exactly the same N1 inputs as C1Engine.detect() (9 detector signals, 6 prediction stats,
batch size clipped to the calibrated range), so predict_harm(engine, X) equals
detect(X)["predicted_harm"] * 100 for the same seed. Engine 1.2.3 is not changed.

Units: percentage points of the engine's harm metric (accuracy drop for UCI Adult, AUC drop for
IEEE-CIS), the unit C2 uses. C1's contract output reports the same number as a proportion.

Usage:
    from c1.src.engine_io import load_engine
    from c1.src.harm_only import predict_harm, make_harm_fn
    eng = load_engine(r"C:\\J26\\data\\uci\\portable")
    predict_harm(eng, batch_df)                  # e.g. 4.21
    harm_fn = make_harm_fn(eng)                  # callable df -> pp, for C2's LabelFreeImpactScorer
"""
import numpy as np

from c1.src.detectors import DETECTOR_SIGNALS, run_all_detectors
from c1.src.engine import MIN_ROWS, PRED
from c1.src.prediction_stats import prediction_stats


def predict_harm(engine, batch_df, seed=0):
    """C1's predicted harm for one batch, in percentage points. No labels are used."""
    missing = [f for f in engine.features if f not in batch_df.columns]
    if missing:
        raise ValueError(f"batch is missing features: {missing}")
    X = batch_df[engine.features]
    if len(X) < MIN_ROWS:
        raise ValueError(f"batch has {len(X)} rows; at least {MIN_ROWS} are needed")
    if X.isna().any().any():
        raise ValueError("batch contains NaN values")
    bs = float(min(max(len(X), engine.sizes[0]), engine.sizes[-1]))
    scores, _ = run_all_detectors(engine.profile, X, seed=seed)
    ps = prediction_stats(engine.base_model, X)
    x1 = np.array([[scores[k] for k in DETECTOR_SIGNALS] + [ps[k] for k in PRED] + [bs]])
    return float(engine.n1.predict(x1)[0])


def make_harm_fn(engine, seed=0, lock=None):
    """A callable df -> predicted harm (pp), named for C2's reports. Pass a lock if the same engine
    object is also used by other threads (the platform serialises calls per engine)."""
    def harm(df):
        if lock is None:
            return predict_harm(engine, df, seed=seed)
        with lock:
            return predict_harm(engine, df, seed=seed)
    harm.name = "C1 predict_harm (engine 1.2.3)"
    return harm
