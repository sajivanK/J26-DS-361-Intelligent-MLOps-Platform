"""
Label-free statistics of the deployed model's predictions on a batch.
J26-DS-361 | C1 Drift Detection Engine | Sajivan K (IT23172296)

CONFIDENCE (v1.2, as defined in the proposal): max class probability,
    i.e. how sure the model is about whichever class it predicts.
    conf_mean, conf_std, conf_pct_lt_06 (share of rows with max-prob < 0.6)

POSITIVE-CLASS stats (v1.1 signals, binary only, kept for comparison):
    mean_pos_prob, pos_prob_std, pct_pos_prob_lt_06
"""
import numpy as np

CONFIDENCE_SIGNALS = ["conf_mean", "conf_std", "conf_pct_lt_06"]
POS_PROB_SIGNALS = ["mean_pos_prob", "pos_prob_std", "pct_pos_prob_lt_06"]


def prediction_stats(model, X):
    proba = model.predict_proba(X)
    pmax = proba.max(axis=1)
    out = {"conf_mean": float(pmax.mean()), "conf_std": float(pmax.std()),
           "conf_pct_lt_06": float(np.mean(pmax < 0.6))}
    if proba.shape[1] == 2:
        p = proba[:, 1]
        out.update({"mean_pos_prob": float(p.mean()), "pos_prob_std": float(p.std()),
                    "pct_pos_prob_lt_06": float(np.mean(p < 0.6))})
    return out
