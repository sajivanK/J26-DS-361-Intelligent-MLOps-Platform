"""Classifier two-sample test (C2ST): XGBoost separates reference rows from
batch rows; 3-fold CV AUC (0.5 = no drift).

Kept identical to the v1.1 benchmark detector (same subsample, seeds and
hyperparameters) so its scores can be checked against v1.1.
Trees split on label-encoded categoricals without assuming a numeric meaning
beyond ordering, which is acceptable for a separability test.
"""
import numpy as np
import pandas as pd

CLASSIFIER_SUBSAMPLE = 1000


def clf_detector(profile, batch, seed=0):
    from xgboost import XGBClassifier
    from sklearn.model_selection import cross_val_score
    ref = profile.ref[profile.features]
    cur = batch[profile.features]
    n = min(CLASSIFIER_SUBSAMPLE, len(ref), len(cur))
    X = pd.concat([ref.sample(n=n, random_state=seed),
                   cur.sample(n=n, random_state=seed + 1)], ignore_index=True)
    y = np.array([0] * n + [1] * n)
    clf = XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1,
                        eval_metric="logloss", random_state=42, verbosity=0)
    return {"clf_auc": float(np.mean(cross_val_score(clf, X, y, cv=3,
                                                       scoring="roc_auc", n_jobs=-1)))}
