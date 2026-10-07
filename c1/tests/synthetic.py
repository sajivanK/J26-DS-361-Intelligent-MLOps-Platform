"""
Synthetic stand-in for the UCI Adult artifacts, used only to test C1 code.
Shape mimics Adult: 5 continuous (one point-mass, like capital_gain) and
7 label-encoded categoricals (one with 41 codes, like native_country).
"""
import json
import os

import joblib
import numpy as np
import pandas as pd

CONT = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
CAT = {"workclass": 7, "marital_status": 7, "occupation": 14, "relationship": 6,
       "race": 5, "sex": 2, "native_country": 41}


def make_frame(n, rng):
    df = pd.DataFrame({
        "age": rng.normal(size=n),
        "education_num": rng.normal(size=n),
        "capital_gain": np.where(rng.random(n) < 0.92, -0.15, rng.exponential(3, n)),
        "capital_loss": np.where(rng.random(n) < 0.95, -0.2, rng.exponential(2, n)),
        "hours_per_week": rng.normal(size=n),
    })
    for c, k in CAT.items():
        p = rng.dirichlet(np.ones(k) * 0.7)
        df[c] = rng.choice(k, size=n, p=p)
    logit = (1.2 * df.age + 1.0 * df.education_num + 0.8 * df.capital_gain
             + 0.6 * (df.marital_status == 2) + 0.5 * df.relationship / 5
             - 1.5 + rng.normal(scale=0.8, size=n))
    df["target"] = (logit > 0).astype(int)
    return df


def build(root, seed=0):
    from xgboost import XGBClassifier
    os.makedirs(root, exist_ok=True)
    rng = np.random.default_rng(seed)
    full = make_frame(10000, rng)
    ref, test = full.iloc[:6000].reset_index(drop=True), full.iloc[6000:].reset_index(drop=True)
    ref.to_csv(f"{root}/reference.csv", index=False)
    test.to_csv(f"{root}/test.csv", index=False)
    model = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                          random_state=42, verbosity=0)
    model.fit(ref.drop(columns="target"), ref["target"])
    joblib.dump(model, f"{root}/base_model_adult.pkl")
    ft = {**{c: "continuous" for c in CONT}, **{c: "categorical" for c in CAT}}
    imp = {"feature_types": ft, "feature_groups": {
        "high_importance": {"features": ["marital_status", "age", "relationship",
                                         "education_num", "capital_gain"]},
        "low_importance": {"features": ["capital_loss", "sex", "workclass",
                                        "race", "native_country"]}}}
    json.dump(imp, open(f"{root}/feature_importance.json", "w"), indent=2)
    return ref, test, model, ft
