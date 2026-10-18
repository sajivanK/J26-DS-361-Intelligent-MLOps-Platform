"""
Drift Lab: build a batch with a KNOWN drift and measure its TRUE harm, so C1's answer can be checked.
J26-DS-361 | Intelligent MLOps Platform | Sajivan K (lead)

How a lab batch is made (the same recipe as the C1 fresh test, c1/scripts/retest_engine_fresh.py):
  1. sample `batch_size` rows from unseen labelled data (test.csv), seeded
  2. inject the drift with the shared injector (shared/injection/inject_drift.py)
  3. true harm = fixed baseline (the model on the whole test set) minus the model on the drifted rows:
       UCI  accuracy drop in pp,  IEEE-CIS  AUC drop in pp   (the metric C1 predicts)
     The paired harm (same rows before vs after injection) is returned too.
The labels stay here: the components only ever receive the batch without the target column.
"""
import json
import sys
import threading
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

from .settings import LAB_FEATURE_INFO, LAB_SOURCE, REPO_ROOT, TARGET_COL

sys.path.insert(0, str(REPO_ROOT / "shared"))
from injection.inject_drift import inject_drift  # noqa: E402

DRIFT_TYPES = ("none", "marginal_single", "marginal_multi", "interaction", "combined")
# how many features each drift type takes: (min, max)
N_FEATURES = {"none": (0, 0), "marginal_single": (1, 1), "marginal_multi": (2, 5),
              "interaction": (2, 2), "combined": (3, 3)}


class LabError(ValueError):
    """Bad request from the user (wrong feature count, unknown feature, ...)."""


class DriftLab:
    def __init__(self):
        self._source, self._info = {}, {}
        self._lock = threading.Lock()

    # ── data, loaded on first use ───────────────────────────────────────────
    def source(self, dataset):
        with self._lock:
            if dataset not in self._source:
                t0 = time.time()
                df = pd.read_csv(LAB_SOURCE[dataset])
                if TARGET_COL not in df.columns:
                    raise RuntimeError(f"{LAB_SOURCE[dataset]} has no '{TARGET_COL}' column")
                self._source[dataset] = df
                print(f"[lab] {dataset}: {len(df):,} labelled rows loaded in {time.time() - t0:.1f} s", flush=True)
            return self._source[dataset]

    def feature_info(self, dataset, engine):
        """Feature groups for the lab: high / low importance (benchmark) and the rest, ranked."""
        if dataset in self._info:
            return self._info[dataset]
        high, low, ranked, note = [], [], [], None
        path = Path(LAB_FEATURE_INFO[dataset])
        if path.exists():
            d = json.loads(path.read_text(encoding="utf-8"))
            groups = d.get("feature_groups", {})
            high = list(groups.get("high_importance", {}).get("features", []))
            low = list(groups.get("low_importance", {}).get("features", []))
            rank = d.get("shap_importance_ranked") or d.get("importance_ranked") or {}
            ranked = list(rank) if isinstance(rank, (dict, list)) else []
        else:
            note = f"feature importance file not found ({path}); all features listed as 'other'"
        feats = list(engine.features)
        high = [f for f in high if f in feats]
        low = [f for f in low if f in feats]
        used = set(high) | set(low)
        # features never drifted in calibration, most important first
        other = [f for f in ranked if f in feats and f not in used]
        other += [f for f in feats if f not in used and f not in other]
        cat = set(engine.profile.categorical)
        info = {"high": high, "low": low, "other": other,
                "types": {f: ("categorical" if f in cat else "continuous") for f in feats},
                "note": note}
        self._info[dataset] = info
        return info

    # ── one lab batch ───────────────────────────────────────────────────────
    def check(self, engine, dataset, drift_type, features, magnitude, batch_size, seed):
        if drift_type not in DRIFT_TYPES:
            raise LabError(f"drift_type must be one of {DRIFT_TYPES}")
        lo, hi = N_FEATURES[drift_type]
        if not lo <= len(features) <= hi:
            raise LabError(f"{drift_type} needs {lo if lo == hi else f'{lo} to {hi}'} features, got {len(features)}")
        if len(set(features)) != len(features):
            raise LabError("each feature can be picked once")
        unknown = [f for f in features if f not in engine.features]
        if unknown:
            raise LabError(f"unknown features: {unknown}")
        if drift_type != "none" and not 0.05 <= magnitude <= 1.0:
            raise LabError("strength must be between 0.05 and 1.0")
        n = len(self.source(dataset))
        if not 100 <= batch_size <= n:
            raise LabError(f"batch size must be between 100 and {n}")

    def make_batch(self, engine, dataset, drift_type, features, magnitude, batch_size, seed):
        """Return (batch without labels, truth dict, injection dict, batch WITH labels)."""
        self.check(engine, dataset, drift_type, features, magnitude, batch_size, seed)
        src = self.source(dataset)
        types = self.feature_info(dataset, engine)["types"]
        kw = {"features": [], "interaction_pairs": None}
        if drift_type in ("marginal_single", "marginal_multi", "interaction"):
            kw["features"] = list(features)
        elif drift_type == "combined":     # first feature shifts, the other two get an interaction break
            kw = {"features": [features[0]], "interaction_pairs": [(features[1], features[2])]}
        mag = 0.0 if drift_type == "none" else float(magnitude)

        t0 = time.time()
        res = inject_drift(reference_df=src, model=engine.base_model, drift_type=drift_type,
                           magnitude=mag, feature_types={f: types[f] for f in features},
                           batch_size=int(batch_size), target_col=TARGET_COL, seed=int(seed),
                           source_df=src, paired=True, **kw)
        gt, drifted = res["ground_truth"], res["drifted_df"]
        X = drifted.drop(columns=[TARGET_COL])

        metric = getattr(engine, "harm_metric", "accuracy_drop")
        if metric == "auc_drop":
            auc_b = roc_auc_score(drifted[TARGET_COL].values, engine.base_model.predict_proba(X)[:, 1])
            fixed, paired = (engine.auc - auc_b) * 100, (gt["baseline_auc"] - auc_b) * 100
            base, after, before = engine.auc, auc_b, gt["baseline_auc"]
        else:
            fixed, paired = (engine.p - gt["drifted_accuracy"]) * 100, gt["actual_accuracy_drop"]
            base, after, before = engine.p, gt["drifted_accuracy"], gt["baseline_accuracy"]
        truth = {"harm_metric": metric, "true_harm_pp": round(float(fixed), 4),
                 "true_harm_paired_pp": round(float(paired), 4),
                 "baseline": round(float(base), 4), "on_batch": round(float(after), 4),
                 "same_rows_before": round(float(before), 4),
                 "note": "fixed baseline (whole test set) minus the model on this batch; labels never reach the components"}
        injection = {"dataset": dataset, "drift_type": drift_type, "features": list(features),
                     "shifted": kw["features"] if drift_type != "interaction" else [],
                     "interaction_pairs": ([list(features)] if drift_type == "interaction" else
                                           [list(p) for p in (kw["interaction_pairs"] or [])]),
                     "magnitude": mag, "batch_size": len(X), "seed": int(seed),
                     "inject_sec": round(time.time() - t0, 3)}
        return X, truth, injection, drifted


LAB = DriftLab()
