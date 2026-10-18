"""
C1 Drift Detection Engine: train once, then detect() on production batches.
J26-DS-361 | C1 Drift Detection Engine | Sajivan K (IT23172296)

TRAIN (once, from the v1.2 benchmark)
  N1  harm predictor: GBR on 9 detector signals + 6 prediction stats + batch_size,
      trained on 80% of condition groups (first StratifiedGroupKFold split, seed 0).
  N2  split conformal, batch-size normalised: q = conformal quantile of
      |y - yhat| / sigma(bs) on the other 20% (calibration groups, never trained on).
  N3  Option B: RandomForest for marginal_extent (none / single / multi) and
      RandomForest for interaction_suspected (probability), both on the
      feature-agnostic fingerprint, trained on all 1,600 conditions.
  Thresholds: no-drift 95th percentile per calibrated batch size, for the six
      detectors and for every per-feature KS / Chi2 / PSI score.

HARM METRIC (1.2.2)
  accuracy_drop (UCI Adult): sigma(bs) = binomial sampling sd of batch accuracy (finite-population
      corrected), as in 1.2.1.
  auc_drop (IEEE-CIS): sigma(bs) = MEASURED sd of clean-batch AUC at the calibrated batch sizes
      (benchmark config), interpolated on a log-log scale between them.
  The metric is read from the benchmark CSV ("harm_metric" column; missing = accuracy_drop).
  1.2.2 also uses the seeded LSDD detector (c1/src/detectors/lsdd_detector.py).

AFFECTED FEATURES (1.2.2)
  With many features, testing each one at its no-drift 95th percentile lists about 5% of them by
  chance (17 of 199 on clean IEEE-CIS batches). 1.2.2 lists a feature only when EVERY detector that
  scores it (KS or Chi2, and PSI) exceeds a strict threshold: the maximum score seen on no-drift
  benchmark batches at that batch size, floored at 10% of the family's median 95th-percentile
  threshold (guards near-constant features whose threshold is 0). Ranked by score / threshold.
  Chosen on both benchmarks: IEEE-CIS precision 0.09 -> 0.68 (20.4 -> 3.1 features listed), recall
  0.994 -> 0.986; UCI Adult precision 0.64 -> 0.98, recall 1.000 -> 0.987. The 95th-percentile
  thresholds used by N3 are unchanged, so N1 / N2 / N3 outputs are unchanged.

N2 MONDRIAN (1.2.3)
  Split conformal guarantees coverage only marginally over the condition mix; marginal_multi was
  under-covered on both benchmarks (0.87 to 0.92). 1.2.3 calibrates one normalised quantile per
  N3-predicted marginal_extent (none / single / multi), which detect() knows before the interval is
  built. The calibration rows are grouped by OUT-OF-FOLD N3 predictions (group 5-fold on all
  1,600 conditions), because N3 itself is trained on all rows and its in-sample predictions would be
  optimistic. A group with fewer than MIN_GROUP calibration rows uses the global quantile.
  Validated by c1/scripts/run_mondrian_check_v12.py (100 splits, both datasets): multi coverage
  0.96 to 0.97, worst drift-type coverage 0.93 to 0.96, intervals 10 to 20% wider on average.
  N1, N3, thresholds, affected_features and all output fields are unchanged; only harm_interval
  changes. Engines saved before 1.2.3 keep the single global quantile.

DETECT (per batch, no labels needed)
  Returns the C1 output dict (see detect()). Output changes in 1.2.1:
    - drift_detected (new): marginal shift found by N3, or at least 3 detectors fired
    - harm_severity (renamed from drift_severity): based on predicted harm only
    - affected_features: empty unless N3 finds a marginal shift
    - drift_type removed (use marginal_extent + interaction_suspected)
"""
import time
import uuid

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from c1.src.profile import ReferenceProfile
from c1.src.detectors import run_all_detectors, DETECTOR_SIGNALS
from c1.src.prediction_stats import prediction_stats, CONFIDENCE_SIGNALS, POS_PROB_SIGNALS

ENGINE_VERSION = "1.2.3"
EVIDENCE_VERSION_PREFIX = "c1-"
ALPHA = 0.05
NODRIFT_Q = 0.95
MIN_ROWS = 100
FAMILIES = ["ks", "chi2", "psi"]
PRED = CONFIDENCE_SIGNALS + POS_PROB_SIGNALS
N1_FEATS = DETECTOR_SIGNALS + PRED + ["batch_size"]
# the six detectors and the signal that decides whether each one fired
FIRE_SIGNALS = {"ks": "ks_max", "chi2": "chi2_max", "psi": "psi_max",
                "mmd": "mmd", "lsdd": "lsdd", "classifier": "clf_auc"}
MIN_DETECTORS_FOR_DRIFT = 3
SEVERITY = [(1.0, "none"), (3.0, "low"), (7.0, "moderate"), (15.0, "high")]  # pp, else critical
TARGET_COL = {"accuracy_drop": "actual_accuracy_drop", "auc_drop": "actual_auc_drop"}
EXTENTS = ["none", "single", "multi"]
MIN_GROUP = 19  # ceil((n+1)(1-alpha)) <= n needs n >= 19 at alpha = 0.05


def _gbr(seed):
    return GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                                     subsample=0.8, min_samples_leaf=5, random_state=seed)


def _rf(seed):
    return RandomForestClassifier(n_estimators=300, class_weight="balanced", min_samples_leaf=2,
                                  n_jobs=-1, random_state=seed)


def _conformal_q(scores, alpha=ALPHA):
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))


def _severity(pp):
    for lim, name in SEVERITY:
        if pp < lim:
            return name
    return "critical"


class C1Engine:
    # ── TRAIN ────────────────────────────────────────────────────────────
    def train(self, bench_csv, reference_csv, test_csv, model_path, feature_types,
              target_col="target", seed=0, noise_sd=None):
        """noise_sd: {batch_size: clean-batch AUC sd in pp}; required when harm_metric is auc_drop."""
        t0 = time.time()
        df = pd.read_csv(bench_csv)
        assert len(df) == 1600 and (df["benchmark_version"].astype(str) == "1.2").all(), \
            "expected the 1,600-row v1.2 benchmark"
        df["group"] = np.where(df["drift_type"] == "none", df["condition_id"],
                               df["condition_id"].str.rsplit("_", n=1).str[0])
        ref = pd.read_csv(reference_csv)
        test = pd.read_csv(test_csv)
        self.base_model = joblib.load(model_path)
        self.target_col = target_col
        self.profile = ReferenceProfile(ref, feature_types, target_col=target_col)
        self.features = list(self.profile.features)

        # harm metric, fixed baseline and sigma(bs)
        self.harm_metric = (str(df["harm_metric"].iloc[0]) if "harm_metric" in df.columns
                            else "accuracy_drop")
        assert self.harm_metric in TARGET_COL, f"unknown harm metric {self.harm_metric}"
        self.sizes = sorted(int(b) for b in df["batch_size"].unique())
        yt, Xt = test[target_col].values, test[self.features]
        self.p = float(accuracy_score(yt, self.base_model.predict(Xt)))
        self.N = len(test)
        assert abs(round(self.p, 4) - df["fixed_baseline_accuracy"].iloc[0]) < 1.01e-4, \
            "fixed baseline accuracy differs from the benchmark (wrong model or test file?)"
        if self.harm_metric == "auc_drop":
            self.auc = float(roc_auc_score(yt, self.base_model.predict_proba(Xt)[:, 1]))
            assert abs(round(self.auc, 4) - df["fixed_baseline_auc"].iloc[0]) < 1.01e-4, \
                "fixed baseline AUC differs from the benchmark (wrong model or test file?)"
            assert noise_sd is not None, "noise_sd (clean-batch AUC sd per batch size) is required for auc_drop"
            self.noise_sd = {int(k): float(v) for k, v in noise_sd.items()}
            missing = [b for b in self.sizes if b not in self.noise_sd]
            assert not missing, f"noise_sd has no value for batch sizes {missing}"

        # per-feature columns in a fixed order
        self.fam_cols = {f: [c for c in df.columns if c.startswith(f + "__")] for f in FAMILIES}
        self.fam_feats = {f: [c.split("__", 1)[1] for c in cols] for f, cols in self.fam_cols.items()}
        self.fp_feats = sorted({x for v in self.fam_feats.values() for x in v})

        # no-drift thresholds per calibrated batch size (all no-drift rows)
        bs = df["batch_size"].values
        none = df["drift_type"].values == "none"
        self.thr = {f: {b: np.quantile(df.loc[none & (bs == b), cols].values.astype(float),
                                       NODRIFT_Q, axis=0)
                        for b in self.sizes} for f, cols in self.fam_cols.items()}
        self.det_thr = {name: {b: float(np.quantile(df.loc[none & (bs == b), sig].values, NODRIFT_Q))
                               for b in self.sizes} for name, sig in FIRE_SIGNALS.items()}

        # affected_features thresholds (strict, floored); N3 keeps using self.thr
        self.thr_max, self.aff_floor = {}, {}
        for f, cols in self.fam_cols.items():
            self.thr_max[f], self.aff_floor[f] = {}, {}
            for b in self.sizes:
                ref_sc = df.loc[none & (bs == b), cols].values.astype(float)
                t95 = self.thr[f][b]
                self.thr_max[f][b] = ref_sc.max(axis=0)
                self.aff_floor[f][b] = float(0.1 * np.median(t95[t95 > 0])) if (t95 > 0).any() else 1e-6
        self.n_fams = {}
        for f in FAMILIES:
            for feat in self.fam_feats[f]:
                self.n_fams[feat] = self.n_fams.get(feat, 0) + 1

        # N1 + N2: 80% groups train, 20% groups calibrate
        y = df[TARGET_COL[self.harm_metric]].values
        X1 = df[N1_FEATS].values.astype(float)
        tr, ca = next(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
                      .split(df, df["drift_type"], df["group"]))
        tr, ca = np.asarray(tr), np.asarray(ca)
        # StratifiedGroupKFold yields (train, test); the 20% test fold is our calibration set
        self.n1 = _gbr(seed).fit(X1[tr], y[tr])
        sig_ca = self._sigma(df["batch_size"].values[ca].astype(float))
        scores_ca = np.abs(y[ca] - self.n1.predict(X1[ca])) / sig_ca
        self.q = _conformal_q(scores_ca)  # global quantile (1.2.2 behaviour, also the fallback)
        self.n_train, self.n_cal = len(tr), len(ca)

        # N3 (Option B) on all 1,600 rows
        y5 = df["drift_type"].values
        y_ext = np.where(np.isin(y5, ["none", "interaction"]), "none",
                         np.where(y5 == "marginal_multi", "multi", "single"))
        y_flag = np.isin(y5, ["interaction", "combined"]).astype(int)
        Ms = {f: df[cols].values.astype(float) for f, cols in self.fam_cols.items()}
        X3 = np.hstack([df[DETECTOR_SIGNALS + PRED + ["batch_size"]].values.astype(float),
                        self._fingerprint(Ms, bs)])
        self.n3_extent = _rf(seed).fit(X3, y_ext)
        self.n3_flag = _rf(seed).fit(X3, y_flag)

        # N2 Mondrian (1.2.3): one quantile per N3-predicted extent, calibration rows grouped by
        # OUT-OF-FOLD N3 predictions (the in-sample ones above would be optimistic)
        oof_ext = np.empty(len(df), dtype=object)
        for f_tr, f_te in StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed).split(
                df, y5, df["group"]):
            oof_ext[f_te] = _rf(seed).fit(X3[f_tr], y_ext[f_tr]).predict(X3[f_te])
        self.n3_oof_extent_accuracy = float((oof_ext == y_ext).mean())
        g_ca = oof_ext[ca]
        self.q_ext, self.n_cal_ext = {}, {}
        for g in EXTENTS:
            m = g_ca == g
            self.n_cal_ext[g] = int(m.sum())
            self.q_ext[g] = _conformal_q(scores_ca[m]) if m.sum() >= MIN_GROUP else self.q

        self.trained_at = time.strftime("%Y-%m-%d %H:%M:%S")
        self.train_sec = round(time.time() - t0, 1)
        return self

    # ── helpers ──────────────────────────────────────────────────────────
    def _sigma(self, bs):
        """Sampling sd (pp) of the harm metric at batch size bs.
        accuracy_drop: binomial sd of batch accuracy, finite-population corrected.
        auc_drop: measured clean-batch AUC sd, log-log interpolated between calibrated sizes."""
        bs = np.asarray(bs, dtype=float)
        if getattr(self, "harm_metric", "accuracy_drop") == "auc_drop":
            ks = sorted(self.noise_sd)
            return np.exp(np.interp(np.log(bs), np.log(ks), np.log([self.noise_sd[k] for k in ks])))
        return 100 * np.sqrt(self.p * (1 - self.p) / bs * (self.N - bs) / (self.N - 1))

    def _nearest(self, b):
        return min(self.sizes, key=lambda s: (abs(s - b), s))  # ties go to the smaller size

    def _fingerprint(self, Ms, bs):
        """Feature-agnostic fingerprint (same as run_novelty3b_v12.fingerprint):
        per family top-3 sorted scores + count above threshold, then count of features
        above threshold in any family. bs = calibrated batch size per row."""
        n = len(bs)
        any_ex = np.zeros((n, len(self.fp_feats)), dtype=bool)
        blocks = []
        for f in FAMILIES:
            M = Ms[f]
            top = -np.sort(-M, axis=1)[:, :3]
            if top.shape[1] < 3:
                top = np.hstack([top, np.zeros((n, 3 - top.shape[1]))])
            thr = np.vstack([self.thr[f][int(b)] for b in bs])
            ex = M > thr
            for j, feat in enumerate(self.fam_feats[f]):
                any_ex[:, self.fp_feats.index(feat)] |= ex[:, j]
            blocks += [top, ex.sum(1, keepdims=True)]
        blocks.append(any_ex.sum(1, keepdims=True))
        return np.hstack(blocks).astype(float)

    # ── DETECT ───────────────────────────────────────────────────────────
    def detect(self, batch_df, model_id="adult-xgb-v1", reference_dataset_uri="",
               current_dataset_uri="", seed=0):
        t0 = time.time()
        if not isinstance(batch_df, pd.DataFrame):
            raise TypeError("batch_df must be a pandas DataFrame")
        missing = [f for f in self.features if f not in batch_df.columns]
        if missing:
            raise ValueError(f"batch is missing features: {missing}")
        X = batch_df[self.features]
        if X.isna().any().any():
            raise ValueError("batch contains NaN values: "
                             f"{X.columns[X.isna().any()].tolist()}")
        if len(X) < MIN_ROWS:
            raise ValueError(f"batch has {len(X)} rows; at least {MIN_ROWS} are needed")

        warnings = []
        bs_raw = len(X)
        bs = float(min(max(bs_raw, self.sizes[0]), self.sizes[-1]))
        if bs != bs_raw:
            warnings.append(f"batch size {bs_raw} is outside the calibrated range "
                            f"{self.sizes[0]}-{self.sizes[-1]}; treated as {int(bs)}")
        bc = self._nearest(bs)
        for f in self.profile.categorical:
            unseen = int((~X[f].isin(self.profile.categories[f])).sum())
            if unseen:
                warnings.append(f"{unseen} rows have a category of '{f}' never seen in the "
                                f"reference; harm interval may be unreliable")

        scores, per = run_all_detectors(self.profile, X, seed=seed)
        ps = prediction_stats(self.base_model, X)

        # N1 + N2
        x1 = np.array([[scores[k] for k in DETECTOR_SIGNALS] + [ps[k] for k in PRED] + [bs]])
        harm_pp = float(self.n1.predict(x1)[0])

        # N3
        Ms = {f: np.array([[per[f][feat] for feat in self.fam_feats[f]]]) for f in FAMILIES}
        x3 = np.hstack([x1, self._fingerprint(Ms, np.array([bc]))])
        extent = str(self.n3_extent.predict(x3)[0])
        p_int = float(self.n3_flag.predict_proba(x3)[0][list(self.n3_flag.classes_).index(1)])

        # N2 interval: Mondrian quantile for the predicted extent (engines before 1.2.3: global)
        q = getattr(self, "q_ext", {}).get(extent, self.q)
        half = float(q * self._sigma(bs))

        # detectors fired
        fired = [name for name, sig in FIRE_SIGNALS.items() if scores[sig] > self.det_thr[name][bc]]

        # affected features: only when N3 finds a marginal shift
        affected = []
        if extent != "none" and hasattr(self, "thr_max"):
            hits, ratio = {}, {}
            for f in FAMILIES:
                thr = np.maximum(self.thr_max[f][bc], self.aff_floor[f][bc])
                for j, feat in enumerate(self.fam_feats[f]):
                    if per[f][feat] > thr[j]:
                        hits[feat] = hits.get(feat, 0) + 1
                        ratio[feat] = max(ratio.get(feat, 0.0), per[f][feat] / thr[j])
            affected = [k for k, _ in sorted(ratio.items(), key=lambda kv: -kv[1])
                        if hits[k] >= self.n_fams[k]]
        elif extent != "none":  # engines saved before 1.2.2: original rule
            ratio = {}
            for f in FAMILIES:
                for j, feat in enumerate(self.fam_feats[f]):
                    r = per[f][feat] / max(float(self.thr[f][bc][j]), 1e-12)
                    ratio[feat] = max(ratio.get(feat, 0.0), r)
            affected = [k for k, v in sorted(ratio.items(), key=lambda kv: -kv[1]) if v > 1.0]

        drift_detected = bool(extent != "none" or len(fired) >= MIN_DETECTORS_FOR_DRIFT)

        return {
            "evidence_version": EVIDENCE_VERSION_PREFIX + uuid.uuid4().hex[:8],
            "engine_version": ENGINE_VERSION,
            "model_id": model_id,
            "reference_dataset_uri": reference_dataset_uri,
            "current_dataset_uri": current_dataset_uri,
            "batch_size": int(bs_raw),
            "drift_detected": drift_detected,
            "predicted_harm": round(harm_pp / 100, 4),
            "harm_interval": [round((harm_pp - half) / 100, 4), round((harm_pp + half) / 100, 4)],
            "harm_metric": getattr(self, "harm_metric", "accuracy_drop"),
            "harm_severity": _severity(harm_pp),
            "marginal_extent": extent,
            "interaction_suspected": round(p_int, 4),
            "affected_features": affected,
            "detector_scores": {k: round(float(scores[k]), 4) for k in DETECTOR_SIGNALS},
            "detectors_fired": len(fired),
            "detectors_fired_list": fired,
            "prediction_stats": {k: round(float(ps[k]), 4) for k in PRED},
            "warnings": warnings,
            "runtime_sec": round(time.time() - t0, 3),
        }

    # ── SAVE / LOAD / SUMMARY ────────────────────────────────────────────
    def save(self, path):
        joblib.dump(self, path, compress=3)
        return path

    @staticmethod
    def load(path):
        eng = joblib.load(path)
        assert isinstance(eng, C1Engine), "file is not a C1Engine"
        return eng

    def summary(self):
        return {"engine_version": ENGINE_VERSION, "trained_at": self.trained_at,
                "train_sec": self.train_sec,
                "n1_train_conditions": self.n_train, "n2_calibration_conditions": self.n_cal,
                "n2_quantile_normalised": round(self.q, 4),
                "n2_method": "mondrian_extent" if hasattr(self, "q_ext") else "global",
                "n2_quantile_by_extent": ({g: round(v, 4) for g, v in self.q_ext.items()}
                                          if hasattr(self, "q_ext") else None),
                "n2_calibration_by_extent": getattr(self, "n_cal_ext", None),
                "n3_oof_extent_accuracy": (round(self.n3_oof_extent_accuracy, 4)
                                           if hasattr(self, "n3_oof_extent_accuracy") else None),
                "calibrated_batch_sizes": self.sizes,
                "harm_metric": getattr(self, "harm_metric", "accuracy_drop"),
                "fixed_baseline_accuracy": round(self.p, 4),
                "fixed_baseline_auc": round(self.auc, 4) if hasattr(self, "auc") else None,
                "noise_sd_pp": getattr(self, "noise_sd", None),
                "n_features": len(self.features), "features": self.features}
