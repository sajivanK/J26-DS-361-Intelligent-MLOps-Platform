"""
Scenario generator: controlled drift scenarios with known ground truth.
J26-DS-361 | C2 XAI Root Cause Diagnosis Engine | Tharmika N (IT22247940)

Wraps the shared inject_drift() (v1.1) so every C2 experiment uses the same
settings as the team benchmark:
  - production batches are drawn from UNSEEN data (test.csv) via source_df
  - harm is measured paired (same rows before vs after injection)
  - feature types always come from feature_importance_adult.json

Two kinds of scenario:
  make_scenario()       one production batch (for snapshot methods)
  make_temporal_stream() several consecutive monitoring cycles, with drift
                         switching on at a chosen cycle (for Novelty 1)
"""
import numpy as np

from c2.src.config import TARGET_COL, load_artifacts
from injection.inject_drift import inject_drift

_ART = None


def artifacts():
    """Load the shared artifacts once and reuse them."""
    global _ART
    if _ART is None:
        _ART = load_artifacts()
    return _ART


def true_root_causes(drift_type, features, interaction_pairs=None):
    """Features that count as a correct diagnosis for a scenario."""
    if drift_type == "none":
        return []
    causes = list(features)
    for a, b in (interaction_pairs or []):
        causes += [a, b]
    return list(dict.fromkeys(causes))   # unique, order kept


def make_scenario(drift_type, features, magnitude, seed=42, batch_size=1000,
                  interaction_pairs=None, paired=True):
    """
    One drifted production batch plus its ground truth.

    Example:
        s = make_scenario("marginal_single", ["education_num"], 0.4)
        s["drifted_df"], s["ground_truth"], s["true_causes"]
    """
    a = artifacts()
    involved = true_root_causes(drift_type, features, interaction_pairs)
    ftypes = {f: a["feature_types"][f] for f in involved}

    result = inject_drift(
        reference_df=a["ref_df"],
        model=a["model"],
        drift_type=drift_type,
        features=list(features),
        magnitude=float(magnitude),
        feature_types=ftypes,
        interaction_pairs=interaction_pairs,
        batch_size=batch_size,
        target_col=TARGET_COL,
        seed=int(seed),
        source_df=a["test_df"],
        paired=paired,
    )
    return {
        "drifted_df": result["drifted_df"],
        "ground_truth": result["ground_truth"],
        "true_causes": involved,
        "spec": {
            "drift_type": drift_type, "features": list(features),
            "magnitude": magnitude, "seed": seed, "batch_size": batch_size,
            "interaction_pairs": interaction_pairs,
        },
    }


def make_temporal_stream(drift_type, features, magnitude, n_cycles=5, onset=2,
                         seed=42, batch_size=1000, interaction_pairs=None):
    """
    A sequence of monitoring cycles (e.g. one every 6 hours).

    Cycles before `onset` are clean (drift_type 'none'); from `onset` onwards
    the given drift is injected. Each cycle draws its own batch from test.csv
    with its own seed, so cycle-to-cycle sampling noise is realistic.

    Returns a list of scenario dicts (same format as make_scenario) with an
    extra key 'cycle'.
    """
    rng = np.random.default_rng(seed)
    cycle_seeds = rng.integers(0, 1_000_000, size=n_cycles)
    stream = []
    for c in range(n_cycles):
        drifted = drift_type != "none" and c >= onset
        s = make_scenario(
            drift_type if drifted else "none",
            features if drifted else [],
            magnitude if drifted else 0.0,
            seed=int(cycle_seeds[c]),
            batch_size=batch_size,
            interaction_pairs=interaction_pairs if drifted else None,
        )
        s["cycle"] = c
        stream.append(s)
    return stream


# ── STANDARD SCENARIO SUITE ────────────────────────────────────────────────
# (drift_type, features, magnitude, interaction_pairs, group)
# Built from the JSON importance groups so nothing is hardcoded twice.
def scenario_suite(magnitudes=(0.2, 0.4, 0.6, 0.8)):
    a = artifacts()
    suite = []
    for f in a["high_importance"]:
        for m in magnitudes:
            suite.append(("marginal_single", [f], m, None, "high_importance"))
    for f in a["low_importance"]:
        for m in magnitudes:
            suite.append(("marginal_single", [f], m, None, "low_importance"))
    for pair in [("education_num", "occupation"), ("age", "marital_status")]:
        for m in (0.4, 0.8):
            suite.append(("interaction", list(pair), m, None, "interaction"))
    for m in (0.4, 0.8):
        suite.append(("combined", ["education_num"], m,
                      [("age", "occupation")], "combined"))
    suite.append(("none", [], 0.0, None, "none"))
    return suite


if __name__ == "__main__":
    # Quick self-check: python3.11 -m c2.src.scenarios
    s = make_scenario("marginal_single", ["education_num"], 0.4)
    gt = s["ground_truth"]
    print(f"marginal  education_num m=0.4: batch={gt['batch_size']}  "
          f"drop={gt['actual_accuracy_drop']:.2f}pp  severity={gt['severity']}")

    s = make_scenario("marginal_single", ["native_country"], 0.4)
    print(f"benign    native_country m=0.4: drop="
          f"{s['ground_truth']['actual_accuracy_drop']:.2f}pp")

    s = make_scenario("interaction", ["education_num", "occupation"], 0.8)
    print(f"interaction edu x occ m=0.8: drop="
          f"{s['ground_truth']['actual_accuracy_drop']:.2f}pp  "
          f"KS={s['ground_truth']['per_feature_ks']}")

    s = make_scenario("none", [], 0.0)
    print(f"no drift: drop={s['ground_truth']['actual_accuracy_drop']:.2f}pp")

    stream = make_temporal_stream("marginal_single", ["age"], 0.6)
    print("temporal stream (age, onset at cycle 2):")
    for c in stream:
        print(f"  cycle {c['cycle']}: {c['ground_truth']['drift_type']:<16} "
              f"drop={c['ground_truth']['actual_accuracy_drop']:.2f}pp")
    print(f"suite size: {len(scenario_suite())} scenarios")
    print("SCENARIOS OK")