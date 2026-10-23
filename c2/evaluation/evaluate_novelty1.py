"""
Novelty 1 evaluation: Temporal SHAP Monitoring vs existing approaches.
J26-DS-361 | C2 XAI Root Cause Diagnosis Engine | Tharmika N (IT22247940)

Every method sees the same monitoring streams (5 cycles, drift switches on at
cycle 2, batches from unseen test.csv) and is judged at the last cycle.

Methods
  A          Baseline A, drift-only (KS / chi-square) on the latest batch
  B          Baseline B, snapshot SHAP (mean |SHAP| change) on the latest batch
  B_pooled   ablation: snapshot SHAP on ALL drifted batches pooled
             (same data as N1 has seen, so any N1 gain is not just "more data")
  N1_single  ablation: Novelty 1 noise band + two channels, latest batch only
  N1         Novelty 1, full temporal monitor (noise band + persistence)

Metrics
  top1 / top3     correct root cause ranked 1st / in top 3 (drifted streams)
  detection       alarm raised on drifted streams
  false_alarm     alarm raised on clean streams (A: Bonferroni p < 0.05/12;
                  B has no decision rule, it always names a root cause)
  stability       N1 diagnosis_stability at the last cycle
  delay           cycles from drift onset to N1's first alarm

Run:   python3.11 -m c2.evaluation.evaluate_novelty1            (full, ~10 min)
       python3.11 -m c2.evaluation.evaluate_novelty1 --quick    (~2 min)
Writes c2/evaluation/results/novelty1_runs.csv (per stream, git-ignored) and
       c2/evaluation/results/novelty1_summary.md (tables, committed).
"""
import argparse
import time
import warnings

import numpy as np
import pandas as pd

from c2.evaluation.baselines import SnapshotSHAP, drift_only_ranking
from c2.src.config import RESULTS_DIR
from c2.src.scenarios import artifacts, make_temporal_stream, scenario_suite
from c2.src.temporal_shap import TemporalSHAPMonitor

warnings.filterwarnings("ignore", category=UserWarning)

N_CYCLES, ONSET = 5, 2
METHODS = ["A", "B", "B_pooled", "N1_single", "N1"]


def hit(ranking, causes, k):
    return bool(causes) and any(f in causes for f in list(ranking)[:k])


def run_stream(spec, seed, a, snap, mon):
    dtype, feats, mag, pairs, group = spec
    stream = make_temporal_stream(dtype, feats, mag, n_cycles=N_CYCLES,
                                  onset=ONSET, seed=seed, interaction_pairs=pairs)
    causes = stream[-1]["true_causes"]
    last = stream[-1]["drifted_df"]

    mon.reset()
    first_alarm = None
    for s in stream:
        r = mon.update(s["drifted_df"])
        if r["alarm"] and first_alarm is None and r["cycle"] >= ONSET:
            first_alarm = r["cycle"]
    single = mon.history[-1]

    drifted = [s["drifted_df"] for s in stream[ONSET:]]
    A = drift_only_ranking(a["ref_df"], last, a["feature_types"])
    n_feat = len(a["feature_names"])
    rankings = {
        "A": A["ranking"],
        "B": snap.rank(last)["ranking"],
        "B_pooled": snap.rank(pd.concat(drifted))["ranking"],
        "N1_single": single["ranking"],
        "N1": r["ranking"],
    }
    alarms = {
        "A": min(A["pvalues"].values()) < 0.05 / n_feat,
        "B": np.nan,
        "B_pooled": np.nan,
        "N1_single": len(single["significant"]) > 0,
        "N1": r["alarm"],
    }
    row = {
        "group": group, "drift_type": dtype, "features": "+".join(causes) or "-",
        "magnitude": mag, "seed": seed,
        "harm_pp": stream[-1]["ground_truth"]["actual_accuracy_drop"],
        "N1_stability": r["diagnosis_stability"],
        "N1_pattern": r["shift_pattern"],
        "N1_delay": (first_alarm - ONSET) if first_alarm is not None else np.nan,
    }
    for m in METHODS:
        row[f"{m}_top1"] = hit(rankings[m], causes, 1) if causes else np.nan
        row[f"{m}_top3"] = hit(rankings[m], causes, 3) if causes else np.nan
        row[f"{m}_alarm"] = alarms[m]
        row[f"{m}_pred"] = next(iter(rankings[m]))
    return row


def summarise(df):
    drifted = df[df["drift_type"] != "none"]
    clean = df[df["drift_type"] == "none"]
    lines = []

    def table(title, frame, cols):
        lines.append(f"\n### {title}\n")
        lines.append("| Method | " + " | ".join(cols) + " |")
        lines.append("|---|" + "---|" * len(cols))
        for m in METHODS:
            vals = []
            for c in cols:
                key = f"{m}_{c}"
                v = frame[key].astype(float).mean() if key in frame else np.nan
                vals.append("n/a" if np.isnan(v) else f"{100 * v:.1f}%")
            lines.append(f"| {m} | " + " | ".join(vals) + " |")

    table(f"All drifted streams (n={len(drifted)})", drifted, ["top1", "top3", "alarm"])
    for g in ["high_importance", "low_importance", "interaction", "combined"]:
        sub = drifted[drifted["group"] == g]
        if len(sub):
            table(f"{g} (n={len(sub)})", sub, ["top1", "top3", "alarm"])
    lines.append(f"\n### False alarms on clean streams (n={len(clean)})\n")
    lines.append("| Method | False alarm rate |")
    lines.append("|---|---|")
    for m in METHODS:
        v = clean[f"{m}_alarm"].astype(float).mean()
        lines.append(f"| {m} | {'n/a (always names a cause)' if np.isnan(v) else f'{100 * v:.1f}%'} |")

    det = drifted["N1_delay"].dropna()
    lines.append("\n### Novelty 1 temporal behaviour\n")
    lines.append(f"- Mean detection delay after onset: {det.mean():.2f} cycles "
                 f"(detected {len(det)}/{len(drifted)} drifted streams)")
    lines.append(f"- Mean diagnosis stability at last cycle (drifted): "
                 f"{drifted['N1_stability'].mean():.3f}")
    inter = drifted[drifted["group"] == "interaction"]
    if len(inter):
        share = (inter["N1_pattern"] == "behaviour_shift").mean()
        lines.append(f"- Interaction streams labelled behaviour_shift: {100 * share:.1f}%")
    marg = drifted[drifted["drift_type"] == "marginal_single"]
    if len(marg):
        share = (marg["N1_pattern"] == "distribution_shift").mean()
        lines.append(f"- Marginal streams labelled distribution_shift: {100 * share:.1f}%")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="1 seed, fewer clean streams")
    args = ap.parse_args()
    seeds = [11] if args.quick else [11, 22, 33]
    n_clean = 5 if args.quick else 20

    a = artifacts()
    snap = SnapshotSHAP(a["model"], a["ref_df"])
    mon = TemporalSHAPMonitor(a["model"], a["ref_df"], a["feature_types"])
    specs = [s for s in scenario_suite() if s[0] != "none"]
    jobs = [(s, seed) for s in specs for seed in seeds]
    jobs += [(("none", [], 0.0, None, "none"), 1000 + i) for i in range(n_clean)]

    rows, t0 = [], time.time()
    for i, (spec, seed) in enumerate(jobs, 1):
        rows.append(run_stream(spec, seed, a, snap, mon))
        if i % 10 == 0 or i == len(jobs):
            print(f"  {i}/{len(jobs)} streams  ({time.time() - t0:.0f}s)")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "novelty1_runs.csv", index=False)
    summary = summarise(df)
    header = (f"# Novelty 1 results: Temporal SHAP vs baselines\n\n"
              f"Streams: {len(df)} ({N_CYCLES} cycles, drift onset at cycle {ONSET}, "
              f"batch 1000 from test.csv, seeds {seeds}). "
              f"N1 alarm threshold z > {mon.z_thresh:.2f}.\n")
    (RESULTS_DIR / "novelty1_summary.md").write_text(header + summary + "\n")
    print(header + summary)


if __name__ == "__main__":
    main()