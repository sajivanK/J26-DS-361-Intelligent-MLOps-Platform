"""
Novelty 2 evaluation: Drift Impact Score vs single-evidence baselines.
J26-DS-361 | C2 XAI Root Cause Diagnosis Engine | Tharmika N (IT22247940)

Same monitoring streams as Novelty 1 (5 cycles, drift onset at cycle 2, batches
from unseen test.csv). The two clean warm-up cycles calibrate the performance
noise band; every method is judged on the last cycle.

Methods (proposal Section 3.3)
  B1_drift        Baseline 1, drift-only           (-log10 p, KS / chi-square)
  B2_explanation  Baseline 2, explanation-only     (Novelty 1 temporal SHAP z)
  B3_performance  Baseline 3, performance-only     (restore-test z)
  B4_subgroup     Baseline 4, subgroup-only        (SliceLine slice score)
  G_subgroup_chg  the subgroup signal used INSIDE the DIS on its own (largest
                  slice error-rate rise vs the clean warm-up cycles)
  N2_DIS          Novelty 2, fused Drift Impact Score, four signals
                  (default weights E/D/P/G = 0.35/0.15/0.35/0.15)

Metrics
  top1 / top3          correct root cause ranked 1st / in top 3
  harm correlation     Spearman between each method's top score and the true
                       accuracy drop, over all streams (does the score track damage?)
  harm estimate        N2 estimated harm vs true harm (Spearman, mean abs error)
  malignant detection  team definition: true drop > 2pp
  weight ablation      top-1 for every 4-signal weight combination on a 0.1 grid,
                       including the earlier 3-signal setting (0.4/0.2/0.4/0)

Run:   python3.11 -m c2.evaluation.evaluate_novelty2            (full, ~8 min)
       python3.11 -m c2.evaluation.evaluate_novelty2 --quick    (~2 min)
Writes c2/evaluation/results/novelty2_runs.csv (git-ignored) and
       c2/evaluation/results/novelty2_summary.md (committed).
"""
import argparse
import itertools
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from c2.evaluation.baselines import subgroup_only_ranking
from c2.src.config import RESULTS_DIR
from c2.src.drift_impact import MALIGNANT_PP, DriftImpactScorer
from c2.src.scenarios import artifacts, make_temporal_stream, scenario_suite
from c2.src.temporal_shap import TemporalSHAPMonitor

warnings.filterwarnings("ignore", category=UserWarning)

N_CYCLES, ONSET = 5, 2
METHODS = ["B1_drift", "B2_explanation", "B3_performance", "B4_subgroup",
           "G_subgroup_chg", "N2_DIS"]
SIGNALS = ("E", "D", "P", "G")
GROUPS = ["high_importance", "low_importance", "interaction", "combined"]


def _rank(scores):
    return sorted(scores, key=scores.get, reverse=True)


def run_stream(spec, seed, a, mon, scorer):
    dtype, feats, mag, pairs, group = spec
    stream = make_temporal_stream(dtype, feats, mag, n_cycles=N_CYCLES,
                                  onset=ONSET, seed=seed, interaction_pairs=pairs)
    causes = stream[-1]["true_causes"]
    last = stream[-1]["drifted_df"]
    gt = stream[-1]["ground_truth"]

    mon.reset()
    for s in stream:
        r = mon.update(s["drifted_df"])
    scorer.calibrate_performance([s["drifted_df"] for s in stream[:ONSET]])
    out = scorer.score(last, r)
    raw, norm = out["signals_raw"], out["signals_normalised"]
    sub = subgroup_only_ranking(last, a["model"], a["feature_types"], ref_df=a["ref_df"])

    scores = {
        "B1_drift": {f: v["D"] for f, v in raw.items()},
        "B2_explanation": {f: v["E"] for f, v in raw.items()},
        "B3_performance": {f: v["P"] for f, v in raw.items()},
        "B4_subgroup": sub["ranking"],
        "G_subgroup_chg": {f: v["G"] for f, v in raw.items()},
        "N2_DIS": out["drift_impact_scores"],
    }
    row = {"group": group, "drift_type": dtype, "features": "+".join(causes) or "-",
           "magnitude": mag, "seed": seed,
           "harm_pp": gt["actual_accuracy_drop"],
           "true_malignant": gt["actual_accuracy_drop"] > MALIGNANT_PP,
           "N2_est_harm_pp": out["estimated_harm_pp"],
           "N2_malignant": out["is_malignant"],
           "N1_alarm": r["alarm"]}
    for m in METHODS:
        order = _rank(scores[m])
        row[f"{m}_pred"] = order[0]
        row[f"{m}_top_score"] = scores[m][order[0]]
        row[f"{m}_top1"] = (order[0] in causes) if causes else np.nan
        row[f"{m}_top3"] = any(f in causes for f in order[:3]) if causes else np.nan
    # keep normalised signals for the offline weight ablation
    for f, v in norm.items():
        for k in SIGNALS:
            row[f"sig_{k}_{f}"] = v[k]
    row["_causes"] = causes
    return row


def weight_grid(step=0.1):
    vals = np.round(np.arange(0, 1 + 1e-9, step), 2)
    grid = [(e, d, p, abs(round(1 - e - d - p, 2)))
            for e, d, p in itertools.product(vals, vals, vals) if e + d + p <= 1 + 1e-9]
    return grid + [(0.35, 0.15, 0.35, 0.15)]


def ablation(df, features):
    drifted = df[df["drift_type"] != "none"]
    sig = {k: drifted[[f"sig_{k}_{f}" for f in features]].values for k in SIGNALS}
    causes = list(drifted["_causes"])
    out = []
    for w in weight_grid():
        dis = sum(wk * sig[k] for wk, k in zip(w, SIGNALS))
        top = [features[i] for i in dis.argmax(axis=1)]
        hits = [t in c for t, c in zip(top, causes)]
        out.append({"w_explanation": w[0], "w_drift": w[1], "w_performance": w[2],
                    "w_subgroup": w[3], "top1": float(np.mean(hits))})
    return (pd.DataFrame(out).drop_duplicates(
        subset=["w_explanation", "w_drift", "w_performance", "w_subgroup"])
        .sort_values("top1", ascending=False))


def pct(v):
    return "n/a" if pd.isna(v) else f"{100 * v:.1f}%"


def summarise(df, abl):
    drifted = df[df["drift_type"] != "none"]
    L = []

    def table(title, frame):
        L.append(f"\n### {title}\n")
        L.append("| Method | top1 | top3 |")
        L.append("|---|---|---|")
        for m in METHODS:
            L.append(f"| {m} | {pct(frame[f'{m}_top1'].astype(float).mean())} | "
                     f"{pct(frame[f'{m}_top3'].astype(float).mean())} |")

    table(f"Root cause ranking, all drifted streams (n={len(drifted)})", drifted)
    for g in GROUPS:
        sub = drifted[drifted["group"] == g]
        if len(sub):
            table(f"{g} (n={len(sub)})", sub)

    L.append(f"\n### Does the top score track real damage? (all streams, n={len(df)})\n")
    L.append("| Method | Spearman r with true accuracy drop | p-value |")
    L.append("|---|---|---|")
    for m in METHODS:
        r, p = spearmanr(df[f"{m}_top_score"], df["harm_pp"])
        L.append(f"| {m} | {r:.3f} | {p:.2g} |")

    r, p = spearmanr(df["N2_est_harm_pp"], df["harm_pp"])
    mae = (df["N2_est_harm_pp"] - df["harm_pp"]).abs().mean()
    L.append(f"\n### Novelty 2 harm estimate\n")
    L.append(f"- Spearman r (estimated vs true drop): {r:.3f} (p={p:.2g})")
    L.append(f"- Mean absolute error: {mae:.2f} pp")

    y = df["true_malignant"].astype(bool)
    L.append(f"\n### Malignant detection (true drop > {MALIGNANT_PP}pp; "
             f"{int(y.sum())} malignant of {len(df)} streams)\n")
    L.append("| Rule | Precision | Recall | F1 | Accuracy |")
    L.append("|---|---|---|---|---|")
    for name, pred in [("N1 alarm (any explanation change = malignant)", df["N1_alarm"]),
                       ("N2 is_malignant", df["N2_malignant"])]:
        pred = pred.astype(bool)
        tp = int((pred & y).sum())
        prec = tp / max(pred.sum(), 1)
        rec = tp / max(y.sum(), 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        acc = (pred == y).mean()
        L.append(f"| {name} | {pct(prec)} | {pct(rec)} | {pct(f1)} | {pct(acc)} |")

    cols = ["w_explanation", "w_drift", "w_performance", "w_subgroup"]
    L.append("\n### Weight ablation (top-1 on drifted streams)\n")
    L.append("| w_explanation | w_drift | w_performance | w_subgroup | top1 |")
    L.append("|---|---|---|---|---|")

    def is_w(r, w):
        return all(abs(r[c] - x) < 1e-9 for c, x in zip(cols, w))

    named = {(0.35, 0.15, 0.35, 0.15): " (default, 4 signals)",
             (0.4, 0.2, 0.4, 0.0): " (earlier 3-signal setting)"}
    picks = [abl.head(5)]
    for w in named:
        picks.append(abl[abl.apply(lambda r: is_w(r, w), axis=1)])
    picks.append(abl[(abl[cols] == 1.0).any(axis=1)])
    for _, r in pd.concat(picks).drop_duplicates().iterrows():
        tag = next((t for w, t in named.items() if is_w(r, w)), "")
        tag = tag or (" (single signal)" if max(r[c] for c in cols) == 1.0 else "")
        L.append(f"| {r.w_explanation:.2f} | {r.w_drift:.2f} | {r.w_performance:.2f} | "
                 f"{r.w_subgroup:.2f} | {pct(r.top1)}{tag} |")
    n_ge = int((abl["top1"] >= abl["top1"].max() - 1e-9).sum())
    with_g = abl[abl.w_subgroup > 0]["top1"].max()
    without_g = abl[abl.w_subgroup == 0]["top1"].max()
    L.append(f"\n{n_ge} of {len(abl)} weight combinations reach the best top-1. "
             f"Best with the subgroup signal: {pct(with_g)}; best without it: "
             f"{pct(without_g)}.")
    L.append("Weights are reported as a sensitivity analysis on the same streams; "
             "the default weights were fixed before this run.")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    seeds = [11] if args.quick else [11, 22, 33]
    n_clean = 5 if args.quick else 20

    a = artifacts()
    mon = TemporalSHAPMonitor(a["model"], a["ref_df"], a["feature_types"])
    scorer = DriftImpactScorer(a["model"], a["ref_df"], a["feature_types"], mon.z_thresh)
    specs = [s for s in scenario_suite() if s[0] != "none"]
    jobs = [(s, seed) for s in specs for seed in seeds]
    jobs += [(("none", [], 0.0, None, "none"), 1000 + i) for i in range(n_clean)]

    rows, t0 = [], time.time()
    for i, (spec, seed) in enumerate(jobs, 1):
        rows.append(run_stream(spec, seed, a, mon, scorer))
        if i % 10 == 0 or i == len(jobs):
            print(f"  {i}/{len(jobs)} streams  ({time.time() - t0:.0f}s)")

    df = pd.DataFrame(rows)
    abl = ablation(df, a["feature_names"])
    df.drop(columns=["_causes"]).to_csv(RESULTS_DIR / "novelty2_runs.csv", index=False)
    header = (f"# Novelty 2 results: Drift Impact Score vs single-evidence baselines\n\n"
              f"Streams: {len(df)} ({N_CYCLES} cycles, onset at cycle {ONSET}, batch 1000 "
              f"from test.csv, seeds {seeds}). Default weights (E, D, P, G) = "
              f"{scorer.weights}.\n")
    text = header + summarise(df, abl) + "\n"
    (RESULTS_DIR / "novelty2_summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()