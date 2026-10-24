"""
Novelty 2 evaluation, label-free: does C2 find the responsible column without labels?
J26-DS-361 | C2 XAI Root Cause Diagnosis Engine | Tharmika N (IT22247940)

Production batches are passed WITHOUT the label column: the temporal SHAP
monitor and the label-free scorer never see labels. Labels are used only
afterwards, to grade.

Two sets of monitoring streams (5 cycles, drift onset at cycle 2, batch 1000
from unseen test.csv):
  suite   the scenario suite used for every C2 novelty: one drifted feature
          (high / low importance, 4 strengths), interaction, combined, clean.
          Truth: the injected feature(s).
  pairs   harmful + harmless (C1 label-free plan, 23 Oct 2026): every high-
          importance feature drifts together with every low-importance feature
          (strength 0.6). Both are "drifted", so "most drifted feature" is not
          enough; the question is which one is RESPONSIBLE for the damage.
          Truth (graded with labels, after the fact): restore each drifted
          feature in the final batch and measure how much accuracy comes back.
          A pair counts as harmful+harmless when one feature recovers >= 2 pp
          (team harm line) and the other <= 1 pp.

Seeds: 11 and 22 were used while designing the impact signal (development);
33, 44 and 55 were not looked at before the design was fixed (held out).

Methods
  B1_drift          drift only (-log10 p of KS / chi-square)
  B2_explanation    explanation only (Novelty 1 temporal SHAP z)
  S_decision        the decision-change signal on its own (share of decisions that
                    flip when the feature is restored, rank-preserving restore)
  S_predshift       the prediction-shift concentration signal on its own
  N2_free           Novelty 2 label-free (E, D, decision change, PS; 0.35/0.15/0.35/0.15)
  N2_free_DoC       the same, with the harm-drop impact signal and DoC as the harm
                    predictor: a stand-in for C1's predict_harm, to test the slot
  N2_labelled       the labelled Novelty 2 score: an upper bound only, not usable
                    in production (labels never exist there)

Metrics: top-1 / top-3; on pairs also the wrong-blame rate (the harmless drifted
feature ranked first); damage tracking (Spearman with the true accuracy drop);
drifted vs clean (AUC); weight ablation of the four label-free signals.

Run:   python3.11 -m c2.evaluation.evaluate_novelty2_free            (full, ~50 min)
       python3.11 -m c2.evaluation.evaluate_novelty2_free --quick    (~6 min)
Writes c2/evaluation/results/novelty2_free_runs.csv (git-ignored) and
       c2/evaluation/results/novelty2_free_summary.md (committed).
"""
import argparse
import itertools
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from c2.src.config import RESULTS_DIR, TARGET_COL
from c2.src.drift_impact import DriftImpactScorer
from c2.src.drift_impact_free import LabelFreeImpactScorer, doc_harm
from c2.src.restore import restore
from c2.src.scenarios import artifacts, make_temporal_stream, scenario_suite
from c2.src.temporal_shap import TemporalSHAPMonitor

warnings.filterwarnings("ignore", category=UserWarning)

N_CYCLES, ONSET = 5, 2
DEV_SEEDS, HELD_SEEDS = [11, 22], [33, 44, 55]
METHODS = ["B1_drift", "B2_explanation", "S_decision", "S_predshift",
           "N2_free", "N2_free_DoC", "N2_labelled"]
SIGNALS = ("E", "D", "DI", "PS")
GROUPS = ["high_importance", "low_importance", "interaction", "combined"]
HARM_PP, HARMLESS_PP = 2.0, 1.0


def _rank(scores):
    return sorted(scores, key=scores.get, reverse=True)


def true_recovery(model, batch, feats, ref_df, feature_types, n=3):
    """GRADING ONLY (uses labels): accuracy points recovered when each feature is restored."""
    X, y = batch.drop(columns=[TARGET_COL]), batch[TARGET_COL].values
    acc0 = np.mean(model.predict(X) == y)
    return {f: float(np.mean([100 * (np.mean(model.predict(restore(X, f, ref_df, feature_types,
                                                                   seed=900 + i)) == y) - acc0)
                              for i in range(n)])) for f in feats}


def run_stream(spec, seed, setname, a, mon, free, free_doc, labelled):
    dtype, feats, mag, pairs, group = spec
    stream = make_temporal_stream(dtype, feats, mag, n_cycles=N_CYCLES,
                                  onset=ONSET, seed=seed, interaction_pairs=pairs)
    gt = stream[-1]["ground_truth"]
    unlabelled = [s["drifted_df"].drop(columns=[TARGET_COL]) for s in stream]

    mon.reset()
    for b in unlabelled:                                   # no labels reach the monitor
        r = mon.update(b)
    free.calibrate(unlabelled[:ONSET])
    out = free.score(unlabelled[-1], r)                    # no labels reach the scorer
    free_doc.calibrate(unlabelled[:ONSET])
    out_doc = free_doc.score(unlabelled[-1], r)
    raw, norm = out["signals_raw"], out["signals_normalised"]
    labelled.calibrate_performance([s["drifted_df"] for s in stream[:ONSET]])
    lab = labelled.score(stream[-1]["drifted_df"], r)      # upper bound only

    # truth
    row = {"set": setname, "group": group, "drift_type": dtype, "magnitude": mag, "seed": seed,
           "split": "development" if seed in DEV_SEEDS else "held-out",
           "harm_pp": gt["actual_accuracy_drop"], "N1_alarm": r["alarm"],
           "root_flip_share": out["decision_flip_share"]}
    if setname == "pairs":
        rec = true_recovery(a["model"], stream[-1]["drifted_df"], feats, a["ref_df"], a["feature_types"])
        hi, lo = sorted(feats, key=rec.get, reverse=True)
        row.update({"recover_harmful": rec[hi], "recover_other": rec[lo],
                    "harmful_harmless": rec[hi] >= HARM_PP and rec[lo] <= HARMLESS_PP})
        causes, harmless = [hi], lo
    else:
        causes, harmless = stream[-1]["true_causes"], None
    row["features"] = "+".join(causes) or "-"
    row["harmless_feature"] = harmless or "-"

    scores = {
        "B1_drift": {f: v["D"] for f, v in raw.items()},
        "B2_explanation": {f: v["E"] for f, v in raw.items()},
        "S_decision": {f: v["DI"] for f, v in raw.items()},
        "S_predshift": {f: v["PS"] for f, v in raw.items()},
        "N2_free": out["drift_impact_scores"],
        "N2_free_DoC": out_doc["drift_impact_scores"],
        "N2_labelled": lab["drift_impact_scores"],
    }
    for m in METHODS:
        order = _rank(scores[m])
        row[f"{m}_pred"] = order[0]
        row[f"{m}_top_score"] = scores[m][order[0]]
        row[f"{m}_top1"] = (order[0] in causes) if causes else np.nan
        row[f"{m}_top3"] = any(f in causes for f in order[:3]) if causes else np.nan
        row[f"{m}_wrong_blame"] = (order[0] == harmless) if harmless else np.nan
    for f, v in norm.items():
        for k in SIGNALS:
            row[f"sig_{k}_{f}"] = v[k]
    row["_causes"] = causes
    return row


def weight_grid(step=0.1):
    vals = np.round(np.arange(0, 1 + 1e-9, step), 2)
    return [(e, d, p, abs(round(1 - e - d - p, 2)))
            for e, d, p in itertools.product(vals, vals, vals) if e + d + p <= 1 + 1e-9]


def ablation(frame, features):
    sig = {k: frame[[f"sig_{k}_{f}" for f in features]].values for k in SIGNALS}
    causes = list(frame["_causes"])
    out = []
    for w in weight_grid() + [(0.35, 0.15, 0.35, 0.15)]:
        dis = sum(wk * sig[k] for wk, k in zip(w, SIGNALS))
        top = [features[i] for i in dis.argmax(axis=1)]
        out.append(dict(zip(("w_E", "w_D", "w_DI", "w_PS"), w),
                        top1=float(np.mean([t in c for t, c in zip(top, causes)]))))
    return (pd.DataFrame(out).drop_duplicates(subset=["w_E", "w_D", "w_DI", "w_PS"])
            .sort_values("top1", ascending=False))


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return float("nan")
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def pct(v):
    return "n/a" if pd.isna(v) else f"{100 * v:.1f}%"


def summarise(df, features):
    suite = df[df["set"] == "suite"]
    drifted, clean = suite[suite["drift_type"] != "none"], suite[suite["drift_type"] == "none"]
    pairs_all = df[df["set"] == "pairs"]
    pairs = pairs_all[pairs_all["harmful_harmless"] == True]   # noqa: E712
    L = []

    def table(title, frame, blame=False):
        L.append(f"\n### {title}\n")
        L.append("| Method | top1 | top3 |" + (" blamed the harmless drifted feature |" if blame else ""))
        L.append("|---|---|---|" + ("---|" if blame else ""))
        for m in METHODS:
            tag = " (upper bound, needs labels)" if m == "N2_labelled" else (
                " (DoC stand-in for C1)" if m == "N2_free_DoC" else "")
            cells = [pct(frame[f"{m}_top1"].astype(float).mean()), pct(frame[f"{m}_top3"].astype(float).mean())]
            if blame:
                cells.append(pct(frame[f"{m}_wrong_blame"].astype(float).mean()))
            L.append(f"| {m}{tag} | " + " | ".join(cells) + " |")

    held = lambda f: f[f["split"] == "held-out"]   # noqa: E731
    table(f"A. Harmful + harmless pairs, held-out seeds (n={len(held(pairs))})", held(pairs), blame=True)
    table(f"A. Harmful + harmless pairs, all seeds (n={len(pairs)}; "
          f"{len(pairs_all)} pair streams, {len(pairs)} were harmful+harmless)", pairs, blame=True)
    table(f"B. Scenario suite, held-out seeds, drifted streams (n={len(held(drifted))})", held(drifted))
    table(f"B. Scenario suite, all seeds, drifted streams (n={len(drifted)})", drifted)
    for g in GROUPS:
        sub = drifted[drifted["group"] == g]
        if len(sub):
            table(f"B. {g}, all seeds (n={len(sub)})", sub)

    L.append("\n### Development vs held-out seeds (top-1)\n")
    L.append("The impact signal was designed on seeds 11 and 22; seeds 33, 44 and 55 were not looked "
             "at before the design was fixed.\n")
    L.append("| Method | suite, development | suite, held-out | pairs, development | pairs, held-out |")
    L.append("|---|---|---|---|---|")
    dev = lambda f: f[f["split"] == "development"]   # noqa: E731
    for m in METHODS:
        c = lambda f: pct(f[f"{m}_top1"].astype(float).mean())   # noqa: E731
        L.append(f"| {m} | {c(dev(drifted))} | {c(held(drifted))} | {c(dev(pairs))} | {c(held(pairs))} |")

    L.append(f"\n### Does it track real damage? (suite + pairs, n={len(df)})\n")
    L.append("| Score | Spearman r with true accuracy drop | p-value |")
    L.append("|---|---|---|")
    for name, col in [(f"{m} top score", f"{m}_top_score") for m in METHODS] + [
            ("root feature's share of flipped decisions (label-free)", "root_flip_share")]:
        r, p = spearmanr(df[col], df["harm_pp"])
        L.append(f"| {name} | {r:.3f} | {p:.2g} |")

    L.append(f"\n### Telling drifted from clean streams by the top score "
             f"({len(drifted)} drifted, {len(clean)} clean)\n")
    L.append("| Method | AUC |")
    L.append("|---|---|")
    for m in ("B1_drift", "B2_explanation", "N2_free", "N2_labelled"):
        L.append(f"| {m} | {auc(drifted[f'{m}_top_score'], clean[f'{m}_top_score']):.3f} |")
    L.append(f"| N1 alarm (persistence rule), false alarms on clean | "
             f"{pct(clean['N1_alarm'].astype(float).mean())} |")

    both = pd.concat([drifted, pairs])
    abl = ablation(both, features)
    cols = ["w_E", "w_D", "w_DI", "w_PS"]
    L.append(f"\n### Weight ablation, label-free signals (top-1, suite + harmful/harmless pairs, n={len(both)})\n")
    L.append("| w_E | w_D | w_DI | w_PS | top1 |")
    L.append("|---|---|---|---|---|")

    def is_w(r, w):
        return all(abs(r[c] - x) < 1e-9 for c, x in zip(cols, w))

    picks = [abl.head(5), abl[abl.apply(lambda r: is_w(r, (0.35, 0.15, 0.35, 0.15)), axis=1)],
             abl[abl.apply(lambda r: is_w(r, (0.7, 0.3, 0.0, 0.0)), axis=1)],
             abl[(abl[cols] == 1.0).any(axis=1)]]
    for _, r in pd.concat(picks).drop_duplicates().iterrows():
        tag = (" (default)" if is_w(r, (0.35, 0.15, 0.35, 0.15)) else
               " (explanation + drift only)" if is_w(r, (0.7, 0.3, 0.0, 0.0)) else
               " (single signal)" if max(r[c] for c in cols) == 1.0 else "")
        L.append(f"| {r.w_E:.2f} | {r.w_D:.2f} | {r.w_DI:.2f} | {r.w_PS:.2f} | {pct(r.top1)}{tag} |")
    n_best = int((abl["top1"] >= abl["top1"].max() - 1e-9).sum())
    L.append(f"\n{n_best} of {len(abl)} weight combinations reach the best top-1. The default weights "
             "were fixed before this run (the same as the labelled score); the ablation is a "
             "sensitivity analysis on the same streams.")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    suite_seeds = [11, 33] if args.quick else [11, 22, 33]
    pair_seeds = [44] if args.quick else [11, 33, 44, 55]
    n_clean = 5 if args.quick else 20

    a = artifacts()
    mon = TemporalSHAPMonitor(a["model"], a["ref_df"], a["feature_types"])
    free = LabelFreeImpactScorer(a["model"], a["ref_df"], a["feature_types"], mon.z_thresh)
    free_doc = LabelFreeImpactScorer(a["model"], a["ref_df"], a["feature_types"], mon.z_thresh,
                                     harm_fn=doc_harm(a["model"], a["ref_df"]))
    labelled = DriftImpactScorer(a["model"], a["ref_df"], a["feature_types"], mon.z_thresh)

    jobs = [(s, seed, "suite") for s in scenario_suite() if s[0] != "none" for seed in suite_seeds]
    jobs += [(("none", [], 0.0, None, "none"), 1000 + i, "suite") for i in range(n_clean)]
    jobs += [(("marginal_multi", [h, l], 0.6, None, "pairs"), seed, "pairs")
             for h in a["high_importance"] for l in a["low_importance"] for seed in pair_seeds]

    rows, t0 = [], time.time()
    for i, (spec, seed, setname) in enumerate(jobs, 1):
        rows.append(run_stream(spec, seed, setname, a, mon, free, free_doc, labelled))
        if i % 10 == 0 or i == len(jobs):
            print(f"  {i}/{len(jobs)} streams  ({time.time() - t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    text = ("# Novelty 2 without labels: finding the responsible column\n\n"
            f"{len(df)} monitoring streams ({N_CYCLES} cycles, onset at cycle {ONSET}, batch 1000 from "
            f"test.csv). Scenario suite seeds {suite_seeds}, harmful+harmless pair seeds {pair_seeds}. "
            "Production batches carry no labels; labels are used only to grade. Label-free weights "
            f"(E, D, DI, PS) = {free.weights}. Impact signal: decision change (share of decisions "
            "that flip when the feature is restored); N2_free_DoC tests the harm-drop slot with DoC "
            "standing in for C1's predict_harm.\n"
            + summarise(df, a["feature_names"]) + "\n")
    df.drop(columns=["_causes"]).to_csv(RESULTS_DIR / "novelty2_free_runs.csv", index=False)
    (RESULTS_DIR / "novelty2_free_summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
