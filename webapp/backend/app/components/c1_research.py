"""C1 research evidence for the C1 page "Research evidence" tab (owner: Sajivan K).

Reads the result JSONs committed in c1/results/ (written by the C1 experiment scripts) and returns
a compact summary per dataset. Nothing here is typed in by hand: every number comes from those files.
The optional meta-learner comparison is shown only when its JSON has been committed too.
"""
import json
import math
from pathlib import Path

from ..settings import REPO_ROOT

RESULTS_DIR = REPO_ROOT / "c1" / "results"

FILES = {
    "uci": {"suffix": "adult_v12", "label": "UCI Adult", "metric": "accuracy"},
    "ieee": {"suffix": "ieeecis_v12", "label": "IEEE-CIS Fraud", "metric": "AUC"},
}
# baseline keys in the baselines JSON -> display names (UCI and IEEE use different variants)
BASELINES = [("ATC_MC", "ATC"), ("ATC_Reg", "ATC (regression)"), ("DoC_Feat", "DoC (feature)"),
             ("DoC_Reg", "DoC (regression)"), ("CBPE", "CBPE"), ("CBPE_rel", "CBPE (relative)")]
ABLATIONS = [("ks_only_v11", "KS only"), ("ks_only", "KS only"), ("detectors_v12", "6 detectors only"),
             ("conf_plus_posprob", "Prediction stats only")]
TYPES = [("none", "None"), ("marginal_single", "Single"), ("marginal_multi", "Multi"),
         ("interaction", "Interaction"), ("combined", "Combined")]
FRESH_TYPES = [("none", "No drift"), ("marginal_single", "Single feature"), ("marginal_multi", "Multi feature"),
               ("interaction", "Interaction"), ("combined", "Combined")]


def _clean(x):
    """NaN / inf are not valid JSON: turn them into None."""
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_clean(v) for v in x]
    return x


def _load(name, required=True):
    path = RESULTS_DIR / f"{name}.json"
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"c1/results/{name}.json is missing")
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _mean(v):
    return v[0] if isinstance(v, list) else v


def _n1(n1, bl, target, comparison):
    summ = n1["summary"][target]
    tgt = bl["targets"][target]
    group = tgt.get("ALL conditions") or next(iter(tgt.values()))
    bars = [{"name": label, "mae": _mean(group[k]["mae"]), "kind": "baseline", "key": k}
            for k, label in BASELINES if k in group]
    bars.append({"name": "Constant guess", "mae": _mean(summ["constant_mean"]["mae"]), "kind": "reference", "key": "constant_mean"})
    seen = set()
    for k, label in ABLATIONS:
        if k in summ and label not in seen:
            seen.add(label)
            bars.append({"name": label, "mae": _mean(summ[k]["mae"]), "kind": "ablation", "key": k})
    n1_mae = _mean(summ["everything"]["mae"])
    bars.append({"name": "N1 (detectors + stats)", "mae": n1_mae, "kind": "n1", "key": "everything"})
    bars.sort(key=lambda b: -b["mae"])

    atc = next((b for b in bars if b["key"] in ("ATC_MC", "ATC_Reg")), None)
    wil = group.get("wilcoxon_vs_N1_full", {})
    tests = []
    for k, label in BASELINES:
        if k in wil:
            w = wil[k]
            tests.append({"name": label, "p": w["p_value"], "share_n1_better": w["share_groups_N1_better"],
                          "n1_better": w["median_diff"] < 0 and w["p_value"] < 0.05})
    pair = n1["pairs"][target].get("everything__vs__conf_plus_posprob", {})
    out = {"bars": bars, "n1_mae": n1_mae, "constant_mae": _mean(summ["constant_mean"]["mae"]),
           "atc_name": atc["name"] if atc else None, "atc_mae": atc["mae"] if atc else None,
           "atc_improvement_pct": (1 - n1_mae / atc["mae"]) * 100 if atc else None,
           "tests": tests,
           "detector_gain": {"mae_diff": pair.get("mae_diff_mean"), "repeats": pair.get("repeats_A_better")},
           "model_comparison": None}
    if comparison:
        rows = comparison.get(target)
        if rows:
            out["model_comparison"] = [{"model": r["model"], "mae": r["MAE"], "gbr_better_of_10": r.get("GBR better (of 10)")}
                                       for r in rows if r["model"] != "constant"]
    return out


def _n2(n2, mo, target):
    plain, norm = n2["targets"][target]["plain"], n2["targets"][target]["bs_normalised"]
    sizes = sorted(int(k[6:]) for k in norm if k.startswith("cov_bs"))
    by_batch = [{"batch_size": s, "plain": _mean(plain[f"cov_bs{s}"]), "normalised": _mean(norm[f"cov_bs{s}"]),
                 "width_normalised": _mean(norm.get(f"width_bs{s}"))} for s in sizes]
    m = mo["targets"][target]["summary"]
    one, mond = m["bs_normalised"], m["mondrian_extent"]
    by_type = [{"type": label, "normalised": _mean(one[f"cov_{k}"]), "mondrian": _mean(mond[f"cov_{k}"])} for k, label in TYPES]
    return {"by_batch": by_batch, "by_type": by_type, "splits": n2.get("splits"),
            "coverage_one_q": _mean(one["coverage"]), "coverage_mondrian": _mean(mond["coverage"]),
            "multi_before": _mean(one["cov_marginal_multi"]), "multi_after": _mean(mond["cov_marginal_multi"]),
            "width_before": _mean(one["width"]), "width_after": _mean(mond["width"])}


def _n3(n3):
    f = n3["part2_final"]
    cm = f["extent_confusion"]                    # stored as cm[predicted][true]
    labels = ["none", "single", "multi"]
    matrix = [[cm[p][t] for p in labels] for t in labels]     # rows = true extent, columns = predicted
    return {"accuracy": _mean(f["extent_accuracy"]), "f1": f["extent_per_class_f1"], "labels": labels,
            "matrix": matrix, "interaction_auc": _mean(f["flag_auc"]), "brier": f.get("flag_brier"),
            "brier_base_rate": f.get("flag_brier_base_rate")}


def _fresh(et):
    bt = et["by_type"]
    rows = [{"type": label, "n": bt[k]["n"], "true_pp": bt[k]["true_drop_pp"], "mae": bt[k]["MAE_fixed"],
             "coverage": bt[k]["coverage"], "width": bt[k]["width_pp"], "extent": bt[k]["extent_acc"],
             "top1": bt[k].get("top1_correct")} for k, label in FRESH_TYPES if k in bt]
    return {"n_batches": et["n_batches"], "engine_version": (et.get("engine") or {}).get("engine_version"),
            "overall": et["overall"], "no_drift": et["no_drift"], "rows": rows}


def research(dataset):
    if dataset not in FILES:
        raise KeyError(f"unknown dataset '{dataset}'")
    meta = FILES[dataset]
    s = meta["suffix"]
    n1, bl = _load(f"novelty1_results_{s}"), _load(f"baselines_results_{s}")
    n2, mo = _load(f"novelty2_results_{s}"), _load(f"mondrian_{s}")
    n3, et = _load(f"novelty3_final_results_{s}"), _load(f"engine_test_{s}")
    comp_all = _load("novelty1_model_comparison_v12", required=False)
    comp = (comp_all or {}).get("results", {}).get(dataset)
    out = {
        "dataset": dataset, "label": meta["label"], "metric": meta["metric"],
        "protocol": {"repeats": n1["protocol"].get("repeats"), "folds": n1["protocol"].get("folds"),
                     "groups": n1["protocol"].get("groups"), "conditions": 1600},
        "n1": {t: _n1(n1, bl, t, comp) for t in ("fixed", "paired")},
        "n2": {t: _n2(n2, mo, t) for t in ("fixed", "paired")},
        "n3": _n3(n3),
        "fresh": _fresh(et),
        "sources": [f"novelty1_results_{s}.json", f"baselines_results_{s}.json", f"novelty2_results_{s}.json",
                    f"mondrian_{s}.json", f"novelty3_final_results_{s}.json", f"engine_test_{s}.json"]
                   + (["novelty1_model_comparison_v12.json"] if comp else []),
    }
    return _clean(out)
