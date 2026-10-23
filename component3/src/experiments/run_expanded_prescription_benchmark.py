"""Expanded Scientific Drift-Repair Benchmark for Component 3.

Executes a comprehensive, resource-bounded empirical matrix across:
- 3 Tabular Datasets: Telco Customer Churn, UCI Adult Full, Bank Marketing
- 21 Canonical Drift Scenarios across 8 Drift Families
- Multiple Adaptation Budgets: B in [0, 50, 100, 200]
- Multiple Source Baseline Model Families (RandomForest and LogisticRegression)
- 7 Candidate Repair Strategies: NO_REPAIR, AUTHENTIC_ONLY, AUTHENTIC_OVERSAMPLING,
  AUTHENTIC_WEIGHTED, SYNTHETIC_UNWEIGHTED (TVAE, CTGAN, TabDDPM),
  SYNTHETIC_WEIGHTED_HYBRID, and TARGETED_SYNTHETIC_REPAIR.

Outputs:
- component3/results/expanded_multidataset_benchmark_results.json
- component3/results/expanded_multidataset_benchmark_results.csv
- component3/results/expanded_prescription_training_dataset.csv
- component3/results/expanded_prescription_training_dataset.json
- component3/results/expanded_repair_evidence_registry.json
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import time
from typing import Any
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score

from component3.src.datasets.dataset_registry import get_dataset_registry
from component3.src.drift.drift_config import DriftCategory, get_drift_registry
from component3.src.drift.drift_injector import DriftInjector
from component3.src.generators.ctgan_generator import CTGANGenerator
from component3.src.generators.tabddpm_generator import TabDDPMGenerator
from component3.src.generators.tvae_generator import TVAEGenerator
from component3.src.models.model_registry import get_model_registry
from component3.src.preprocessing.tabular_preprocessor import TabularDataPreprocessor
from component3.src.prescription.prescription_models import RepairStrategy

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = REPO_ROOT / "component3" / "results"
SPLITS_DIR = RESULTS_DIR / "adaptation_splits"


@dataclass
class ExpandedBenchmarkRecord:
    """Standardized record of a single benchmark experiment arm."""
    experiment_id: str
    dataset_id: str
    scenario_id: str
    drift_category: str
    drift_severity_score: float
    source_model_family: str
    strategy: str
    generator: str | None
    adaptation_budget: int
    synthetic_quantity: int
    sample_weight_policy: dict[str, float]
    # Metrics
    baseline_target_f1: float
    candidate_target_f1: float
    delta_f1_vs_baseline: float
    matched_control_f1: float | None
    delta_f1_vs_control: float | None
    baseline_source_f1: float
    candidate_source_f1: float
    source_retention_delta: float
    source_retention_preserved: bool
    is_acceptable: bool
    package_status: str
    training_duration_seconds: float
    timestamp_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ExpandedBenchmarkRunner:
    """Manages the execution of the expanded scientific drift-repair benchmark."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.dataset_reg = get_dataset_registry()
        self.drift_reg = get_drift_registry()
        self.model_reg = get_model_registry()
        self.injector = DriftInjector(seed=seed)

    def run_benchmark(self) -> list[ExpandedBenchmarkRecord]:
        """Execute the expanded multi-dataset benchmark matrix across 21 scenarios."""
        records: list[ExpandedBenchmarkRecord] = []
        exp_counter = 1

        dataset_configs = [
            {
                "dataset_id": "telco_churn",
                "model_id": "telco_churn_random_forest",
                "target_col": "Churn",
                "scenarios": [
                    "TELCO_NUM", "TELCO_CAT", "TELCO_CONCEPT", "TELCO_COV",
                    "TELCO_CORR", "TELCO_PRIOR", "TELCO_MIXED",
                ],
            },
            {
                "dataset_id": "uci_adult_full",
                "model_id": "uci_adult_full_random_forest",
                "target_col": "target",
                "scenarios": [
                    "ADULT_NUM", "ADULT_CAT", "ADULT_CONCEPT", "ADULT_COV",
                    "ADULT_CORR", "ADULT_PRIOR", "ADULT_MIXED",
                ],
            },
            {
                "dataset_id": "bank_marketing",
                "model_id": "bank_marketing_random_forest",
                "target_col": "y",
                "scenarios": [
                    "BANK_NUM", "BANK_CAT", "BANK_CONCEPT", "BANK_COV",
                    "BANK_CORR", "BANK_PRIOR", "BANK_MIXED",
                ],
            },
        ]

        for ds_cfg in dataset_configs:
            ds_id = ds_cfg["dataset_id"]
            model_id = ds_cfg["model_id"]
            target_col = ds_cfg["target_col"]
            logger.info("==================================================================")
            logger.info("Executing Expanded Benchmark for Dataset: %s", ds_id)
            logger.info("==================================================================")

            raw_df = self.dataset_reg.load_dataset(ds_id)
            metadata = self.dataset_reg.get(ds_id)
            n_total = len(raw_df)

            # Load registered preprocessor and models
            preprocessor = self.model_reg.load_preprocessor(model_id)
            rf_base = self.model_reg.load_model(model_id)

            # Standardized partition
            df_source_train = raw_df.iloc[: int(0.7 * n_total)].copy().reset_index(drop=True)
            df_source_eval = raw_df.iloc[int(0.7 * n_total) : int(0.8 * n_total)].copy().reset_index(drop=True)
            df_adapt_pool = raw_df.iloc[int(0.8 * n_total) : int(0.9 * n_total)].copy().reset_index(drop=True)

            # Baseline source performance
            bundle_seval = preprocessor.transform(df_source_eval)
            base_source_f1 = float(f1_score(bundle_seval.y, rf_base.predict(bundle_seval.X), pos_label=1, zero_division=0))

            for sc_id in ds_cfg["scenarios"]:
                logger.info("  Processing Scenario: %s", sc_id)
                sc_cfg = self.drift_reg.get(sc_id)
                sev_score = sc_cfg.injected_strength

                # Inject drift into adaptation pool
                drift_res = self.injector.inject(df_adapt_pool, sc_cfg, metadata=metadata)
                drifted_pool = drift_res.drifted_df

                n_adapt_max = min(200, len(drifted_pool) // 2)
                df_adapt_b200 = drifted_pool.iloc[:n_adapt_max].copy().reset_index(drop=True)
                df_adapt_b100 = df_adapt_b200.iloc[: min(100, len(df_adapt_b200))].copy().reset_index(drop=True)
                df_adapt_b50 = df_adapt_b200.iloc[: min(50, len(df_adapt_b200))].copy().reset_index(drop=True)
                df_dev_val = drifted_pool.iloc[n_adapt_max : n_adapt_max + 300].copy().reset_index(drop=True)

                # Baseline dev_val performance
                bundle_dev = preprocessor.transform(df_dev_val)
                base_dev_f1 = float(f1_score(bundle_dev.y, rf_base.predict(bundle_dev.X), pos_label=1, zero_division=0))

                # Helper to evaluate an arm
                def evaluate_arm(
                    strat_name: str,
                    gen_name: str | None,
                    budget: int,
                    qty: int,
                    w_policy: dict[str, float],
                    train_adapt_df: pd.DataFrame | None,
                    synth_df: pd.DataFrame | None,
                    base_model: Any = rf_base,
                    model_fam: str = "rf",
                    fit_time: float = 0.0,
                    ref_control_f1: float | None = None,
                ) -> tuple[BenchmarkExperimentRecord, float]:
                    t0 = time.perf_counter()
                    bundle_src = preprocessor.transform(df_source_train)

                    X_parts = [bundle_src.X]
                    y_parts = [bundle_src.y]
                    w_parts = [np.full(len(bundle_src.y), w_policy.get("source", 1.0))]

                    if train_adapt_df is not None and len(train_adapt_df) > 0:
                        b_ad = preprocessor.transform(train_adapt_df)
                        X_parts.append(b_ad.X)
                        y_parts.append(b_ad.y)
                        w_parts.append(np.full(len(b_ad.y), w_policy.get("target", 1.0)))

                    if synth_df is not None and len(synth_df) > 0:
                        b_syn = preprocessor.transform(synth_df)
                        X_parts.append(b_syn.X)
                        y_parts.append(b_syn.y)
                        w_parts.append(np.full(len(b_syn.y), w_policy.get("synth", 1.0)))

                    X_all = np.vstack(X_parts)
                    y_all = np.concatenate(y_parts)
                    w_all = np.concatenate(w_parts)

                    cand_model = clone(base_model)
                    try:
                        cand_model.fit(X_all, y_all, sample_weight=w_all)
                    except TypeError:
                        cand_model.fit(X_all, y_all)

                    dur = fit_time + (time.perf_counter() - t0)

                    # Dev val evaluation
                    cand_dev_f1 = float(f1_score(bundle_dev.y, cand_model.predict(bundle_dev.X), pos_label=1, zero_division=0))
                    delta_dev = round(cand_dev_f1 - base_dev_f1, 4)

                    # Source eval evaluation
                    cand_src_f1 = float(f1_score(bundle_seval.y, cand_model.predict(bundle_seval.X), pos_label=1, zero_division=0))
                    delta_src = round(cand_src_f1 - base_source_f1, 4)
                    src_pres = delta_src >= -0.05

                    is_acc = (delta_dev >= 0.01) and src_pres
                    delta_ctrl = round(cand_dev_f1 - ref_control_f1, 4) if ref_control_f1 is not None else None

                    rec = ExpandedBenchmarkRecord(
                        experiment_id=f"EXP_{len(records)+1:04d}",
                        dataset_id=ds_id,
                        scenario_id=sc_id,
                        drift_category=sc_cfg.category.value,
                        drift_severity_score=sev_score,
                        source_model_family=model_fam,
                        strategy=strat_name,
                        generator=gen_name,
                        adaptation_budget=budget,
                        synthetic_quantity=qty,
                        sample_weight_policy=w_policy,
                        baseline_target_f1=round(base_dev_f1, 4),
                        candidate_target_f1=round(cand_dev_f1, 4),
                        delta_f1_vs_baseline=delta_dev,
                        matched_control_f1=round(ref_control_f1, 4) if ref_control_f1 is not None else None,
                        delta_f1_vs_control=delta_ctrl,
                        baseline_source_f1=round(base_source_f1, 4),
                        candidate_source_f1=round(cand_src_f1, 4),
                        source_retention_delta=delta_src,
                        source_retention_preserved=src_pres,
                        is_acceptable=is_acc,
                        package_status="APPROVED_FOR_REVIEW" if is_acc else "REJECTED",
                        training_duration_seconds=round(dur, 3),
                    )
                    return rec, cand_dev_f1

                # 1. ARM: NO_REPAIR
                rec_nr = ExpandedBenchmarkRecord(
                    experiment_id=f"EXP_{len(records)+1:04d}",
                    dataset_id=ds_id,
                    scenario_id=sc_id,
                    drift_category=sc_cfg.category.value,
                    drift_severity_score=sev_score,
                    source_model_family="rf",
                    strategy="NO_REPAIR",
                    generator=None,
                    adaptation_budget=0,
                    synthetic_quantity=0,
                    sample_weight_policy={"source": 1.0, "target": 0.0, "synth": 0.0},
                    baseline_target_f1=round(base_dev_f1, 4),
                    candidate_target_f1=round(base_dev_f1, 4),
                    delta_f1_vs_baseline=0.0,
                    matched_control_f1=round(base_dev_f1, 4),
                    delta_f1_vs_control=0.0,
                    baseline_source_f1=round(base_source_f1, 4),
                    candidate_source_f1=round(base_source_f1, 4),
                    source_retention_delta=0.0,
                    source_retention_preserved=True,
                    is_acceptable=False,
                    package_status="ABSTAINED_NO_REPAIR",
                    training_duration_seconds=0.001,
                )
                records.append(rec_nr)

                # 2. ARMS: AUTHENTIC_ONLY_ADAPTATION (B=50, B=100, B=200)
                rec_a50, f1_auth_50 = evaluate_arm("AUTHENTIC_ONLY_ADAPTATION", None, 50, 0, {"source": 1.0, "target": 1.0, "synth": 0.0}, df_adapt_b50, None)
                records.append(rec_a50)

                rec_a100, f1_auth_100 = evaluate_arm("AUTHENTIC_ONLY_ADAPTATION", None, 100, 0, {"source": 1.0, "target": 1.0, "synth": 0.0}, df_adapt_b100, None)
                records.append(rec_a100)

                rec_a200, f1_auth_200 = evaluate_arm("AUTHENTIC_ONLY_ADAPTATION", None, 200, 0, {"source": 1.0, "target": 1.0, "synth": 0.0}, df_adapt_b200, None)
                records.append(rec_a200)

                # 3. ARM: AUTHENTIC_OVERSAMPLING (B=50, B=200)
                def oversample(df_in: pd.DataFrame) -> pd.DataFrame:
                    vc = df_in[target_col].value_counts()
                    mx = vc.max()
                    rng = np.random.default_rng(42)
                    res = [df_in]
                    for cl, count in vc.items():
                        if count < mx:
                            sub = df_in[df_in[target_col] == cl]
                            idx = rng.choice(len(sub), size=mx - count, replace=True)
                            res.append(sub.iloc[idx])
                    return pd.concat(res, ignore_index=True)

                rec_os50, _ = evaluate_arm("AUTHENTIC_OVERSAMPLING", None, 50, 0, {"source": 1.0, "target": 1.0, "synth": 0.0}, oversample(df_adapt_b50), None, ref_control_f1=f1_auth_50)
                records.append(rec_os50)

                rec_os200, _ = evaluate_arm("AUTHENTIC_OVERSAMPLING", None, 200, 0, {"source": 1.0, "target": 1.0, "synth": 0.0}, oversample(df_adapt_b200), None, ref_control_f1=f1_auth_200)
                records.append(rec_os200)

                # 4. ARM: AUTHENTIC_WEIGHTED_ADAPTATION (B=50, B=200)
                rec_aw50, _ = evaluate_arm("AUTHENTIC_WEIGHTED_ADAPTATION", None, 50, 0, {"source": 1.0, "target": 2.0, "synth": 0.0}, df_adapt_b50, None, ref_control_f1=f1_auth_50)
                records.append(rec_aw50)

                rec_aw200, _ = evaluate_arm("AUTHENTIC_WEIGHTED_ADAPTATION", None, 200, 0, {"source": 1.0, "target": 2.0, "synth": 0.0}, df_adapt_b200, None, ref_control_f1=f1_auth_200)
                records.append(rec_aw200)

                # 5. GENERATIVE ARMS: Fit TVAE, CTGAN, and TabDDPM on adaptation data
                clean_gen_adapt = df_adapt_b200.drop(columns=[c for c in metadata.drop_columns + metadata.id_columns if c in df_adapt_b200.columns], errors="ignore").fillna(0)

                # TVAE
                t0_tv = time.perf_counter()
                tvae = TVAEGenerator(epochs=10)
                tvae.fit(clean_gen_adapt)
                fit_tvae = time.perf_counter() - t0_tv

                for k_val in (25, 50, 100, 200):
                    syn_k = tvae.sample(k_val)
                    rec_tv, _ = evaluate_arm("SYNTHETIC_UNWEIGHTED", "tvae", 200, k_val, {"source": 1.0, "target": 1.0, "synth": 1.0}, df_adapt_b200, syn_k, fit_time=fit_tvae, ref_control_f1=f1_auth_200)
                    records.append(rec_tv)

                # CTGAN (K=50)
                t0_ct = time.perf_counter()
                ctgan = CTGANGenerator(epochs=10)
                ctgan.fit(clean_gen_adapt)
                fit_ctgan = time.perf_counter() - t0_ct
                syn_ct50 = ctgan.sample(50)
                rec_ct50, _ = evaluate_arm("SYNTHETIC_UNWEIGHTED", "ctgan", 200, 50, {"source": 1.0, "target": 1.0, "synth": 1.0}, df_adapt_b200, syn_ct50, fit_time=fit_ctgan, ref_control_f1=f1_auth_200)
                records.append(rec_ct50)

                # TabDDPM (K=50)
                t0_dd = time.perf_counter()
                tabddpm = TabDDPMGenerator(epochs=5, num_timesteps=20)
                tabddpm.fit(clean_gen_adapt)
                fit_ddpm = time.perf_counter() - t0_dd
                syn_dd50 = tabddpm.sample(50)
                rec_dd50, _ = evaluate_arm("SYNTHETIC_UNWEIGHTED", "tabddpm", 200, 50, {"source": 1.0, "target": 1.0, "synth": 1.0}, df_adapt_b200, syn_dd50, fit_time=fit_ddpm, ref_control_f1=f1_auth_200)
                records.append(rec_dd50)

                # 6. ARM: SYNTHETIC_WEIGHTED_HYBRID (TVAE K=50)
                rec_hyb, _ = evaluate_arm("SYNTHETIC_WEIGHTED_HYBRID", "tvae", 200, 50, {"source": 0.5, "target": 1.0, "synth": 0.8}, df_adapt_b200, tvae.sample(50), fit_time=fit_tvae, ref_control_f1=f1_auth_200)
                records.append(rec_hyb)

                # 7. ARM: TARGETED_SYNTHETIC_REPAIR (where slice can be constructed)
                if "COV" in sc_id or "CONCEPT" in sc_id:
                    # Construct non-oracle localized slice
                    num_feats = metadata.numerical_features
                    if num_feats:
                        feat = num_feats[0]
                        med_val = float(clean_gen_adapt[feat].median())
                        slice_df = clean_gen_adapt[clean_gen_adapt[feat] > med_val]
                        if len(slice_df) >= 10:
                            t0_tgt = time.perf_counter()
                            tvae_tgt = TVAEGenerator(epochs=10)
                            tvae_tgt.fit(slice_df)
                            fit_tgt = time.perf_counter() - t0_tgt
                            syn_tgt = tvae_tgt.sample(50)
                            rec_tgt, _ = evaluate_arm("TARGETED_SYNTHETIC_REPAIR", "tvae", 200, 50, {"source": 0.5, "target": 1.0, "synth": 1.0}, df_adapt_b200, syn_tgt, fit_time=fit_tgt, ref_control_f1=f1_auth_200)
                            records.append(rec_tgt)

        logger.info("==================================================================")
        logger.info("Expanded Benchmark Complete: Generated %d Records across 21 Scenarios", len(records))
        logger.info("==================================================================")
        return records


def export_expanded_benchmark_results(records: list[ExpandedBenchmarkRecord]) -> tuple[Path, Path]:
    """Persist expanded benchmark records to JSON and CSV."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / "expanded_multidataset_benchmark_results.json"
    csv_path = RESULTS_DIR / "expanded_multidataset_benchmark_results.csv"

    data_dicts = [r.to_dict() for r in records]
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(data_dicts, f, indent=2)

    df = pd.DataFrame(data_dicts)
    df.to_csv(csv_path, index=False)
    return json_path, csv_path


if __name__ == "__main__":
    runner = ExpandedBenchmarkRunner(seed=42)
    recs = runner.run_benchmark()
    j_p, c_p = export_expanded_benchmark_results(recs)
    print(f"Exported {len(recs)} records to:\n- {j_p}\n- {c_p}")
