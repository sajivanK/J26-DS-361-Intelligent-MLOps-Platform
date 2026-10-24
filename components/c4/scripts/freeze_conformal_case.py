"""Freeze simulated decision-time evidence for the admitted conformal case.

Does not train, deploy, evaluate future data, or alter original repair approval.
"""
import io
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from _common import (parser, resolve_data_root, checked_json, checked_bytes,
                     require, save_json, sha, utc_now, git_context, C4_ROOT, REPO_ROOT)
from _scoped_benchmark import frozen_tail_screen
from _conformal_pipeline import group_context
from _admitted_training import verify_corruption
from c4.contracts.models import UpstreamInput
from c4.data import FEATURE_COLUMNS
from _portable_model_io import load_champion, load_candidate
from c4.scenarios.slices import SliceRule
from c4.scenarios.config import CONTINUOUS_FEATURES
from c4.scenarios.diagnosis import diagnosed_slice_mask


def main():
    p = parser(__doc__)
    p.add_argument('--freeze', action='store_true', help='Write one immutable simulation evidence freeze.')
    p.add_argument('--scenario-id', required=True)
    args = p.parse_args()
    SELECTED = args.scenario_id
    root = resolve_data_root(args.data_root)
    from _conformal_group import verify_n1
    n1_binding = verify_n1(root)
    dependencies = {}

    def remember(path, expected=None):
        path = Path(path)
        contents = path.read_bytes() if expected is None else checked_bytes(path, expected)
        dependencies[str(path)] = sha(contents)
        return contents

    def document(directory, stem='metadata'):
        payload, checksum = checked_json(directory, stem)
        remember(Path(directory) / (stem + '.json'), checksum)
        remember(Path(directory) / (stem + '.sha256'))
        return payload, checksum

    n1_directory = root / 'c4_artifacts/forecasting/adult/adult-simulation-pp1-v2/adult-n1-training-v1/n1-fit-v1'
    n1_metadata, n1_metadata_hash = document(n1_directory)
    require(n1_metadata_hash == n1_binding['run_metadata_sha256'], 'Wrong frozen N1 run.')
    document(n1_directory.parent, 'specification')
    for filename, checksum in n1_metadata['files'].items():
        remember(n1_directory / filename, checksum)
    frozen_dir = root / 'c4_artifacts/benchmark_protocols/adult/adult-simulation-pp1-v2'
    frozen, freeze_hash = document(frozen_dir, 'freeze')
    protocol = json.loads(remember(frozen_dir / 'protocol.json', frozen['protocol_sha256']))
    manifest, manifest_hash = document(frozen_dir, 'scenario_manifest')
    require(manifest_hash == frozen['scenario_manifest_sha256'], 'Scenario manifest differs.')
    rows = [r for r in manifest['scenarios'] if r['scenario_id'] == SELECTED]
    require(len(rows) == 1 and rows[0]['split_role'] == 'conformal_calibration', 'Wrong scenario scope.')
    row = rows[0]
    run = root / 'c4_artifacts/benchmark_runs/adult/adult-simulation-pp1-v2' / SELECTED
    validation, validation_hash = document(run / 'scoped_candidate_validation-v1', 'validation')
    admission, admission_hash = document(run, 'admission')
    require(validation['scenario'] == row and admission['scenario'] == row, 'Scenario links differ.')
    require(validation['admission_sha256'] == admission_hash and
            validation['freeze_sha256'] == admission['freeze_sha256'] == freeze_hash, 'Freeze/admission links differ.')
    require(validation['candidate_training_executed'] is True and
            validation['future_evaluation_executed'] is False, 'Wrong execution state.')
    require(admission['simulation_admission'] == 'READY_FOR_SCOPED_CANDIDATE_VALIDATION', 'Admission failed.')
    panel, panel_hash, _ = group_context(root, SELECTED)
    require(panel_hash == validation['group_admissions_sha256'], 'Group link differs.')
    remember(run.parent / 'group_admissions' / panel['group_id'] / 'v1/group_admissions.json', panel_hash)
    paths = {'scenario_adapter': C4_ROOT / 'src/c4/scenarios/adapter.py',
             'tvae_generation': C4_ROOT / 'src/c4/experiments/tvae_generation.py',
             'distance_metric': C4_ROOT / 'scripts/_privacy_distance.py',
             'shared_C1_mock': REPO_ROOT / 'shared/contracts/c1_fake_output.py',
             'C4_contracts': C4_ROOT / 'src/c4/contracts/models.py',
             'manifest_builder': C4_ROOT / 'scripts/_benchmark_manifest.py'}
    for key, path in paths.items():
        remember(path, frozen['source_implementation_sha256'][key])
    candidate_dir = run / 'scoped_candidate_validation-v1'
    candidate_md, candidate_md_hash = document(candidate_dir)
    require(candidate_md_hash == validation['candidate_metadata_sha256'], 'Candidate metadata link differs.')
    remember(candidate_dir / 'claim.json')
    require(candidate_md['scope'] == 'controlled_offline_simulation' and
            candidate_md['c3_package']['production_package_approved'] is False,
            'Candidate simulation scope differs.')
    champion_dir = root / 'c4_artifacts/champion/adult' / protocol['source_versions']['champion']
    champion, champion_md = load_champion(champion_dir)
    candidate, _ = load_candidate(candidate_dir)
    remember(champion_dir / 'metadata.json')
    remember(champion_dir / 'champion.ubj', frozen['champion_model_sha256'])
    remember(candidate_dir / 'candidate.ubj', candidate_md['model_sha256'])
    require(candidate_md['champion_model_sha256'] == champion_md['model_sha256'], 'Different champion.')
    reference = pd.read_csv(io.BytesIO(remember(root / 'uci_adult_data/reference.csv', champion_md['reference_csv_sha256'])))
    md = root / 'c4_artifacts/splits/adult' / protocol['source_versions']['split_manifest']
    source, source_hash = document(md, 'manifest')
    require(source_hash == frozen['split_manifest_sha256'], 'Split manifest differs.')
    remember(root / 'uci_adult_data/test.csv', source['source_checksums']['test_csv_sha256'])
    package = root / 'c4_artifacts/offline_repair/adult' / admission['artifact_version']
    generation, generation_hash = document(package)
    require(generation_hash == admission['generation_metadata_sha256'], 'Generation link differs.')
    for path, checksum in verify_corruption(root, package, generation).items():
        remember(path, checksum)
    remember(package / 'synthetic_unvalidated.csv', generation['synthetic_csv_sha256'])
    remember(package / 'generator.pkl', generation['generator_sha256'])
    diagnostics, _ = document(package, 'diagnostics')
    distribution, _ = document(package, 'distribution_audit')
    privacy, privacy_hash = document(package, 'privacy_diagnostics')
    require(privacy_hash == admission['privacy_report_sha256'], 'Privacy report link differs.')
    for report in (diagnostics, distribution, privacy):
        require(report['generation_metadata_sha256'] == generation_hash, 'Diagnostic generation link differs.')
    remember(package / 'utility_probe.ubj', diagnostics['probe_model_sha256'])
    prep_dir = root / 'c4_artifacts/repair_preparation/adult' / admission['preview_id'] / admission['preparation_version']
    prep, prep_hash = document(prep_dir)
    require(prep_hash == generation['preparation_metadata_sha256'], 'Preparation link differs.')
    for filename, checksum in prep['files'].items():
        remember(prep_dir / filename, checksum)
    fit = pd.read_csv(io.BytesIO((prep_dir / 'generator_fit.csv').read_bytes()), index_col=0)
    synthetic = pd.read_csv(io.BytesIO((package / 'synthetic_unvalidated.csv').read_bytes()))
    require(synthetic.columns.tolist() == [*FEATURE_COLUMNS, 'target'] and
            np.isfinite(synthetic.to_numpy(dtype=float)).all(), 'Synthetic schema or finite values differ.')
    for column in generation['discrete_columns']:
        require(synthetic[column].isin(fit[column].unique()).all(), 'Synthetic discrete domain differs.')

    for report in (diagnostics, distribution):
        require(report['preparation_metadata_sha256'] == prep_hash, 'Diagnostic preparation link differs.')
    screen = frozen_tail_screen(protocol, privacy['summaries']['synthetic_to_generator_fit'],
            privacy['summaries']['diagnosed_internal_validation_to_generator_fit'], protocol_kind='benchmark')
    require(screen['screen_status'] == 'PASS', 'Frozen screen failed on reverification.')
    evidence_dir = root / 'c4_artifacts/pre_release/adult' / SELECTED
    evidence, evidence_hash = document(evidence_dir, 'evidence')
    require(evidence_hash == validation['pre_release_evidence_sha256'], 'Pre-release evidence differs.')
    require(evidence['candidate_model_sha256'] == candidate_md['model_sha256'] and
            evidence['repair_csv_sha256'] == generation['synthetic_csv_sha256'], 'Model/repair evidence links differ.')
    frames = {}
    for filename, key in [('clean_release_validation.csv', 'clean_release_validation_csv_sha256'),
                          ('release_validation.csv', 'release_validation_csv_sha256'),
                          ('diagnosed_membership.csv', 'diagnosed_membership_csv_sha256')]:
        contents = remember(evidence_dir / filename, evidence[key])
        frames[filename] = pd.read_csv(io.BytesIO(contents), float_precision='round_trip')
    clean = frames['clean_release_validation.csv'].set_index('source_row_id')
    transformed = frames['release_validation.csv'].set_index('source_row_id')
    require(clean.index.tolist() == source['partitions']['release_validation']['source_row_ids'] and
            clean.index.equals(transformed.index), 'Release membership differs.')
    diagnosis = admission['simulated_c2_diagnosis']
    from c4.contracts.models import C2Diagnosis
    c2 = C2Diagnosis.model_validate(diagnosis)
    require(c2.diagnosis_stability >= .50, 'Diagnosis stability failed.')
    diagnosed_mask = diagnosed_slice_mask(clean, c2, rule_feature_space='processed')
    membership = frames['diagnosed_membership.csv']
    require(membership['source_row_id'].tolist() == clean.index.tolist() and
            membership['diagnosed_slice_member'].tolist() == diagnosed_mask.tolist(), 'Diagnosed membership differs.')
    true = protocol['true_slice']
    true_rule = SliceRule(name=true['name'], feature=true['feature'], operator=true['operator'], value=true['resolved_threshold'])
    true_mask = true_rule.mask(clean)
    baseline = float(np.mean(champion.predict(clean.loc[:, list(FEATURE_COLUMNS)]) == clean['target']))
    drifted = float(np.mean(champion.predict(transformed.loc[:, list(FEATURE_COLUMNS)]) == transformed['target']))
    sys.path.insert(0, str(REPO_ROOT / 'shared'))
    from contracts.c1_fake_output import build_fake_c1_output
    gt = {'drift_type': row['mechanism'], 'actual_accuracy_drop': 100 * (baseline - drifted),
          'batch_size': len(clean), 'injected_features': row['features'],
          'per_feature_ks': {f: float(ks_2samp(clean[f], transformed[f]).statistic) for f in CONTINUOUS_FEATURES}}
    c1 = build_fake_c1_output(gt, seed=row['c1_mock_seed'],
            model_id='adult-xgb-' + champion_md['artifact_version'],
            reference_dataset_uri=str(root / 'uci_adult_data/reference.csv'),
            current_dataset_uri=str(evidence_dir / 'release_validation.csv'))
    fidelity = diagnostics['mean_univariate_fidelity']
    coverage = distribution['processed_rule_coverage']['synthetic']['rule_coverage']
    tv = distribution['synthetic_target_comparisons']['generator_fit']['target_total_variation']
    c3 = {'repair_dataset_uri': str(package / 'synthetic_unvalidated.csv'),
          'repair_package_status': 'APPROVED', 'generator': 'TVAE',
          'n_synthetic_rows': generation['synthetic_rows'],
          'augmentation_ratio': generation['synthetic_rows'] / len(reference),
          'fidelity_score': fidelity,
          'utility_score': diagnostics['internal_utility_comparisons']['internal_diagnosed_slice']['accuracy_margin'],
          'privacy_status': 'PASS', 'target_coverage': coverage,
          'prescription_confidence': float(np.mean([fidelity, coverage, 1-tv])),
          'repair_version': admission['artifact_version']}
    upstream = UpstreamInput.model_validate({'c1_evidence': c1, 'c2_diagnosis': diagnosis, 'c3_repair_package': c3})
    sample = reference.loc[:, list(FEATURE_COLUMNS)].head(10)
    require(champion.get_booster().feature_names == candidate.get_booster().feature_names == list(FEATURE_COLUMNS), 'Feature schema differs.')
    with tempfile.TemporaryDirectory(prefix='c4-freeze-artifact-restore-') as temp:
        for filename in ('metadata.json', 'champion.ubj'):
            (Path(temp) / filename).write_bytes((champion_dir / filename).read_bytes())
        restored, _ = load_champion(temp)
        require(np.array_equal(champion.predict(sample), restored.predict(sample)) and
                np.allclose(champion.predict_proba(sample), restored.predict_proba(sample)), 'Artifact restore failed.')
    preview = root / 'c4_artifacts/scenario_previews/adult' / admission['preview_id'] / 'release_validation'
    preview_md, preview_hash = document(preview)
    require(preview_hash == validation['release_preview_metadata_sha256'], 'Release preview differs.')
    for filename, checksum in preview_md['files'].items():
        remember(preview / filename, checksum)
    # Pin the exact implementations used for training and pre-release scoring.
    claim = json.loads((candidate_dir / 'claim.json').read_bytes())
    for relative, checksum in claim['source_sha256'].items():
        remember(C4_ROOT / relative, checksum)
    for path, checksum in validation.get('recovery_source_sha256', {}).items():
        remember(path, checksum)
    for filename in ('_portable_model_io.py', 'evaluate_conformal_case_future.py',
                     '_all_mechanism_transforms.py', '_future_research.py', '_conformal_pipeline.py', '_conformal_group.py', '_scenario_meta.py'):
        remember(C4_ROOT / 'scripts' / filename)
    remember(Path(__file__))
    import joblib
    from _scenario_meta import feature_row
    feature = feature_row(upstream.model_dump(mode='json'), evidence)
    require(len(feature) == 75, 'N1 decision-time feature schema differs.')
    feature_frame = pd.DataFrame([feature], index=pd.Index([SELECTED], name='scenario_id'))
    n1_forecasts = {}
    for method in ('lightgbm', 'linear'):
        fitted = joblib.load(io.BytesIO(remember(n1_directory / (method + '.joblib'), n1_metadata['files'][method + '.joblib'])))
        n1_forecasts[method] = {key: bool(value) if isinstance(value, (bool, np.bool_)) else float(value)
                                for key, value in fitted.predict(feature_frame).iloc[0].to_dict().items()}
    require(verify_n1(root) == n1_binding, 'N1 changed before evidence freeze.')
    print('FIXED N1 FORECASTS GENERATED FROM PRE-RELEASE EVIDENCE: PASS')
    print('UPSTREAM CONTRACT / MEASURED SCORE / ARTIFACT LINKS: PASS')
    print('DIAGNOSED ROWS:', int(diagnosed_mask.sum()), 'TRUE AFFECTED ROWS:', int(true_mask.sum()))
    print('SIMULATION ONLY; ORIGINAL C3 PRIVACY/APPROVAL METADATA UNCHANGED')
    if not args.freeze:
        print('PLAN ONLY. Use --freeze to persist this complete simulation decision-evidence snapshot.')
        return
    output = run / 'pre_release_freeze-v1'
    require(not output.exists(), 'Decision evidence freeze already exists; preserve it.')
    output.mkdir(parents=True, exist_ok=False)
    envelope = {'scope': 'controlled_offline_simulation', 'source_declarations': {'c1': 'mock_oracle', 'c2': 'simulated', 'c3': 'offline_simulated'},
        'actual_producer_performance_claims_allowed': False, 'production_package_approved': False,
        'privacy_interpretation': 'PASS refers only to the frozen exploratory simulation screen, not privacy certification.',
        'prescription_confidence_origin': 'computed_simulation_proxy_not_real_C3_prescription',
        'repair_timestamp': generation['created_at_utc'], 'upstream': upstream.model_dump(mode='json')}
    forecast_document = {'purpose': 'fixed_n1_forecasts_before_conformal_outcomes', 'scenario_id': SELECTED,
        'n1_binding': n1_binding, 'feature_columns': list(feature), 'decision_time_features': feature,
        'forecasts': n1_forecasts, 'N2_calibrated': False, 'future_targets_included': False}
    forecast_hash = save_json(output / 'n1_forecasts.json', forecast_document)
    remember(output / 'n1_forecasts.json', forecast_hash)
    envelope_hash = save_json(output / 'upstream_simulation.json', envelope)
    membership_bytes = pd.DataFrame({'source_row_id': clean.index, 'true_slice_member': true_mask.to_numpy(),
        'diagnosed_slice_member': diagnosed_mask.to_numpy()}).to_csv(index=False, lineterminator='\n').encode()
    (output / 'release_membership.csv').write_bytes(membership_bytes)
    remember(output / 'upstream_simulation.json', envelope_hash)
    remember(output / 'release_membership.csv', sha(membership_bytes))
    # Recheck every dependency immediately before committing the freeze marker.
    for path, checksum in dependencies.items():
        checked_bytes(path, checksum)
    result = {'purpose': 'complete_simulation_decision_evidence_freeze',
        'model_io_mode': 'verified_bytearray_and_byte_identical_short_temporary_mirrors', 'created_at_utc': utc_now(), **git_context(),
        'scenario': row, 'n1_binding': n1_binding, 'n1_run_directory': str(n1_directory),
        'n1_forecasts_frozen': True, 'n1_forecasts_path': str(output / 'n1_forecasts.json'),
        'n1_forecasts_sha256': forecast_hash, 'protocol_freeze_sha256': freeze_hash, 'pre_release_evidence_sha256': evidence_hash,
        'upstream_simulation_sha256': envelope_hash, 'release_membership_sha256': sha(membership_bytes),
        'runtime_checks': {'schema_compatible': True, 'rollback_ready': True,
                           'rollback_scope': 'verified_offline_artifact_restore_not_live_production'},
        'dependencies': dependencies, 'complete_decision_evidence_frozen': True,
        'future_evaluation_executed': False, 'production_package_approved': False,
        'release_authorized': False, 'N1_N4_controller_complete': False,
        'scope': 'one_admitted_conformal_scenario_under_simulated_upstream_evidence'}
    checksum = save_json(output / 'freeze.json', result)
    print('COMPLETE SIMULATION DECISION-EVIDENCE FREEZE: PASS')
    print('REPORT:', output / 'freeze.json')
    print('REPORT SHA256:', checksum)
    print('FUTURE EVALUATION: False; RELEASE AUTHORIZED: False')


if __name__ == '__main__':
    main()
