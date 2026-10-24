"""Train and evaluate an admitted case of any frozen mechanism in a frozen policy group.

No production C3 approval is fabricated. No sealed-future evaluator is called.
Pre-release metrics are saved; complete decision-evidence freeze is a later step.
"""
import io
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from _common import (parser, resolve_data_root, checked_json, checked_bytes,
                     require, save_json, sha, git_context, utc_now,
                     manifest_root, C4_ROOT, REPO_ROOT)
from _scoped_benchmark import frozen_tail_screen
from c4.contracts.models import C2Diagnosis
from c4.data import FEATURE_COLUMNS, TARGET_COLUMN
from c4.data.split_manifest import (load_adult_split_manifest, assert_release_membership,
                                    verify_clean_release_source)
from save_role_previews import verify_preview
from c4.models import ChampionConfig, train_champion
from _portable_model_io import load_champion, load_candidate, save_pre_release_evidence

from _policy_pipeline import group_context
from _admitted_training import verify_corruption


def validate_frame(frame, label):
    require(frame.columns.tolist() == [*FEATURE_COLUMNS, TARGET_COLUMN],
            f'{label}: invalid feature schema.')
    require(len(frame) > 0 and all(pd.api.types.is_numeric_dtype(frame[c])
            for c in frame.columns), f'{label}: empty or nonnumeric data.')
    require(np.isfinite(frame.to_numpy(dtype=float)).all(), f'{label}: nonfinite values.')
    require(set(frame[TARGET_COLUMN].unique()) == {0, 1},
            f'{label}: both binary classes are required.')


def main():
    p = parser(__doc__)
    p.add_argument('--execute', action='store_true', help='Train this admitted simulation case only.')
    p.add_argument('--scenario-id', required=True)
    args = p.parse_args()
    SELECTED = args.scenario_id
    root = resolve_data_root(args.data_root)
    from _policy_group import verify_n1
    n1_binding = verify_n1(root)
    frozen_dir = root / 'c4_artifacts/benchmark_protocols/adult/adult-simulation-pp1-v2'
    frozen, freeze_hash = checked_json(frozen_dir, 'freeze')
    protocol = json.loads(checked_bytes(frozen_dir / 'protocol.json', frozen['protocol_sha256']))
    manifest, manifest_hash = checked_json(frozen_dir, 'scenario_manifest')
    require(manifest_hash == frozen['scenario_manifest_sha256'], 'Scenario manifest differs.')
    require(frozen['full_execution_enabled'] is False and protocol['full_execution_enabled'] is False,
            'Only scoped execution is supported.')
    rows = [r for r in manifest['scenarios'] if r['scenario_id'] == SELECTED]
    require(len(rows) == 1 and rows[0]['split_role'] == 'policy_calibration', 'Wrong scenario scope.')
    row = rows[0]
    run_dir = root / 'c4_artifacts/benchmark_runs/adult/adult-simulation-pp1-v2' / SELECTED
    panel, panel_hash, group_record = group_context(root, SELECTED)
    admission, admission_hash = checked_json(run_dir, 'admission')
    require(admission['scenario'] == row and admission['freeze_sha256'] == freeze_hash,
            'Admission scenario or protocol link differs.')
    require(admission['simulation_admission'] == 'READY_FOR_SCOPED_CANDIDATE_VALIDATION',
            'Scenario has not passed simulation admission.')
    require(admission['production_package_approved'] is False, 'Unexpected production approval.')
    require(admission['artifact_schema_checked'] is True and admission['offline_champion_restore_checked'] is True,
            'Admission runtime checks are missing.')
    # Check the full panel is complete and includes this exact admission record.
    records = panel.get('results', panel.get('records', []))
    require(panel['freeze_sha256'] == freeze_hash and panel['cases'] == 37, 'Group protocol or case count differs.')
    require(any(r.get('scenario_id') == SELECTED and r.get('admission_sha256') == admission_hash
                for r in records), 'Panel does not link this admission report.')
    paths = {
        'scenario_adapter': C4_ROOT / 'src/c4/scenarios/adapter.py',
        'tvae_generation': C4_ROOT / 'src/c4/experiments/tvae_generation.py',
        'distance_metric': C4_ROOT / 'scripts/_privacy_distance.py',
        'shared_C1_mock': REPO_ROOT / 'shared/contracts/c1_fake_output.py',
        'C4_contracts': C4_ROOT / 'src/c4/contracts/models.py',
        'manifest_builder': C4_ROOT / 'scripts/_benchmark_manifest.py',
    }
    for key, path in paths.items():
        checked_bytes(path, frozen['source_implementation_sha256'][key])
    md = manifest_root(root, protocol['source_versions']['split_manifest'])
    checked_bytes(md / 'manifest.json', frozen['split_manifest_sha256'])
    source = load_adult_split_manifest(md, data_root=root)
    repair_dir = root / 'c4_artifacts/offline_repair/adult' / admission['artifact_version']
    generation, generation_hash = checked_json(repair_dir)
    require(generation_hash == admission['generation_metadata_sha256'], 'Generation link differs.')
    corruption_links = verify_corruption(root, repair_dir, generation)
    privacy, privacy_hash = checked_json(repair_dir, 'privacy_diagnostics')
    require(privacy_hash == admission['privacy_report_sha256'] and
            privacy['generation_metadata_sha256'] == generation_hash, 'Privacy report link differs.')
    screen = frozen_tail_screen(protocol, privacy['summaries']['synthetic_to_generator_fit'],
                privacy['summaries']['diagnosed_internal_validation_to_generator_fit'], protocol_kind='benchmark')
    diagnosis = C2Diagnosis.model_validate(admission['simulated_c2_diagnosis'])
    require(screen['screen_status'] == 'PASS' and diagnosis.diagnosis_stability >= .50,
            'Rechecked simulation screen or stability failed.')
    prep_dir = root / 'c4_artifacts/repair_preparation/adult' / admission['preview_id'] / admission['preparation_version']
    prep, prep_hash = checked_json(prep_dir)
    require(prep_hash == generation['preparation_metadata_sha256'], 'Preparation link differs.')
    require(prep['benchmark_context']['scenario_id'] == SELECTED, 'Preparation scenario differs.')
    require(prep['provenance']['split_manifest_sha256'] == frozen['split_manifest_sha256'], 'Preparation split differs.')
    ids = set(source['partitions']['repair']['source_row_ids'])
    fit_ids, validation_ids = generation['generator_fit_row_ids'], generation['internal_validation_row_ids']
    require(set(fit_ids) <= ids and set(validation_ids) <= ids and not set(fit_ids) & set(validation_ids),
            'Repair fitting/validation provenance is invalid.')
    for filename, checksum in prep['files'].items():
        checked_bytes(prep_dir / filename, checksum)
    repair_bytes = checked_bytes(repair_dir / 'synthetic_unvalidated.csv', generation['synthetic_csv_sha256'])
    repair = pd.read_csv(io.BytesIO(repair_bytes))
    validate_frame(repair, 'Repair')
    require(len(repair) == generation['synthetic_rows'] == protocol['repair_generation']['n_rows'], 'Repair row count differs.')
    champion_dir = root / 'c4_artifacts/champion/adult' / protocol['source_versions']['champion']
    champion, champion_md = load_champion(champion_dir)
    require(champion_md['model_sha256'] == frozen['champion_model_sha256'], 'Champion differs.')
    reference_bytes = checked_bytes(root / 'uci_adult_data/reference.csv', champion_md['reference_csv_sha256'])
    reference = pd.read_csv(io.BytesIO(reference_bytes))
    validate_frame(reference, 'Reference')
    preview_dir = root / 'c4_artifacts/scenario_previews/adult' / admission['preview_id'] / 'release_validation'
    verify_preview(preview_dir, root, md, 'release_validation')
    preview, preview_hash = checked_json(preview_dir)
    frames = {filename: pd.read_csv(io.BytesIO(checked_bytes(preview_dir / filename, preview['files'][filename])), index_col=0)
              for filename in ('clean.csv', 'transformed.csv')}
    require(frames['clean.csv'].index.equals(frames['transformed.csv'].index), 'Release row identity differs.')
    require(np.array_equal(frames['clean.csv'][TARGET_COLUMN], frames['transformed.csv'][TARGET_COLUMN]), 'Release targets changed.')
    for filename in frames:
        validate_frame(frames[filename], filename)
    assert_release_membership(frames['clean.csv'], source)
    assert_release_membership(frames['transformed.csv'], source)
    # CSV parsing can move a float by one ULP after a to_csv/read_csv round trip.
    # Verify the exact source serialization BEFORE restoring canonical floats.
    # Exact replay supports each frozen mechanism without tolerance-based bypasses.
    test_bytes = checked_bytes(root / 'uci_adult_data/test.csv',
                               source['source_checksums']['test_csv_sha256'])
    selected_ids = set(source['partitions']['release_validation']['source_row_ids'])
    # Match save_role_preview exactly: role-specific parsing and unnamed index.
    original_role = pd.read_csv(io.BytesIO(test_bytes), skiprows=lambda line_number: (
        line_number > 0 and line_number - 1 not in selected_ids))
    require(len(original_role) == len(selected_ids), 'Source record layout differs.')
    original_role.index = sorted(selected_ids)
    original_role = original_role.loc[frames['clean.csv'].index].copy()
    expected_clean_bytes = original_role.to_csv(index=True,
                                                lineterminator='\n').encode('utf-8')
    # The strict source verifier uses the full original CSV parsing procedure.
    source_test = pd.read_csv(io.BytesIO(test_bytes))
    canonical_clean = source_test.loc[frames['clean.csv'].index].copy()
    actual_clean_bytes = checked_bytes(preview_dir / 'clean.csv', preview['files']['clean.csv'])
    require(expected_clean_bytes == actual_clean_bytes,
            'Saved clean preview is not the exact serialization of original source rows.')
    from _all_mechanism_transforms import canonical_release_frames
    from c4.scenarios.config import ScenarioConfig
    from c4.scenarios.slices import SliceRule
    true = protocol['true_slice']
    rule = SliceRule(name=true['name'], feature=true['feature'], operator=true['operator'],
                     value=true['resolved_threshold'])
    drift_config = ScenarioConfig(mechanism=row['mechanism'], magnitude=row['magnitude'],
        features=tuple(row['features']), interaction_pairs=tuple(tuple(p) for p in row['interaction_pairs']),
        seed=row['drift_seeds']['release_validation'])
    transformed_bytes = checked_bytes(preview_dir / 'transformed.csv', preview['files']['transformed.csv'])
    frames['clean.csv'], frames['transformed.csv'] = canonical_release_frames(
        original_role, canonical_clean, actual_clean_bytes, transformed_bytes, config=drift_config, true_rule=rule)
    verify_clean_release_source(frames['clean.csv'], source, data_root=root)
    print('EXACT CLEAN CSV SERIALIZATION / CANONICAL SOURCE RECONSTRUCTION: PASS')
    print('SCENARIO:', SELECTED)
    print('ADMISSION / PROVENANCE / SOURCE CHECKS: PASS')
    print('SCOPE: controlled simulation; production approval remains False')
    print('TRAINING ROWS:', len(reference) + len(repair))
    if not args.execute:
        print('PLAN ONLY. Use --execute to train and save pre-release metrics.')
        return
    directory = run_dir / 'scoped_candidate_validation-v1'
    require(not directory.exists(), 'Scoped candidate validation already claimed; preserve existing output.')
    require(not (root / 'c4_artifacts/pre_release/adult' / SELECTED).exists(), 'Pre-release evidence already exists.')
    directory.mkdir(parents=True, exist_ok=False)
    code_hashes = {str(path.relative_to(C4_ROOT)): sha(path.read_bytes()) for path in
                  [Path(__file__), C4_ROOT / 'src/c4/models/champion.py',
                   C4_ROOT / 'src/c4/models/candidate.py', C4_ROOT / 'src/c4/models/pre_release.py',
                   C4_ROOT / 'src/c4/models/evaluation.py', C4_ROOT / 'scripts/_admitted_training.py', C4_ROOT / 'scripts/_policy_pipeline.py', C4_ROOT / 'scripts/_all_mechanism_transforms.py', C4_ROOT / 'scripts/_portable_model_io.py',
                   C4_ROOT / 'scripts/_training_group.py', C4_ROOT / 'scripts/_policy_group.py', C4_ROOT / 'scripts/_scenario_meta.py']}
    save_json(directory / 'claim.json', {'scenario_id': SELECTED, 'created_at_utc': utc_now(), **git_context(),
        'source_sha256': code_hashes, 'admission_sha256': admission_hash, 'freeze_sha256': freeze_hash,
        'production_package_approved': False, 'future_evaluation_executed': False, 'n1_binding': n1_binding})
    try:
        config = ChampionConfig(**champion_md['training_config'])
        training = pd.concat([reference, repair], ignore_index=True)
        model = train_champion(training, config)
        import tempfile
        with tempfile.TemporaryDirectory(prefix='c4-save-') as temporary:
            short_path = Path(temporary) / 'candidate.ubj'
            model.save_model(short_path)
            (directory / 'candidate.ubj').write_bytes(short_path.read_bytes())
        metadata = {'artifact_schema_version': '1.0', 'artifact_version': admission['preview_id'] + '-simulation-candidate-v1',
            'role': 'candidate', 'scope': 'controlled_offline_simulation', 'scenario_id': SELECTED,
            'created_at_utc': utc_now(), **git_context(), 'training_config': asdict(config),
            'feature_columns': list(FEATURE_COLUMNS), 'target_column': TARGET_COLUMN,
            'champion_model_sha256': champion_md['model_sha256'], 'champion_version': champion_md['artifact_version'],
            'model_sha256': sha((directory / 'candidate.ubj').read_bytes()),
            'reference_csv_sha256': sha(reference_bytes), 'repair_csv_sha256': sha(repair_bytes),
            'reference_rows': len(reference), 'repair_rows': len(repair), 'training_rows': len(training),
            'c3_package': {'repair_version': admission['artifact_version'], 'source': 'offline_simulation',
                           'production_package_approved': False, 'privacy_status': 'NOT_EVALUATED'},
            'admission_sha256': admission_hash, 'freeze_sha256': freeze_hash,
            'generation_metadata_sha256': generation_hash, 'preparation_metadata_sha256': prep_hash,
            'release_validation_used_for_training': False, 'sealed_future_used_for_training': False,
            'environment': champion_md.get('environment', {}),
            'environment_note': 'Champion environment retained as source provenance; candidate library versions below.',
        }
        import platform, sklearn, xgboost
        metadata['candidate_environment'] = {'python': platform.python_version(), 'numpy': np.__version__,
            'pandas': pd.__version__, 'scikit_learn': sklearn.__version__, 'xgboost': xgboost.__version__}
        save_json(directory / 'metadata.json', metadata)
        restored, _ = load_candidate(directory)
        sample = reference.loc[:, list(FEATURE_COLUMNS)].head(10)
        require(np.array_equal(model.predict(sample), restored.predict(sample)) and
                np.allclose(model.predict_proba(sample), restored.predict_proba(sample)), 'Candidate reload differs.')
        artifact = save_pre_release_evidence(champion_directory=champion_dir, candidate_directory=directory,
            split_manifest_directory=md, clean_release_validation=frames['clean.csv'],
            release_validation=frames['transformed.csv'], diagnosis=diagnosis, rule_feature_space='processed',
            scenario_id=SELECTED, data_root=root)
        evidence, evidence_hash = checked_json(artifact.evidence_path.parent, 'evidence')
        for filename, key in [('clean_release_validation.csv', 'clean_release_validation_csv_sha256'),
                              ('release_validation.csv', 'release_validation_csv_sha256'),
                              ('diagnosed_membership.csv', 'diagnosed_membership_csv_sha256')]:
            checked_bytes(artifact.evidence_path.parent / filename, evidence[key])
        report = {'purpose': 'scoped_simulation_candidate_and_pre_release_metrics', 'scenario': row, 'n1_binding': n1_binding,
            'created_at_utc': utc_now(), **git_context(), 'group_admissions_sha256': panel_hash,
            'freeze_sha256': freeze_hash, 'admission_sha256': admission_hash,
            'corruption_dependencies': corruption_links,
            'candidate_metadata_sha256': sha((directory / 'metadata.json').read_bytes()),
            'release_preview_metadata_sha256': preview_hash, 'pre_release_evidence_sha256': evidence_hash,
            'clean_preview_serialization_verified_exactly': True,
            'model_io_mode': 'verified_bytearray_and_byte_identical_short_temporary_mirrors',
            'original_candidate_directory': str(directory), 'original_champion_directory': str(champion_dir),
            'evaluation_representation': 'source_canonical_values_with_exact_replayed_perturbed_cells',
            'original_preview_files_modified': False,
            'pre_release_directory': str(artifact.evidence_path.parent),
            'candidate_training_executed': True, 'candidate_reload_verified': True,
            'production_package_approved': False, 'future_evaluation_executed': False,
            'complete_decision_evidence_frozen': False,
            'next_requirement': 'Persist simulated C1/C2/C3 envelope and all dependency links before future evaluation.'}
        report_hash = save_json(directory / 'validation.json', report)
        print('CANDIDATE RELOAD / PRE-RELEASE CHECKSUMS: PASS')
        print('PRE-RELEASE METRICS:', json.dumps(evidence['report'], indent=2))
        print('REPORT:', directory / 'validation.json')
        print('REPORT SHA256:', report_hash)
        print('PRE-RELEASE EVIDENCE:', artifact.evidence_path)
        print('COMPLETE DECISION-EVIDENCE FREEZE: PENDING; FUTURE EVALUATION: False')
    except Exception as error:
        save_json(directory / 'failure.json', {'scenario_id': SELECTED, 'exception_type': type(error).__name__,
            'message': str(error), 'future_evaluation_executed': False})
        raise


if __name__ == '__main__':
    main()
