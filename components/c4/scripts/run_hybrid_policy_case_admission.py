"""Plan or execute ONE explicitly bound fresh policy scenario through admission.

Uses existing preview/preparation/generation/diagnostic implementations.
No candidate or sealed-future evaluator is invoked by this runner.
"""
import io
import json
import subprocess
import sys
import tempfile
from importlib.metadata import version

import numpy as np
import pandas as pd

from _common import parser, resolve_data_root, manifest_root, checked_json, checked_bytes, require, save_json, sha, git_context, utc_now, C4_ROOT, REPO_ROOT
from _scoped_benchmark import frozen_tail_screen
from _training_group import corrupt_labels
from _hybrid_policy_execution import policy_group_cases as training_group_cases, verify_execution_binding
from c4.data import FEATURE_COLUMNS
from c4.models import load_champion
from c4.contracts.models import C2Diagnosis
from c4.scenarios.config import ScenarioConfig
from c4.scenarios.slices import SliceRule
from c4.scenarios.preview import save_role_preview
from c4.experiments.repair_preparation import prepare_repair_data
from c4.experiments.tvae_generation import TVAEGenerationConfig, generate_tvae_repair
from c4.data.split_manifest import load_adult_split_manifest


def main():
    p = parser(__doc__)
    p.add_argument('--scenario-id', required=True, help='One planned case in the explicitly selected training group.')
    p.add_argument('--group-id', required=True, help='Fresh policy-calibration group ID from frozen manifest.')
    p.add_argument('--execute-selected', action='store_true', help='Explicitly run the one selected training scenario only.')
    args = p.parse_args()
    root = resolve_data_root(args.data_root)
    verify_execution_binding(root, args.group_id)
    frozen_directory = root / 'c4_artifacts/benchmark_protocols/adult/adult-hybrid-fresh-validation-v1'
    frozen, freeze_hash = checked_json(frozen_directory, 'freeze')
    protocol = json.loads(checked_bytes(frozen_directory / 'protocol.json', frozen['protocol_sha256']))
    manifest, manifest_hash = checked_json(frozen_directory, 'scenario_manifest')
    require(manifest_hash == frozen['scenario_manifest_sha256'], 'Frozen scenario manifest differs.')
    require(frozen['status'] == 'frozen_specification_execution_disabled', 'Wrong frozen specification.')
    require(frozen['full_execution_enabled'] is False and protocol['full_execution_enabled'] is False,
            'This runner must remain scoped to one explicit training scenario.')
    matches = [item for item in training_group_cases(manifest, args.group_id) if item['scenario_id'] == args.scenario_id]
    require(len(matches) == 1, 'Requested scenario is outside the selected training group.')
    row = matches[0]
    print('SELECTED SCENARIO:', row['scenario_id'], flush=True)
    print('GROUP:', row['group_id'], 'FOLD:', row['split_role'], flush=True)
    print('CONDITION:', row['mechanism'], row['magnitude'], row['diagnosis_condition'], row['repair_condition'], flush=True)
    print('GENERATOR SEED:', row['generator_seed'], flush=True)
    print('FULL BENCHMARK EXECUTION: DISABLED', flush=True)
    if not args.execute_selected:
        print('PLAN ONLY: no artifacts generated. Use --execute-selected for this one training scenario.')
        return
    require(version('ctgan') == protocol['repair_generation']['ctgan_version'],
            'Installed ctgan differs from frozen protocol.')
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
    # Existing group previews pin the actual shared injector and feature metadata.
    from c4.scenarios.adapter import _load_shared_functions
    from c4.scenarios.feature_metadata import load_adult_feature_metadata
    shared_hash = _load_shared_functions()[1]
    feature_hash = load_adult_feature_metadata().source_sha256
    group_root = root / 'c4_artifacts/benchmark_runs/adult/adult-hybrid-fresh-validation-v1'
    for previous in training_group_cases(manifest, args.group_id):
        previous_path = group_root / previous['scenario_id'] / 'admission.json'
        if previous_path.exists():
            prior, _ = checked_json(previous_path.parent, 'admission')
            preview, _ = checked_json(root / 'c4_artifacts/scenario_previews/adult' / prior['preview_id'] / 'repair')
            require(preview['transformation']['shared_drift_types_sha256'] in (None, shared_hash), 'Shared drift code changed.')
            require(preview['transformation']['feature_metadata_sha256'] == feature_hash, 'Feature metadata changed.')
    md = manifest_root(root, protocol['source_versions']['split_manifest'])
    checked_bytes(md / 'manifest.json', frozen['split_manifest_sha256'])
    source_manifest = load_adult_split_manifest(md, data_root=root)
    champion_directory = root / 'c4_artifacts/champion/adult' / protocol['source_versions']['champion']
    champion, champion_metadata = load_champion(champion_directory)
    require(champion_metadata['model_sha256'] == frozen['champion_model_sha256'], 'Champion differs.')
    reference_bytes = checked_bytes(root / 'uci_adult_data/reference.csv', champion_metadata['reference_csv_sha256'])
    reference = pd.read_csv(io.BytesIO(reference_bytes))
    require(champion.get_booster().feature_names == list(FEATURE_COLUMNS), 'Champion schema differs.')
    sample = reference.loc[:, list(FEATURE_COLUMNS)].head(10)
    before = champion.predict(sample)
    with tempfile.TemporaryDirectory(prefix='c4-artifact-rollback-') as temporary:
        from pathlib import Path
        copy_dir = Path(temporary)
        for filename in ('champion.ubj', 'metadata.json'):
            (copy_dir / filename).write_bytes((champion_directory / filename).read_bytes())
        restored, _ = load_champion(copy_dir)
        require(np.array_equal(before, restored.predict(sample)), 'Champion restore predictions differ.')
        require(np.allclose(champion.predict_proba(sample), restored.predict_proba(sample)), 'Restore probabilities differ.')
    # The temporary-copy test is artifact rollback only, not live deployment rollback.
    identifier = 'bmk-' + sha(row['scenario_id'].encode())[:16]
    preview_id, preparation_version = identifier, 'preparation-v1'
    artifact_version = identifier + '-repair-v1'
    output = root / 'c4_artifacts/benchmark_runs/adult/adult-hybrid-fresh-validation-v1' / row['scenario_id']
    if output.exists():
        raise FileExistsError('Selected scenario has already been claimed; preserve its outputs.')
    for path in (root / 'c4_artifacts/scenario_previews/adult' / preview_id,
                 root / 'c4_artifacts/repair_preparation/adult' / preview_id,
                 root / 'c4_artifacts/offline_repair/adult' / artifact_version):
        require(not path.exists(), f'Scenario artifact already exists: {path}')
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / 'claim.json', {'purpose': 'one_selected_fresh_policy_admission_claim', 'created_at_utc': utc_now(),
              **git_context(), 'freeze_sha256': freeze_hash, 'scenario_manifest_sha256': manifest_hash,
              'scenario': row, 'full_execution_enabled': False, 'candidate_training_executed': False,
              'future_evaluation_executed': False})
    try:
        true = protocol['true_slice']
        true_rule = SliceRule(name=true['name'], feature=true['feature'], operator=true['operator'], value=true['resolved_threshold'])
        frames = {}
        for role in ('repair', 'release_validation'):
            config = ScenarioConfig(mechanism=row['mechanism'], magnitude=row['magnitude'],
                                    features=tuple(row['features']), interaction_pairs=tuple(tuple(pair) for pair in row['interaction_pairs']),
                                    seed=row['drift_seeds'][role])
            save_role_preview(manifest_directory=md, config=config, true_slice=true_rule,
                              role=role, preview_id=preview_id, data_root=root)
            directory = root / 'c4_artifacts/scenario_previews/adult' / preview_id / role
            metadata, _ = checked_json(directory)
            frames[role] = pd.read_csv(io.BytesIO(checked_bytes(directory / 'clean.csv', metadata['files']['clean.csv'])), index_col=0)
        diagnosed = row['diagnosed_rule']
        rule = SliceRule(name=diagnosed['name'], feature=diagnosed['feature'], operator=diagnosed['operator'], value=diagnosed['resolved_threshold'])
        baseline = rule.mask(frames['repair']).to_numpy()
        rng = np.random.default_rng(row['c2_bootstrap_seed'])
        ages = reference[rule.feature].to_numpy()
        repair_values = frames['repair'][rule.feature].to_numpy()
        overlaps = []
        for _ in range(protocol['diagnosis_evidence']['bootstrap_replicates']):
            threshold = float(np.quantile(rng.choice(ages, size=len(ages), replace=True), diagnosed['reference_quantile']))
            mask = repair_values <= threshold
            union = np.logical_or(baseline, mask).sum()
            overlaps.append(float(np.logical_and(baseline, mask).sum() / union) if union else 1.)
        stability = float(np.mean(overlaps))
        ranked_features = list(dict.fromkeys([*row['features'],
                                               *(feature for pair in row['interaction_pairs'] for feature in pair)]))
        if not ranked_features:
            ranked_features = [rule.feature]  # Explicit no-drift simulation convention.
        category = {'marginal_multi': 'marginal', 'interaction': 'interaction',
                    'combined': 'combined', 'none': 'none'}[row['mechanism']]
        diagnosis = C2Diagnosis(root_cause_category=category, root_cause_feature=ranked_features[0],
                               affected_slice_rule=f'{rule.feature} <= {rule.value!r}',
                               slice_support=float(rule.mask(frames['release_validation']).mean()),
                               diagnosis_stability=stability, affected_features_ranked=ranked_features,
                               diagnosis_version=identifier+'-sim-c2')
        prepared = prepare_repair_data(
            preview_directory=root / 'c4_artifacts/scenario_previews/adult' / preview_id / 'repair',
            manifest_directory=md, diagnosis=diagnosis, rule_feature_space='processed', data_root=root,
            validation_fraction=protocol['repair_generation']['internal_validation_fraction'],
            split_seed=protocol['repair_generation']['internal_split_seed'])
        files = {
            'generator_fit.csv': prepared.generator_fit.to_csv(index=True, index_label='source_row_id', lineterminator='\n').encode(),
            'internal_validation.csv': prepared.internal_validation.to_csv(index=True, index_label='source_row_id', lineterminator='\n').encode(),
            'validation_diagnosed_membership.csv': prepared.validation_diagnosed_membership.rename('diagnosed_slice_member').rename_axis('source_row_id').to_csv(lineterminator='\n').encode(),
        }
        prep_dir = root / 'c4_artifacts/repair_preparation/adult' / preview_id / preparation_version
        prep_dir.mkdir(parents=True, exist_ok=False)
        for filename, contents in files.items():
            (prep_dir / filename).write_bytes(contents)
        save_json(prep_dir / 'metadata.json', {
            'purpose': 'development_repair_preparation', 'created_at_utc': utc_now(), **git_context(),
            'benchmark_context': {'protocol_sha256': frozen['protocol_sha256'], 'scenario_id': row['scenario_id'], 'group_id': row['group_id']},
            'generator_fit_rows': len(prepared.generator_fit), 'internal_validation_rows': len(prepared.internal_validation),
            'diagnosed_internal_validation_rows': int(prepared.validation_diagnosed_membership.sum()),
            'generator_fitted': False, 'repair_package_approved': False,
            'provenance': prepared.provenance, 'files': {filename: sha(contents) for filename, contents in files.items()},
        })
        settings = protocol['repair_generation']
        if row['repair_condition'] == 'targeted_repair':
            require(row['label_flip_fraction'] == 0., 'Targeted condition must not corrupt labels.')
            generate_tvae_repair(preparation_directory=prep_dir, manifest_directory=md, artifact_version=artifact_version,
                                data_root=root, config=TVAEGenerationConfig(epochs=settings['epochs'], batch_size=settings['batch_size'],
                                                                         n_rows=settings['n_rows'], seed=row['generator_seed']))
        else:
            require(row['repair_condition'] == 'label_corrupted_repair', 'Unknown repair condition.')
            base_rows = [r for r in manifest['scenarios'] if r['shared_base_package_id'] == row['shared_base_package_id']
                         and r['repair_condition'] == 'targeted_repair']
            require(len(base_rows) == 1, 'A unique targeted base is required.')
            base_row = base_rows[0]
            base_run = output.parent / base_row['scenario_id']
            base_admission, base_admission_hash = checked_json(base_run, 'admission')
            require(base_admission['scenario'] == base_row and base_admission['freeze_sha256'] == freeze_hash,
                    'Paired base admission context differs.')
            base_dir = root / 'c4_artifacts/offline_repair/adult' / base_admission['artifact_version']
            base_metadata, base_metadata_hash = checked_json(base_dir)
            base_csv = checked_bytes(base_dir / 'synthetic_unvalidated.csv', base_metadata['synthetic_csv_sha256'])
            base_prep = root / 'c4_artifacts/repair_preparation/adult' / base_admission['preview_id'] / base_admission['preparation_version']
            base_preparation, base_prep_hash = checked_json(base_prep)
            require(base_prep_hash == base_metadata['preparation_metadata_sha256'], 'Base preparation link differs.')
            current_preparation, current_prep_hash = checked_json(prep_dir)
            for filename in files:
                require(base_preparation['files'][filename] == current_preparation['files'][filename],
                        'Paired conditions must have identical fitting and internal-validation data.')
            require(base_metadata['generation_config']['seed'] == row['generator_seed'], 'Base generator seed differs.')
            base = pd.read_csv(io.BytesIO(base_csv), float_precision='round_trip')
            corrupted, corruption_mask = corrupt_labels(base, fraction=row['label_flip_fraction'], seed=row['corruption_seed'])
            package_dir = root / 'c4_artifacts/offline_repair/adult' / artifact_version
            package_dir.mkdir(parents=True, exist_ok=False)
            generator_bytes = checked_bytes(base_dir / 'generator.pkl', base_metadata['generator_sha256'])
            (package_dir / 'generator.pkl').write_bytes(generator_bytes)
            corrupted_bytes = corrupted.to_csv(index=False, lineterminator='\n').encode()
            mask_bytes = corruption_mask.to_csv(index=False, lineterminator='\n').encode()
            (package_dir / 'synthetic_unvalidated.csv').write_bytes(corrupted_bytes)
            (package_dir / 'corruption_mask.csv').write_bytes(mask_bytes)
            import copy
            variant_metadata = copy.deepcopy(base_metadata)
            variant_metadata.update(artifact_version=artifact_version, created_at_utc=utc_now(), **git_context(),
                preparation_metadata_sha256=current_prep_hash, synthetic_csv_sha256=sha(corrupted_bytes),
                quality_metrics=None, privacy_status='NOT_EVALUATED', repair_package_approved=False,
                corruption_variant={'procedure': 'flip_100_distinct_binary_targets_no_feature_changes',
                    'fraction': .2, 'seed': row['corruption_seed'], 'mask_sha256': sha(mask_bytes),
                    'base_artifact_version': base_admission['artifact_version'],
                    'base_metadata_sha256': base_metadata_hash, 'base_csv_sha256': sha(base_csv),
                    'base_admission_sha256': base_admission_hash, 'generator_refitted': False,
                    'quality_scores_reused': False})
            save_json(package_dir / 'metadata.json', variant_metadata)
            print('PAIRED REPAIR CORRUPTION: 100 / 500; GENERATOR REFITTED: False', flush=True)
        shared_args = ['--data-root', str(root), '--preview-id', preview_id,
                       '--preparation-version', preparation_version, '--manifest-version', protocol['source_versions']['split_manifest'],
                       '--artifact-version', artifact_version]
        for script in ('audit_development_repair.py', 'check_repair_distribution.py', 'check_repair_privacy.py'):
            command = [sys.executable, str(C4_ROOT / 'scripts' / script), *shared_args]
            if script == 'check_repair_distribution.py':
                command += ['--slice-feature', rule.feature, '--slice-operator', rule.operator, '--slice-threshold', repr(rule.value)]
            subprocess.run(command, check=True)
        package = root / 'c4_artifacts/offline_repair/adult' / artifact_version
        generation, generation_hash = checked_json(package)
        privacy, privacy_hash = checked_json(package, 'privacy_diagnostics')
        require(privacy['generation_metadata_sha256'] == generation_hash, 'Privacy generation link differs.')
        summaries = privacy['summaries']
        screen = frozen_tail_screen(protocol, summaries['synthetic_to_generator_fit'],
                                     summaries['diagnosed_internal_validation_to_generator_fit'], protocol_kind='benchmark')
        accepted = screen['screen_status'] == 'PASS' and stability >= .50
        result = {
            **screen, 'purpose': 'fresh_policy_scenario_simulation_admission', 'created_at_utc': utc_now(), **git_context(),
            'scenario': row, 'freeze_sha256': freeze_hash, 'generation_metadata_sha256': generation_hash,
            'privacy_report_sha256': privacy_hash, 'simulated_c2_diagnosis': diagnosis.model_dump(mode='json'),
            'diagnosis_stability_origin': 'defined_reference_bootstrap_mask_agreement_not_real_C2_output',
            'artifact_schema_checked': True, 'offline_champion_restore_checked': True,
            'simulation_admission': 'READY_FOR_SCOPED_CANDIDATE_VALIDATION' if accepted else 'BLOCKED_ESCALATE',
            'candidate_training_executed': False, 'future_evaluation_executed': False,
            'production_package_approved': False, 'full_benchmark_execution': False,
            'artifact_version': artifact_version, 'preview_id': preview_id, 'preparation_version': preparation_version,
            'scope_note': 'One scoped simulation scenario, not actual C1/C2/C3 producer performance or final N1-N4 validation.',
        }
        checksum = save_json(output / 'admission.json', result)
        print('SIMULATION ADMISSION:', result['simulation_admission'])
        print('SCREEN:', screen['screen_status'], 'RATIO:', screen['tail_distance_ratio'])
        print('MEASURED BOOTSTRAP STABILITY:', stability)
        print('ARTIFACT VERSION:', artifact_version)
        print('REPORT:', output / 'admission.json')
        print('REPORT SHA256:', checksum)
        print('CANDIDATE TRAINING / FUTURE EVALUATION EXECUTED: False')
    except Exception as error:
        save_json(output / 'failure.json', {'purpose': 'selected_scenario_execution_failure', 'created_at_utc': utc_now(),
                  'scenario_id': row['scenario_id'], 'exception_type': type(error).__name__, 'message': str(error),
                  'candidate_training_executed': False, 'future_evaluation_executed': False})
        raise


if __name__ == '__main__':
    main()
