"""Recover the one observed pre-TVAE-import failure using verified original preparation."""
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
from _conformal_group import conformal_group_cases, verify_n1
from _conformal_recovery import SCENARIO, GROUP, validate_interrupted_case, retain_files, verify_retained
from c4.data import FEATURE_COLUMNS
from c4.models import load_champion
from c4.contracts.models import C2Diagnosis
from c4.scenarios.config import ScenarioConfig
from c4.scenarios.slices import SliceRule
from c4.experiments.repair_preparation import prepare_repair_data
from c4.experiments.tvae_generation import TVAEGenerationConfig, generate_tvae_repair
from c4.data.split_manifest import load_adult_split_manifest


def main():
    p = parser(__doc__)
    p.add_argument('--finish', action='store_true', help='Fit the original, not-yet-fitted case once from verified saved preparation.')
    args = p.parse_args()
    args.scenario_id, args.group_id = SCENARIO, GROUP
    root = resolve_data_root(args.data_root)
    n1_binding = verify_n1(root)
    frozen_directory = root / 'c4_artifacts/benchmark_protocols/adult/adult-simulation-pp1-v2'
    frozen, freeze_hash = checked_json(frozen_directory, 'freeze')
    protocol = json.loads(checked_bytes(frozen_directory / 'protocol.json', frozen['protocol_sha256']))
    manifest, manifest_hash = checked_json(frozen_directory, 'scenario_manifest')
    require(manifest_hash == frozen['scenario_manifest_sha256'], 'Frozen scenario manifest differs.')
    require(frozen['status'] == 'frozen_specification_execution_disabled', 'Wrong frozen specification.')
    require(frozen['full_execution_enabled'] is False and protocol['full_execution_enabled'] is False,
            'This runner must remain scoped to one explicit conformal scenario.')
    matches = [item for item in conformal_group_cases(manifest, args.group_id) if item['scenario_id'] == args.scenario_id]
    require(len(matches) == 1, 'Requested scenario is outside the selected conformal group.')
    require(freeze_hash == n1_binding['protocol_freeze_sha256'], 'N1 and conformal protocol differ.')
    row = matches[0]
    print('SELECTED SCENARIO:', row['scenario_id'], flush=True)
    print('GROUP:', row['group_id'], 'FOLD:', row['split_role'], flush=True)
    print('CONDITION:', row['mechanism'], row['magnitude'], row['diagnosis_condition'], row['repair_condition'], flush=True)
    print('GENERATOR SEED:', row['generator_seed'], flush=True)
    print('FULL BENCHMARK EXECUTION: DISABLED', flush=True)
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
    group_root = root / 'c4_artifacts/benchmark_runs/adult/adult-simulation-pp1-v2'
    for previous in conformal_group_cases(manifest, args.group_id):
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
    output = root / 'c4_artifacts/benchmark_runs/adult/adult-simulation-pp1-v2' / row['scenario_id']
    from pathlib import Path
    claim, claim_hash = checked_json(output, 'claim')
    failure, failure_hash = checked_json(output, 'failure')
    validate_interrupted_case(claim, failure, row, n1_binding)
    require(claim['freeze_sha256'] == freeze_hash and claim['scenario_manifest_sha256'] == manifest_hash,
            'Original interrupted claim links to a different protocol.')
    require(not (output / 'admission.json').exists(), 'This case already has an admission result; never regenerate it.')
    require(not (output / 'future_research-v1').exists(), 'Unexpected future execution in admission recovery.')
    package_path = root / 'c4_artifacts/offline_repair/adult' / artifact_version
    require(not package_path.exists(), 'A generated/partial package exists; inspect it rather than fitting again.')
    original_paths = [output / 'claim.json', output / 'claim.sha256', output / 'failure.json', output / 'failure.sha256']
    for directory in (root / 'c4_artifacts/scenario_previews/adult' / preview_id,
                      root / 'c4_artifacts/repair_preparation/adult' / preview_id):
        require(directory.is_dir(), 'Original preview/preparation directory is missing.')
        original_paths.extend(p for p in directory.rglob('*') if p.is_file())
    retained = retain_files(original_paths)
    recovery = output / 'import-recovery-v1'
    require(not recovery.exists(), 'Recovery attempt already exists; inspect it without overwriting.')
    # Verify the original failed runner against its original group claim.
    group_directory = output.parent / 'group_admissions' / GROUP / 'v1'
    group_claim, _ = checked_json(group_directory, 'claim')
    for filename, expected in group_claim['source_sha256'].items():
        checked_bytes(C4_ROOT / 'scripts' / filename, expected)
    require(version('torch') == '2.14.1' and version('ctgan') == '0.12.1',
            'Environment versions changed from the reported recovery checkpoint.')
    try:
        true = protocol['true_slice']
        true_rule = SliceRule(name=true['name'], feature=true['feature'], operator=true['operator'], value=true['resolved_threshold'])
        frames = {}
        for role in ('repair', 'release_validation'):
            config = ScenarioConfig(mechanism=row['mechanism'], magnitude=row['magnitude'],
                                    features=tuple(row['features']), interaction_pairs=tuple(tuple(pair) for pair in row['interaction_pairs']),
                                    seed=row['drift_seeds'][role])
            directory = root / 'c4_artifacts/scenario_previews/adult' / preview_id / role
            metadata, _ = checked_json(directory)
            require(metadata['role'] == role and metadata['split_manifest_sha256'] == frozen['split_manifest_sha256'] and
                    metadata['source_row_ids'] == source_manifest['partitions'][role]['source_row_ids'] and
                    metadata['transformation']['configuration'] == config.model_dump(mode='json') and
                    metadata['transformation']['true_slice'] == true_rule.model_dump(mode='json'),
                    'Saved role preview differs from the frozen case.')
            for filename, expected in metadata['files'].items():
                checked_bytes(directory / filename, expected)
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
        prep, prep_hash = checked_json(prep_dir)
        require(prep['purpose'] == 'development_repair_preparation' and
                prep['benchmark_context'] == {'protocol_sha256': frozen['protocol_sha256'],
                    'scenario_id': row['scenario_id'], 'group_id': row['group_id']},
                'Original preparation context differs.')
        require(prep['provenance'] == json.loads(json.dumps(prepared.provenance)),
                'Reconstructed repair-role provenance differs.')
        for filename, contents in files.items():
            require(checked_bytes(prep_dir / filename, prep['files'][filename]) == contents,
                    'Original preparation differs from deterministic reconstruction: ' + filename)
        require(prep['generator_fitted'] is False and prep['repair_package_approved'] is False,
                'Unexpected preparation status.')
        # Import before creating a recovery claim; this operation never fits a model.
        import torch
        from ctgan import TVAE
        print('TORCH / TVAE IMPORT: PASS', flush=True)
        print('ORIGINAL PREPARATION / SOURCE LINKS / RECONSTRUCTION: PASS', flush=True)
        print('GENERATOR SEED:', row['generator_seed'], 'ORIGINAL INPUTS PRESERVED: True', flush=True)
        print('FIT ROWS:', len(prepared.generator_fit), 'INTERNAL VALIDATION ROWS:', len(prepared.internal_validation), flush=True)
        if not args.finish:
            print('PLAN ONLY. Use --finish to fit this original case once and complete admission; no candidate/future evaluation.')
            return
        verify_retained(retained)
        recovery.mkdir(parents=True, exist_ok=False)
        recovery_sources = {str(Path(__file__)): sha(Path(__file__).read_bytes()),
            str(C4_ROOT / 'scripts/_conformal_recovery.py'): sha((C4_ROOT / 'scripts/_conformal_recovery.py').read_bytes())}
        save_json(recovery / 'claim.json', {'purpose': 'pre_fit_import_failure_recovery', 'created_at_utc': utc_now(),
            **git_context(), 'scenario': row, 'original_claim_sha256': claim_hash, 'original_failure_sha256': failure_hash,
            'preparation_metadata_sha256': prep_hash, 'preserved_original_files': retained,
            'recovery_sources_sha256': recovery_sources, 'n1_binding': n1_binding,
            'torch_version': torch.__version__, 'ctgan_version': version('ctgan'),
            'generator_seed_changed': False, 'candidate_training_executed': False, 'future_evaluation_executed': False})
        settings = protocol['repair_generation']
        generate_tvae_repair(preparation_directory=prep_dir, manifest_directory=md, artifact_version=artifact_version,
                            data_root=root, config=TVAEGenerationConfig(epochs=settings['epochs'], batch_size=settings['batch_size'],
                                                                     n_rows=settings['n_rows'], seed=row['generator_seed']))
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
            **screen, 'purpose': 'conformal_group_scenario_simulation_admission', 'created_at_utc': utc_now(), **git_context(),
            'scenario': row, 'n1_binding': n1_binding, 'freeze_sha256': freeze_hash, 'generation_metadata_sha256': generation_hash,
            'privacy_report_sha256': privacy_hash, 'simulated_c2_diagnosis': diagnosis.model_dump(mode='json'),
            'diagnosis_stability_origin': 'defined_reference_bootstrap_mask_agreement_not_real_C2_output',
            'artifact_schema_checked': True, 'offline_champion_restore_checked': True,
            'simulation_admission': 'READY_FOR_SCOPED_CANDIDATE_VALIDATION' if accepted else 'BLOCKED_ESCALATE',
            'candidate_training_executed': False, 'future_evaluation_executed': False,
            'production_package_approved': False, 'full_benchmark_execution': False,
            'artifact_version': artifact_version, 'preview_id': preview_id, 'preparation_version': preparation_version,
            'scope_note': 'One scoped simulation scenario, not actual C1/C2/C3 producer performance or final N1-N4 validation.',
        }
        result['recovery_context'] = {'original_claim_sha256': claim_hash, 'original_failure_sha256': failure_hash,
            'preparation_metadata_sha256': prep_hash, 'recovery_claim_path': str(recovery / 'claim.json'),
            'generator_seed_changed': False, 'reason': 'original TVAE import failed before fitting'}
        verify_retained(retained)
        require(verify_n1(root) == n1_binding, 'Frozen N1 changed during recovery.')
        for path, expected in recovery_sources.items():
            checked_bytes(path, expected)
        checksum = save_json(output / 'admission.json', result)
        save_json(recovery / 'completion.json', {'created_at_utc': utc_now(), 'admission_sha256': checksum,
            'original_evidence_unchanged': True, 'candidate_training_executed': False, 'future_evaluation_executed': False})
        print('SIMULATION ADMISSION:', result['simulation_admission'])
        print('SCREEN:', screen['screen_status'], 'RATIO:', screen['tail_distance_ratio'])
        print('MEASURED BOOTSTRAP STABILITY:', stability)
        print('ARTIFACT VERSION:', artifact_version)
        print('REPORT:', output / 'admission.json')
        print('REPORT SHA256:', checksum)
        print('CANDIDATE TRAINING / FUTURE EVALUATION EXECUTED: False')
    except Exception as error:
        if not recovery.exists():
            raise
        save_json(recovery / 'failure.json', {'purpose': 'selected_scenario_recovery_failure', 'created_at_utc': utc_now(),
                  'scenario_id': row['scenario_id'], 'exception_type': type(error).__name__, 'message': str(error),
                  'candidate_training_executed': False, 'future_evaluation_executed': False})
        raise


if __name__ == '__main__':
    main()
