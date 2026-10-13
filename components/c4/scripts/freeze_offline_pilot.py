"""Freeze one new generator trial, not the complete PP1 benchmark."""
import json
from importlib.metadata import version

from _common import (
    parser, resolve_data_root, manifest_root, checked_json, checked_bytes,
    require, save_json, sha, git_context, utc_now, C4_ROOT,
)
from prepare_offline_repair import verify_preparation
from c4.models import load_champion
from c4.data.split_manifest import load_adult_split_manifest


def main():
    args = parser(__doc__).parse_args()
    root = resolve_data_root(args.data_root)
    source = (C4_ROOT / 'configs' / 'offline_pilot_protocol.json').read_bytes()
    config = json.loads(source)
    require(config['schema_version'] == '1.0'
            and config['protocol_version'] == 'adult-prospective-pilot-v1', 'Unsupported pilot version.')
    require(config['full_benchmark_execution_enabled'] is False
            and config['production_c3_approval_export_enabled'] is False, 'Wrong scope.')
    a = config['generator_attempt']
    expected = {'artifact_version': 'pilot-independent-tvae-v1', 'ctgan_version': '0.12.1',
                'epochs': 100, 'batch_size': 256, 'n_rows': 500, 'seed': 123,
                'sampling': 'single_unstratified_sample', 'maximum_attempts': 1,
                'adaptive_retries_allowed': False, 'new_fit_required': True,
                'post_sampling_stratification': False}
    for key, value in expected.items():
        require(a[key] == value, f'Unexpected v1 generator setting: {key}')
    require(version('ctgan') == a['ctgan_version'], 'Installed ctgan differs from protocol.')
    f = config['candidate_and_future']
    require(f['freeze_complete_pre_release_evidence_before_future'] is True, 'Future freeze guard is required.')
    for key in ('use_release_validation_for_training_or_generator_selection',
                'use_future_for_training_calibration_policy_or_selection',
                'open_future_for_failed_or_indeterminate_trial', 'automatic_deployment_allowed'):
        require(f[key] is False, f'Forbidden pilot permission: {key}')
    output = root / 'c4_artifacts' / 'protocols' / 'offline_pilot' / 'adult' / config['protocol_version']
    if output.exists():
        raise FileExistsError(f'Frozen protocol already exists: {output}')
    require(not (root / 'c4_artifacts' / 'offline_repair' / 'adult' / a['artifact_version']).exists(),
            'Trial already exists; it cannot be prospectively frozen.')
    md = manifest_root(root, config['split_manifest_version'])
    manifest = load_adult_split_manifest(md, data_root=root)
    preparation = (root / 'c4_artifacts' / 'repair_preparation' / 'adult'
                   / config['preview_id'] / config['preparation_version'])
    verify_preparation(preparation, root, md)
    _, prep_hash = checked_json(preparation)
    _, champion = load_champion(root / 'c4_artifacts' / 'champion' / 'adult' / config['champion_version'])
    checked_bytes(root / 'uci_adult_data' / 'reference.csv', champion['reference_csv_sha256'])
    require(champion['reference_csv_sha256'] == manifest['source_checksums']['reference_csv_sha256'],
            'Champion and manifest reference sources differ.')
    implementations = {'generation': C4_ROOT / 'src' / 'c4' / 'experiments' / 'tvae_generation.py',
                       'distance': C4_ROOT / 'scripts' / '_privacy_distance.py'}
    record = {
        'purpose': 'frozen_offline_pilot_protocol', 'status': 'frozen_for_pilot_only',
        'created_at_utc': utc_now(), **git_context(),
        'protocol_sha256': sha(source), 'split_manifest_sha256': sha((md / 'manifest.json').read_bytes()),
        'preparation_metadata_sha256': prep_hash, 'champion_model_sha256': champion['model_sha256'],
        'implementation_sha256': {key: sha(path.read_bytes()) for key, path in implementations.items()},
        'full_benchmark_execution_enabled': False, 'production_c3_approval_export_enabled': False,
    }
    output.mkdir(parents=True, exist_ok=False)
    with (output / 'protocol.json').open('xb') as file:
        file.write(source)
    checksum = save_json(output / 'freeze.json', record)
    print('FREEZE STATUS: frozen_for_pilot_only')
    print('PROTOCOL:', output / 'protocol.json')
    print('PROTOCOL SHA256:', sha(source))
    print('FREEZE SHA256:', checksum)
    print('NEW FIT SEED: 123; MAXIMUM ATTEMPTS: 1')
    print('FULL BENCHMARK ENABLED: False; PRODUCTION APPROVAL EXPORT: False')


if __name__ == '__main__':
    main()
