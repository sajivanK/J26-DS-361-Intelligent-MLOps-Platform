"""Run the single new fit pinned by the saved prospective pilot protocol."""
import json
import time
from importlib.metadata import version

from _common import (
    parser, resolve_data_root, manifest_root, checked_json, checked_bytes,
    require, save_json, git_context, utc_now, C4_ROOT,
)
from prepare_offline_repair import verify_preparation
from c4.models import load_champion
from c4.experiments.tvae_generation import TVAEGenerationConfig, generate_tvae_repair


def main():
    args = parser(__doc__).parse_args()
    root = resolve_data_root(args.data_root)
    directory = root / 'c4_artifacts' / 'protocols' / 'offline_pilot' / 'adult' / 'adult-prospective-pilot-v1'
    frozen, freeze_hash = checked_json(directory, 'freeze')
    source = checked_bytes(directory / 'protocol.json', frozen['protocol_sha256'])
    config = json.loads(source)
    require(frozen['status'] == 'frozen_for_pilot_only', 'Pilot protocol is not frozen.')
    require(config['full_benchmark_execution_enabled'] is False
            and config['production_c3_approval_export_enabled'] is False, 'Wrong pilot scope.')
    a = config['generator_attempt']
    require(a['maximum_attempts'] == 1 and a['adaptive_retries_allowed'] is False, 'Wrong attempt budget.')
    require(version('ctgan') == a['ctgan_version'], 'ctgan differs from frozen protocol.')
    checked_bytes(C4_ROOT / 'src' / 'c4' / 'experiments' / 'tvae_generation.py',
                  frozen['implementation_sha256']['generation'])
    checked_bytes(C4_ROOT / 'scripts' / '_privacy_distance.py', frozen['implementation_sha256']['distance'])
    md = manifest_root(root, config['split_manifest_version'])
    checked_bytes(md / 'manifest.json', frozen['split_manifest_sha256'])
    preparation = (root / 'c4_artifacts' / 'repair_preparation' / 'adult'
                   / config['preview_id'] / config['preparation_version'])
    verify_preparation(preparation, root, md)
    _, prep_hash = checked_json(preparation)
    require(prep_hash == frozen['preparation_metadata_sha256'], 'Frozen preparation differs.')
    _, champion = load_champion(root / 'c4_artifacts' / 'champion' / 'adult' / config['champion_version'])
    require(champion['model_sha256'] == frozen['champion_model_sha256'], 'Frozen champion differs.')
    output = root / 'c4_artifacts' / 'offline_repair' / 'adult' / a['artifact_version']
    if output.exists() or (directory / 'attempt_claim.json').exists():
        raise FileExistsError('This frozen attempt already exists or has been claimed; do not overwrite/retry it.')
    save_json(directory / 'attempt_claim.json', {
        'purpose': 'prospective_pilot_attempt_claim', 'created_at_utc': utc_now(),
        'freeze_sha256': freeze_hash, 'artifact_version': a['artifact_version'], **git_context(),
    })
    generation_config = TVAEGenerationConfig(**{key: a[key] for key in ('epochs', 'batch_size', 'n_rows', 'seed')})
    print('FROZEN CONFIG:', generation_config, flush=True)
    print('NEW FIT: True; POST-SAMPLING STRATIFICATION: False', flush=True)
    started = time.perf_counter()
    try:
        artifact = generate_tvae_repair(
            preparation_directory=preparation, manifest_directory=md,
            artifact_version=a['artifact_version'], data_root=root, config=generation_config)
    except Exception as error:
        save_json(directory / 'attempt_failure.json', {
            'purpose': 'prospective_pilot_failure', 'created_at_utc': utc_now(),
            'freeze_sha256': freeze_hash, 'exception_type': type(error).__name__, 'message': str(error),
        })
        raise
    generation, generation_hash = checked_json(artifact.directory)
    record = {
        'purpose': 'prospective_offline_pilot_generation', 'created_at_utc': utc_now(), **git_context(),
        'freeze_sha256': freeze_hash, 'protocol_sha256': frozen['protocol_sha256'],
        'generation_metadata_sha256': generation_hash, 'elapsed_seconds': time.perf_counter() - started,
        'new_fitted_generator': True, 'same_preparation_data_as_earlier_pilots': True,
        'independent_dataset_replication': False, 'repair_package_approved': False,
        'benchmark_execution': False,
    }
    checksum = save_json(artifact.directory / 'pilot_context.json', record)
    print('DIRECTORY:', artifact.directory)
    print('SYNTHETIC ROWS:', generation['synthetic_rows'])
    print('GENERATION METADATA SHA256:', generation_hash)
    print('PILOT CONTEXT SHA256:', checksum)
    print('ELAPSED SECONDS:', round(record['elapsed_seconds'], 2))
    print('STATUS: UNVALIDATED; SCREENING STILL REQUIRED')


if __name__ == '__main__':
    main()
