"""Save a new UNVALIDATED target-stratified sampling variant of a saved TVAE."""
import copy
import io
import json
import platform
from importlib.metadata import version

import joblib
import numpy as np
import pandas as pd

from _common import (
    parser, name, resolve_data_root, manifest_root, checked_json, checked_bytes,
    require, save_json, sha, utc_now, git_context,
)
from generate_development_repair import verify_generation
from check_repair_distribution import validate_frame
from _target_stratified_sampling import binary_quotas, select_pool_rows


def main():
    p = parser(__doc__)
    p.add_argument('--preview-id', type=name, default='development-001')
    p.add_argument('--preparation-version', type=name, default='exact_slice-v1')
    p.add_argument('--manifest-version', type=name, default='v1')
    p.add_argument('--parent-version', type=name, default='development-tvae-exact-v1')
    p.add_argument('--artifact-version', type=name, default='development-tvae-target-strat-v1')
    args = p.parse_args()
    # Fixed exploratory treatment; changing these requires a new documented version.
    n_rows, pool_rows, batch_rows, sampling_seed, selection_seed = 500, 20000, 1000, 42, 43
    root = resolve_data_root(args.data_root)
    base = root / 'c4_artifacts' / 'offline_repair' / 'adult'
    parent = base / args.parent_version
    output = base / args.artifact_version
    require(parent != output, 'Variant must not overwrite its parent.')
    if output.exists():
        raise FileExistsError(f'Variant already exists: {output}')
    preparation = (root / 'c4_artifacts' / 'repair_preparation' / 'adult'
                   / args.preview_id / args.preparation_version)
    original, parent_hash = verify_generation(
        parent, preparation, root, manifest_root(root, args.manifest_version))
    require(original.get('sampling_variant') is None,
            'Use the original unmodified TVAE as parent, not another sampling variant.')
    for library in ('ctgan', 'torch'):
        require(version(library) == original['environment'][library],
                f'{library} version differs from parent generator environment.')
    prepared, preparation_hash = checked_json(preparation)
    fit = pd.read_csv(io.BytesIO(checked_bytes(
        preparation / 'generator_fit.csv', prepared['files']['generator_fit.csv'])), index_col=0)
    validate_frame(fit)
    quotas = binary_quotas(fit['target'], n_rows)
    parent_bytes = checked_bytes(parent / 'generator.pkl', original['generator_sha256'])
    generator = joblib.load(io.BytesIO(parent_bytes))
    generator.set_random_state(sampling_seed)
    protocol = {
        'sampling_variant': 'fixed_pool_target_stratification',
        'fit_label_counts': {str(k): int(v) for k, v in fit['target'].value_counts().sort_index().items()},
        'quotas': {str(k): v for k, v in quotas.items()},
        'quota_source': 'generator_fit_labels_only',
        'quota_rounding': 'largest_remainder_class_0_tie_break',
        'fixed_pool_rows': pool_rows, 'sample_batch_rows': batch_rows,
        'sampling_seed': sampling_seed, 'selection_seed': selection_seed,
        'within_class_selection': 'first_rows_in_fixed_pool',
        'final_order': 'seeded_permutation_of_selected_positions',
        'insufficient_class_rows_action': 'fail_without_extending_budget',
        'generator_refitted': False,
        'design_context': 'Exploratory response to observed pilot class imbalance, not a confirmatory benchmark.',
        'privacy_screen_changed': False,
    }
    output.mkdir(parents=True, exist_ok=False)
    # Persist the treatment before producing its sample, and retain state before sampling.
    save_json(output / 'sampling_protocol.json', protocol)
    generator_path = output / 'generator.pkl'
    joblib.dump(generator, generator_path)
    print('CLASS QUOTAS:', quotas, flush=True)
    print('SAMPLING FIXED POOL:', pool_rows, 'rows; no TVAE refitting.', flush=True)
    batches = []
    for _ in range(pool_rows // batch_rows):
        batch = generator.sample(batch_rows)
        require(isinstance(batch, pd.DataFrame) and len(batch) == batch_rows,
                'TVAE returned an invalid sample size.')
        validate_frame(batch)
        for column in original['discrete_columns']:
            require(batch[column].isin(fit[column].unique()).all(),
                    f'Unknown discrete value in pool: {column}')
        batches.append(batch)
    pool = pd.concat(batches, ignore_index=True)
    require(len(pool) == pool_rows, 'Fixed pool size differs.')
    pool_bytes = pool.to_csv(index=False, lineterminator='\n').encode('utf-8')
    with (output / 'sampling_pool.csv').open('xb') as file:
        file.write(pool_bytes)
    try:
        synthetic, selected = select_pool_rows(pool, quotas, selection_seed)
    except ValueError as error:
        save_json(output / 'sampling_failure.json', {
            'purpose': 'development_sampling_failure', 'created_at_utc': utc_now(),
            'parent_metadata_sha256': parent_hash, 'sampling_pool_sha256': sha(pool_bytes),
            'reason': str(error), 'repair_package_approved': False,
        })
        raise
    validate_frame(synthetic)
    selection = pd.DataFrame({'synthetic_row_position': np.arange(n_rows),
                              'sampling_pool_position': selected,
                              'target': synthetic['target'].to_numpy(dtype=int)})
    synthetic_bytes = synthetic.to_csv(index=False, lineterminator='\n').encode('utf-8')
    selection_bytes = selection.to_csv(index=False, lineterminator='\n').encode('utf-8')
    for filename, contents in (('synthetic_unvalidated.csv', synthetic_bytes),
                               ('sampling_selection.csv', selection_bytes)):
        with (output / filename).open('xb') as file:
            file.write(contents)
    metadata = copy.deepcopy(original)
    metadata.update({
        'artifact_version': args.artifact_version, 'created_at_utc': utc_now(), **git_context(),
        'sampling_variant': 'fixed_pool_target_stratification',
        'parent_artifact_version': args.parent_version,
        'parent_metadata_sha256': parent_hash,
        'parent_generator_sha256': original['generator_sha256'],
        'generator_fit_origin': 'reused_parent_fitted_model_no_new_fit',
        'original_training_git_revision': original.get('git_revision'),
        'synthetic_rows': n_rows, 'generator_sha256': sha(generator_path.read_bytes()),
        'synthetic_csv_sha256': sha(synthetic_bytes),
        'sampling_pool_sha256': sha(pool_bytes), 'sampling_selection_sha256': sha(selection_bytes),
        'sampling_protocol_sha256': sha((output / 'sampling_protocol.json').read_bytes()),
        'preparation_metadata_sha256': preparation_hash,
        'quality_metrics': None, 'privacy_status': 'NOT_EVALUATED',
        'repair_package_approved': False, 'validation_status': 'UNVALIDATED',
        'environment': {'python': platform.python_version(), 'ctgan': version('ctgan'),
                        'torch': version('torch'), 'numpy': np.__version__,
                        'pandas': pd.__version__, 'joblib': joblib.__version__},
        'limitations': [
            'Postsampling label stratification is not conditional TVAE training.',
            'The fixed draw budget is larger than original unstratified sampling; record it in comparisons.',
            'The underlying generator weights are reused; this is not an independent fit or seed replication.',
            'This variant was motivated by pilot results and is exploratory.',
            'Class quotas do not guarantee utility, feature/label fidelity, slice coverage, or privacy.',
        ],
    })
    checksum = save_json(output / 'metadata.json', metadata)
    print('TARGET COUNTS:', synthetic['target'].value_counts().sort_index().to_dict())
    print('POOL TARGET COUNTS:', pool['target'].value_counts().sort_index().to_dict())
    print('DIRECTORY:', output)
    print('METADATA SHA256:', checksum)
    print('GENERATOR REFITTED: False')
    print('PRIVACY: NOT_EVALUATED; REPAIR APPROVED: False')


if __name__ == '__main__':
    main()
