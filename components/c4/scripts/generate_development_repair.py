"""Generate or verify an UNVALIDATED development TVAE artifact."""
from pathlib import Path
import io
import time
from importlib.metadata import version
import pandas as pd
from _common import (parser, resolve_data_root, default_yaml, name, manifest_root,
                     checked_json, checked_bytes, require, sha, save_json, git_context)
from prepare_offline_repair import verify_preparation
from c4.experiments.tvae_generation import TVAEGenerationConfig, generate_tvae_repair
from c4.scenarios.benchmark import load_draft_benchmark, verify_draft_sources


def verify_generation(directory, preparation, root, manifest_directory):
    metadata, metadata_hash = checked_json(directory)
    require(metadata.get('purpose') == 'development_synthetic_repair', 'Wrong generation purpose.')
    require(metadata['validation_status'] == 'UNVALIDATED' and metadata['repair_package_approved'] is False,
            'This verifier supports unvalidated development artifacts only.')
    verify_preparation(preparation, root, manifest_directory)
    prepared, preparation_hash = checked_json(preparation)
    require(metadata['preparation_metadata_sha256'] == preparation_hash, 'Preparation link differs.')
    for field in ('generator_fit_row_ids', 'internal_validation_row_ids'):
        require(metadata[field] == prepared['provenance'][field], f'Generation {field} differs.')
    checked_bytes(directory / 'generator.pkl', metadata['generator_sha256'])
    contents = checked_bytes(directory / 'synthetic_unvalidated.csv', metadata['synthetic_csv_sha256'])
    synthetic = pd.read_csv(io.BytesIO(contents))
    require(len(synthetic) == metadata['synthetic_rows'], 'Synthetic count differs.')
    require(synthetic['target'].isin([0, 1]).all(), 'Invalid synthetic target labels.')
    print('DIRECTORY:', directory)
    print('SYNTHETIC ROWS:', len(synthetic))
    print('TARGET COUNTS:', synthetic['target'].value_counts().sort_index().to_dict())
    print('GENERATION CHECKSUMS / LINKS: PASS')
    print('VALIDATION STATUS:', metadata['validation_status'])
    print('METADATA SHA256:', metadata_hash)
    return metadata, metadata_hash


def main():
    p = parser(__doc__)
    p.add_argument('--config', type=Path, default=default_yaml())
    p.add_argument('--preview-id', type=name, default='development-001')
    p.add_argument('--preparation-version', type=name, default='exact_slice-v1')
    p.add_argument('--manifest-version', type=name, default='v1')
    p.add_argument('--artifact-version', type=name, default='development-tvae-exact-v1')
    p.add_argument('--verify-only', action='store_true')
    args = p.parse_args()
    root = resolve_data_root(args.data_root)
    directory = root / 'c4_artifacts' / 'offline_repair' / 'adult' / args.artifact_version
    preparation = root / 'c4_artifacts' / 'repair_preparation' / 'adult' / args.preview_id / args.preparation_version
    manifest_directory = manifest_root(root, args.manifest_version)
    if args.verify_only:
        verify_generation(directory, preparation, root, manifest_directory)
        return
    if directory.exists():
        raise FileExistsError(f'Artifact already exists: {directory}; use --verify-only.')
    draft = load_draft_benchmark(args.config)
    verify_draft_sources(draft, data_root=root)
    require(draft.document['data_roles']['split_manifest_version'] == args.manifest_version,
            'Manifest version differs from draft.')
    settings = draft.document['repair_protocol']['generator_hyperparameters']
    require(version('ctgan') == settings['library_version'], 'ctgan version differs from draft.')
    architecture = {'library': 'ctgan', 'generator': 'TVAE', 'embedding_dim': 128,
                    'compress_dims': [128, 128], 'decompress_dims': [128, 128],
                    'l2scale': 1e-5, 'loss_factor': 2, 'enable_gpu': False}
    for key, expected in architecture.items():
        require(settings[key] == expected, f'Unsupported generator setting: {key}')
    config = TVAEGenerationConfig(**{key: settings[key] for key in ('epochs', 'batch_size', 'n_rows', 'seed')})
    print('Starting CPU TVAE fitting...', flush=True)
    print('CONFIG:', config, flush=True)
    started = time.perf_counter()
    artifact = generate_tvae_repair(preparation_directory=preparation, manifest_directory=manifest_directory,
                                    artifact_version=args.artifact_version, data_root=root, config=config)
    metadata, metadata_hash = verify_generation(directory, preparation, root, manifest_directory)
    context = {'purpose': 'development_generation_context', 'draft_yaml_sha256': draft.yaml_sha256,
               'generation_metadata_sha256': metadata_hash, 'elapsed_seconds': time.perf_counter() - started,
               'benchmark_execution': False, **git_context()}
    save_json(artifact.directory / 'generation_context.json', context)
    print('ELAPSED SECONDS:', round(context['elapsed_seconds'], 2))


if __name__ == '__main__':
    main()
