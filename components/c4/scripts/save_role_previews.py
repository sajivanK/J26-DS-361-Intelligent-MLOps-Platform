"""Create or verify development transformations; not benchmark execution."""
from pathlib import Path
import io
import pandas as pd
from _common import (parser, resolve_data_root, default_yaml, name, manifest_root,
                     checked_json, checked_bytes, require, sha)
from c4.scenarios.benchmark import load_draft_benchmark, verify_draft_sources
from c4.scenarios.config import ScenarioConfig
from c4.scenarios.preview import save_role_preview
from c4.scenarios.slices import SliceRule
from c4.data.split_manifest import load_adult_split_manifest


def verify_preview(directory, root, manifest_directory, role):
    metadata, metadata_hash = checked_json(directory)
    require(metadata.get('purpose') == 'development_preview', 'Not a development preview.')
    require(metadata['role'] == role, 'Preview role differs.')
    manifest = load_adult_split_manifest(manifest_directory, data_root=root)
    require(sha((manifest_directory / 'manifest.json').read_bytes()) == metadata['split_manifest_sha256'],
            'Preview links to a different manifest.')
    require(metadata['source_test_csv_sha256'] == manifest['source_checksums']['test_csv_sha256'],
            'Preview source checksum differs.')
    data = {filename: checked_bytes(directory / filename, metadata['files'][filename])
            for filename in ('clean.csv', 'transformed.csv', 'true_membership.csv')}
    clean = pd.read_csv(io.BytesIO(data['clean.csv']), index_col=0)
    transformed = pd.read_csv(io.BytesIO(data['transformed.csv']), index_col=0)
    membership = pd.read_csv(io.BytesIO(data['true_membership.csv']))
    ids = manifest['partitions'][role]['source_row_ids']
    require(clean.index.is_unique and clean.index.tolist() == ids, 'Clean row IDs differ.')
    require(transformed.index.equals(clean.index), 'Transformed row IDs differ.')
    require(metadata['source_row_ids'] == ids, 'Metadata IDs differ.')
    require(membership['source_row_id'].tolist() == ids, 'Membership IDs differ.')
    rule = SliceRule.model_validate(metadata['transformation']['true_slice'])
    mask = rule.mask(clean)
    require(membership['true_slice_member'].tolist() == mask.tolist(), 'True membership differs.')
    pd.testing.assert_series_equal(clean['target'], transformed['target'])
    pd.testing.assert_frame_equal(clean.loc[~mask], transformed.loc[~mask])
    print('ROLE:', role)
    print('DIRECTORY:', directory)
    print('ROWS:', len(clean))
    print('AFFECTED ROWS:', int(mask.sum()))
    print('PRESERVATION / CHECKSUMS: PASS')
    print('METADATA SHA256:', metadata_hash)
    return metadata


def main():
    p = parser(__doc__)
    p.add_argument('--config', type=Path, default=default_yaml())
    p.add_argument('--preview-id', type=name, default='development-001')
    p.add_argument('--manifest-version', type=name, default='v1')
    p.add_argument('--mechanism', default='marginal_multi')
    p.add_argument('--magnitude', type=float, default=0.5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--verify-only', action='store_true')
    args = p.parse_args()
    root = resolve_data_root(args.data_root)
    manifest_directory = manifest_root(root, args.manifest_version)
    if not args.verify_only:
        draft = load_draft_benchmark(args.config)
        verified = verify_draft_sources(draft, data_root=root)
        require(draft.document['data_roles']['split_manifest_version'] == args.manifest_version,
                'Manifest version differs from draft.')
        matches = [config for config in draft.injection_configs
                   if config.mechanism == args.mechanism and config.magnitude == args.magnitude
                   and config.seed == args.seed]
        require(len(matches) == 1, 'Select one injection specification from the draft.')
        config = matches[0]
        true_slice = SliceRule.model_validate(
            verified['verified_slice_rules'][draft.document['true_slice']['name']])
        # Preflight both roles before writing either one.
        for role in ('repair', 'release_validation'):
            directory = root / 'c4_artifacts' / 'scenario_previews' / 'adult' / args.preview_id / role
            if directory.exists():
                raise FileExistsError(f'Preview already exists: {directory}; use --verify-only.')
        print('DRAFT YAML SHA256:', draft.yaml_sha256)
    for role in ('repair', 'release_validation'):
        directory = root / 'c4_artifacts' / 'scenario_previews' / 'adult' / args.preview_id / role
        if not args.verify_only:
            save_role_preview(manifest_directory=manifest_directory, config=config,
                              true_slice=true_slice, role=role, preview_id=args.preview_id, data_root=root)
        verify_preview(directory, root, manifest_directory, role)
    print('DEVELOPMENT PREVIEW VERIFIED; BENCHMARK NOT EXECUTED')


if __name__ == '__main__':
    main()
