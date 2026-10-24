"""Save descriptive distance diagnostics for an existing unvalidated repair."""
import io

import pandas as pd

from _common import (
    parser, name, resolve_data_root, manifest_root, checked_json, checked_bytes,
    require, save_json, sha, utc_now, git_context,
)
from generate_development_repair import verify_generation
from check_repair_distribution import validate_frame
from _privacy_distance import fit_distance_spec, nearest_distances, distance_summary
from c4.scenarios.feature_metadata import load_adult_feature_metadata
from c4.data import FEATURE_COLUMNS


def main():
    p = parser(__doc__)
    p.add_argument('--preview-id', type=name, default='development-001')
    p.add_argument('--preparation-version', type=name, default='exact_slice-v1')
    p.add_argument('--manifest-version', type=name, default='v1')
    p.add_argument('--artifact-version', type=name, default='development-tvae-exact-v1')
    p.add_argument('--chunk-size', type=int, default=128)
    args = p.parse_args()
    require(args.chunk_size > 0, 'Chunk size must be positive.')
    root = resolve_data_root(args.data_root)
    directory = root / 'c4_artifacts' / 'offline_repair' / 'adult' / args.artifact_version
    preparation = (root / 'c4_artifacts' / 'repair_preparation' / 'adult'
                   / args.preview_id / args.preparation_version)
    for filename in ('privacy_distances.csv', 'privacy_diagnostics.json',
                     'privacy_diagnostics.sha256', 'privacy_diagnostics.md'):
        if (directory / filename).exists():
            raise FileExistsError(f'Privacy diagnostic already exists: {directory / filename}')
    generation, generation_hash = verify_generation(
        directory, preparation, root, manifest_root(root, args.manifest_version))
    prepared, preparation_hash = checked_json(preparation)

    def read_prepared(filename):
        return pd.read_csv(io.BytesIO(checked_bytes(
            preparation / filename, prepared['files'][filename])), index_col=0)

    fit = read_prepared('generator_fit.csv')
    validation = read_prepared('internal_validation.csv')
    membership = read_prepared('validation_diagnosed_membership.csv')['diagnosed_slice_member']
    require(membership.index.equals(validation.index), 'Membership indices differ.')
    require(pd.api.types.is_bool_dtype(membership.dtype) and not membership.isna().any(),
            'Invalid saved membership.')
    diagnosed = validation.loc[membership]
    synthetic = pd.read_csv(io.BytesIO(checked_bytes(
        directory / 'synthetic_unvalidated.csv', generation['synthetic_csv_sha256'])))
    for frame in (fit, validation, diagnosed, synthetic):
        validate_frame(frame)
    feature_metadata = load_adult_feature_metadata()
    require(feature_metadata.source_sha256 == generation['feature_metadata_sha256'],
            'Feature metadata differs from generation.')
    types = {feature: feature_metadata.feature_types[feature] for feature in FEATURE_COLUMNS}
    spec = fit_distance_spec(fit, types)
    comparisons = {
        'synthetic_to_generator_fit': (synthetic, fit, 'synthetic_row_position'),
        'synthetic_to_diagnosed_internal_validation': (synthetic, diagnosed, 'synthetic_row_position'),
        'diagnosed_internal_validation_to_generator_fit': (diagnosed, fit, 'source_row_id'),
    }
    summaries, records = {}, []
    for label, (query, reference, identifier) in comparisons.items():
        values = nearest_distances(query, reference, spec, args.chunk_size)
        summaries[label] = distance_summary(values)
        summaries[label]['reference_rows'] = len(reference)
        ids = list(range(len(query))) if identifier == 'synthetic_row_position' else query.index.tolist()
        records.extend({'comparison': label, 'query_id_kind': identifier,
                        'query_id': int(row_id), 'nearest_distance': float(value)}
                       for row_id, value in zip(ids, values))
        print('DISTANCE SUMMARY:', label, summaries[label], flush=True)
    csv_bytes = pd.DataFrame(records).to_csv(index=False, lineterminator='\n').encode('utf-8')
    limitations = [
        'Descriptive mixed-feature distance, not a privacy guarantee or differential privacy.',
        'Zero distance means an exact processed-feature match; it does not identify a person.',
        'Target and row identifiers are excluded from distance calculation.',
        'Generator-fit ranges define scaling; continuous contributions are capped at one.',
        'Comparison populations have different sizes; do not interpret distances as a calibrated risk score.',
        'Only diagnosed generator-fit records are used as the training-distance reference.',
        'No numeric approval threshold is defined; privacy remains NOT_EVALUATED for gating.',
        'No release-validation or sealed-future outcomes are evaluated.',
    ]
    lines = ['# Development privacy-distance diagnostics', '',
             'These measurements do not approve the repair package.', '',
             '| Comparison | Query rows | Reference rows | Minimum | Median | Exact matches |',
             '|---|---:|---:|---:|---:|---:|']
    for label, result in summaries.items():
        lines.append(f"| {label} | {result['rows']} | {result['reference_rows']} | "
                     f"{result['minimum']:.6f} | {result['median']:.6f} | {result['exact_feature_match_rows']} |")
    lines += ['', 'Protocol: equal feature weights; continuous distance is the absolute',
              'difference divided by the generator-fit range, capped at one. Constant',
              'continuous features and categorical codes use equality mismatch.', '', 'Limitations:', '']
    lines.extend('- ' + item for item in limitations)
    lines += ['', f'Generation metadata SHA256: `{generation_hash}`',
              f'Preparation metadata SHA256: `{preparation_hash}`', '']
    note = '\n'.join(lines).encode('utf-8')
    report = {
        'purpose': 'development_privacy_distance_diagnostics',
        'created_at_utc': utc_now(), **git_context(),
        'generation_metadata_sha256': generation_hash,
        'preparation_metadata_sha256': preparation_hash,
        'feature_metadata_sha256': feature_metadata.source_sha256,
        'synthetic_csv_sha256': generation['synthetic_csv_sha256'],
        'distance_specification': spec, 'summaries': summaries,
        'distances_csv_sha256': sha(csv_bytes), 'markdown_sha256': sha(note),
        'distance_measurements_completed': True,
        'privacy_approval_rule': 'UNRESOLVED', 'privacy_status': 'NOT_EVALUATED',
        'repair_package_approved': False, 'benchmark_execution': False,
        'limitations': limitations,
    }
    checksum = save_json(directory / 'privacy_diagnostics.json', report)
    for filename, contents in (('privacy_distances.csv', csv_bytes), ('privacy_diagnostics.md', note)):
        with (directory / filename).open('xb') as file:
            file.write(contents)
    print('REPORT:', directory / 'privacy_diagnostics.json')
    print('REPORT SHA256:', checksum)
    print('EVIDENCE NOTE:', directory / 'privacy_diagnostics.md')
    print('PRIVACY APPROVAL RULE: UNRESOLVED')
    print('PRIVACY: NOT_EVALUATED; REPAIR APPROVED: False')


if __name__ == '__main__':
    main()
