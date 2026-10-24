"""Create or verify isolated internal repair fit/validation artifacts."""
from pathlib import Path
import io
import pandas as pd
from _common import (parser, resolve_data_root, default_yaml, name, manifest_root,
                     checked_json, checked_bytes, require, sha, save_json,
                     git_context, utc_now, C4_ROOT)
from c4.contracts.models import C2Diagnosis
from c4.experiments.repair_preparation import prepare_repair_data
from c4.scenarios.benchmark import load_draft_benchmark, verify_draft_sources
from c4.scenarios.diagnosis import diagnosed_slice_mask
from c4.data.split_manifest import load_adult_split_manifest
import json


def verify_preparation(directory, root, manifest_directory):
    metadata, metadata_hash = checked_json(directory)
    require(metadata.get('purpose') == 'development_repair_preparation', 'Wrong preparation purpose.')
    provenance = metadata['provenance']
    require(provenance['source_role'] == 'repair', 'Preparation is not repair-only.')
    manifest = load_adult_split_manifest(manifest_directory, data_root=root)
    require(sha((manifest_directory / 'manifest.json').read_bytes()) == provenance['split_manifest_sha256'],
            'Preparation manifest differs.')
    data = {filename: checked_bytes(directory / filename, metadata['files'][filename])
            for filename in ('generator_fit.csv', 'internal_validation.csv',
                             'validation_diagnosed_membership.csv')}
    fit = pd.read_csv(io.BytesIO(data['generator_fit.csv']), index_col=0)
    validation = pd.read_csv(io.BytesIO(data['internal_validation.csv']), index_col=0)
    membership = pd.read_csv(io.BytesIO(data['validation_diagnosed_membership.csv']), index_col=0)
    fit_ids = provenance['generator_fit_row_ids']
    train_ids = provenance['repair_train_row_ids']
    validation_ids = provenance['internal_validation_row_ids']
    repair_ids = manifest['partitions']['repair']['source_row_ids']
    for ids in (fit_ids, train_ids, validation_ids):
        require(len(ids) == len(set(ids)), 'Duplicate preparation IDs.')
    require(set(train_ids).isdisjoint(validation_ids), 'Internal train/validation overlap.')
    require(set(train_ids) | set(validation_ids) == set(repair_ids), 'Internal split is incomplete.')
    require(set(fit_ids) <= set(train_ids), 'Generator fit IDs are outside training.')
    require(fit.index.tolist() == fit_ids, 'Fitting CSV IDs differ.')
    require(validation.index.tolist() == validation_ids, 'Validation CSV IDs differ.')
    require(membership.index.tolist() == validation_ids, 'Membership CSV IDs differ.')
    require(pd.api.types.is_bool_dtype(membership['diagnosed_slice_member'].dtype), 'Invalid membership dtype.')
    require(not membership['diagnosed_slice_member'].isna().any(), 'Missing membership values.')
    require(len(fit) == metadata['generator_fit_rows'], 'Fit count differs.')
    require(len(validation) == metadata['internal_validation_rows'], 'Validation count differs.')
    require(int(membership['diagnosed_slice_member'].sum()) == metadata['diagnosed_internal_validation_rows'],
            'Diagnosed validation count differs.')
    print('DIRECTORY:', directory)
    print('GENERATOR FIT ROWS:', len(fit))
    print('INTERNAL VALIDATION ROWS:', len(validation))
    print('DIAGNOSED INTERNAL VALIDATION ROWS:', int(membership['diagnosed_slice_member'].sum()))
    print('PARTITION IDS / CHECKSUMS: PASS')
    print('METADATA SHA256:', metadata_hash)
    return metadata


def main():
    p = parser(__doc__)
    p.add_argument('--config', type=Path, default=default_yaml())
    p.add_argument('--preview-id', type=name, default='development-001')
    p.add_argument('--preparation-version', type=name, default='exact_slice-v1')
    p.add_argument('--manifest-version', type=name, default='v1')
    p.add_argument('--diagnosis-condition', choices=['exact_slice', 'broadened_slice'], default='exact_slice')
    p.add_argument('--verify-only', action='store_true')
    args = p.parse_args()
    root = resolve_data_root(args.data_root)
    directory = root / 'c4_artifacts' / 'repair_preparation' / 'adult' / args.preview_id / args.preparation_version
    manifest_directory = manifest_root(root, args.manifest_version)
    if args.verify_only:
        verify_preparation(directory, root, manifest_directory)
        return
    if directory.exists():
        raise FileExistsError(f'Preparation already exists: {directory}; use --verify-only.')
    draft = load_draft_benchmark(args.config)
    verify_draft_sources(draft, data_root=root)
    require(draft.document['data_roles']['split_manifest_version'] == args.manifest_version,
            'Manifest version differs from draft.')
    conditions = {item['name']: item for item in draft.document['diagnosis_conditions']}
    slice_spec = conditions[args.diagnosis_condition]['diagnosed_slice']
    if slice_spec == 'same_as_true_slice':
        slice_spec = draft.document['true_slice']
    operators = {'le': '<=', 'gt': '>', 'eq': '=='}
    fixture = json.loads((C4_ROOT / 'tests' / 'fixtures' / 'valid_upstream_input.json').read_text(encoding='utf-8'))
    fields = fixture['c2_diagnosis']
    fields.update(root_cause_category='marginal', root_cause_feature='education_num',
                  affected_slice_rule=f"{slice_spec['feature']} {operators[slice_spec['operator']]} {slice_spec['resolved_threshold']!r}",
                  affected_features_ranked=['education_num', 'hours_per_week'],
                  diagnosis_version=f'development-{args.diagnosis_condition}-v1')
    diagnosis = C2Diagnosis.model_validate(fields)
    settings = draft.document['repair_protocol']['internal_validation_split']
    require(settings['source_role'] == 'repair' and settings['stratify_by'] == 'target'
            and settings['split_before_diagnosed_slice_filtering'] is True, 'Unsupported internal split protocol.')
    prepared = prepare_repair_data(
        preview_directory=root / 'c4_artifacts' / 'scenario_previews' / 'adult' / args.preview_id / 'repair',
        manifest_directory=manifest_directory, diagnosis=diagnosis, rule_feature_space='processed',
        data_root=root, validation_fraction=settings['validation_fraction'], split_seed=settings['split_seed'])
    files = {
        'generator_fit.csv': prepared.generator_fit.to_csv(index=True, index_label='source_row_id', lineterminator='\n').encode(),
        'internal_validation.csv': prepared.internal_validation.to_csv(index=True, index_label='source_row_id', lineterminator='\n').encode(),
        'validation_diagnosed_membership.csv': prepared.validation_diagnosed_membership.rename('diagnosed_slice_member').rename_axis('source_row_id').to_csv(lineterminator='\n').encode(),
    }
    metadata = {
        'purpose': 'development_repair_preparation', 'created_at_utc': utc_now(), **git_context(),
        'draft_yaml_sha256': draft.yaml_sha256, 'generator_fitted': False, 'repair_package_approved': False,
        'diagnosis_evidence_origin': 'simulated rule with fixture confidence fields; not measured C2 output',
        'generator_fit_rows': len(prepared.generator_fit), 'internal_validation_rows': len(prepared.internal_validation),
        'diagnosed_internal_validation_rows': int(prepared.validation_diagnosed_membership.sum()),
        'provenance': prepared.provenance, 'files': {filename: sha(contents) for filename, contents in files.items()},
    }
    directory.mkdir(parents=True, exist_ok=False)
    for filename, contents in files.items():
        (directory / filename).write_bytes(contents)
    save_json(directory / 'metadata.json', metadata)
    verify_preparation(directory, root, manifest_directory)


if __name__ == '__main__':
    main()
