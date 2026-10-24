"""Audit existing synthetic class balance and processed-rule coverage only.

Place beside _common.py in components/c4/scripts. No fitting, approval,
release-validation evaluation or sealed-future evaluation is performed.
"""
import io
from pathlib import Path

import numpy as np
import pandas as pd

from _common import (
    checked_bytes, checked_json, git_context, manifest_root, name, parser,
    require, resolve_data_root, save_json, sha, utc_now,
)
from generate_development_repair import verify_generation
from c4.data import FEATURE_COLUMNS
from c4.scenarios.slices import SliceRule


def validate_frame(frame):
    require(tuple(frame.columns) == (*FEATURE_COLUMNS, 'target'),
            'Dataframe columns differ from the processed Adult schema.')
    require(not frame.empty, 'Cannot audit an empty population.')
    require(all(pd.api.types.is_numeric_dtype(frame[c]) for c in frame),
            'All processed columns must be numeric.')
    require(np.isfinite(frame.to_numpy(dtype=float)).all(),
            'Data contains missing or infinite values.')
    require(frame['target'].isin([0, 1]).all(), 'Invalid binary labels.')


def distribution(frame):
    validate_frame(frame)
    counts = frame['target'].value_counts().reindex([0, 1], fill_value=0)
    return {
        'rows': len(frame),
        'target_counts': {str(label): int(counts[label]) for label in (0, 1)},
        'positive_fraction': float(counts[1] / len(frame)),
    }


def summarize(fit, validation, membership, synthetic, rule):
    for frame in (fit, validation, synthetic):
        validate_frame(frame)
    require(membership.index.equals(validation.index),
            'Internal-validation membership indices differ.')
    require(pd.api.types.is_bool_dtype(membership.dtype)
            and not membership.isna().any(), 'Invalid saved membership.')
    require(bool(membership.any()) and bool((~membership).any()),
            'Internal validation must contain both slice populations.')
    populations = {
        'generator_fit': distribution(fit),
        'internal_global': distribution(validation),
        'internal_diagnosed_slice': distribution(validation.loc[membership]),
        'internal_non_affected': distribution(validation.loc[~membership]),
        'synthetic': distribution(synthetic),
    }
    synthetic_positive = populations['synthetic']['positive_fraction']
    gaps = {}
    for population in ('generator_fit', 'internal_diagnosed_slice'):
        difference = synthetic_positive - populations[population]['positive_fraction']
        gaps[population] = {
            'positive_fraction_difference': difference,
            'difference_percentage_points': 100.0 * difference,
            # For a binary target this equals total variation distance.
            'target_total_variation': abs(difference),
        }
    coverage = {}
    for population, frame in (('generator_fit', fit), ('synthetic', synthetic)):
        mask = rule.mask(frame)
        coverage[population] = {
            'rows': len(frame), 'rule_members': int(mask.sum()),
            'rule_nonmembers': int((~mask).sum()),
            'rule_coverage': float(mask.mean()),
        }
    return {'populations': populations, 'synthetic_target_comparisons': gaps,
            'processed_rule_coverage': coverage}


def evidence_note(report):
    lines = [
        '# Development repair distribution audit', '',
        f"Created: {report['created_at_utc']}", '',
        'Existing artifacts were inspected without training or changing them.', '',
        '| Population | Rows | Class 0 | Class 1 | Class 1 (%) |',
        '|---|---:|---:|---:|---:|',
    ]
    for population, value in report['populations'].items():
        counts = value['target_counts']
        lines.append(f"| {population} | {value['rows']} | {counts['0']} | "
                     f"{counts['1']} | {100 * value['positive_fraction']:.4f} |")
    lines += ['', 'Synthetic minus comparator class-1 proportion:', '']
    for population, value in report['synthetic_target_comparisons'].items():
        lines.append(f"- {population}: {value['difference_percentage_points']:+.4f} "
                     f"percentage points; binary target TV = {value['target_total_variation']:.6f}.")
    lines += ['', f"Processed diagnostic rule: `{report['diagnostic_rule_text']}`.", '',
              '| Population | Rule members | Rows | Coverage (%) |',
              '|---|---:|---:|---:|']
    for population, value in report['processed_rule_coverage'].items():
        lines.append(f"| {population} | {value['rule_members']} | {value['rows']} | "
                     f"{100 * value['rule_coverage']:.4f} |")
    lines += ['', 'Limitations:', '']
    lines.extend(f'- {item}' for item in report['limitations'])
    lines += ['', f"Generation metadata SHA256: `{report['generation_metadata_sha256']}`",
              f"Preparation metadata SHA256: `{report['preparation_metadata_sha256']}`", '']
    return '\n'.join(lines)


def main():
    p = parser(__doc__)
    p.add_argument('--preview-id', type=name, default='development-001')
    p.add_argument('--preparation-version', type=name, default='exact_slice-v1')
    p.add_argument('--manifest-version', type=name, default='v1')
    p.add_argument('--artifact-version', type=name, default='development-tvae-exact-v1')
    p.add_argument('--slice-feature', required=True)
    p.add_argument('--slice-operator', choices=['le', 'gt', 'eq'], required=True)
    p.add_argument('--slice-threshold', type=float, required=True)
    args = p.parse_args()
    root = resolve_data_root(args.data_root)
    directory = root / 'c4_artifacts' / 'offline_repair' / 'adult' / args.artifact_version
    preparation = (root / 'c4_artifacts' / 'repair_preparation' / 'adult'
                   / args.preview_id / args.preparation_version)
    # Refuse overwrites, including partially saved previous reports.
    filenames = ('distribution_audit.json', 'distribution_audit.sha256',
                 'distribution_audit.md')
    for filename in filenames:
        if (directory / filename).exists():
            raise FileExistsError(f'Report already exists: {directory / filename}')
    generation, generation_hash = verify_generation(
        directory, preparation, root, manifest_root(root, args.manifest_version))
    prepared, preparation_hash = checked_json(preparation)

    def prepared_csv(filename):
        contents = checked_bytes(preparation / filename, prepared['files'][filename])
        return pd.read_csv(io.BytesIO(contents), index_col=0)

    fit = prepared_csv('generator_fit.csv')
    validation = prepared_csv('internal_validation.csv')
    membership = prepared_csv('validation_diagnosed_membership.csv')['diagnosed_slice_member']
    synthetic = pd.read_csv(io.BytesIO(checked_bytes(
        directory / 'synthetic_unvalidated.csv', generation['synthetic_csv_sha256'])))
    rule = SliceRule(name='explicit_distribution_audit_rule', feature=args.slice_feature,
                     operator=args.slice_operator, value=args.slice_threshold)
    operators = {'le': '<=', 'gt': '>', 'eq': '=='}
    report = {
        'purpose': 'development_repair_distribution_audit',
        'created_at_utc': utc_now(), **git_context(),
        'generation_metadata_sha256': generation_hash,
        'preparation_metadata_sha256': preparation_hash,
        'synthetic_csv_sha256': generation['synthetic_csv_sha256'],
        'diagnostic_rule': rule.model_dump(mode='json'),
        'diagnostic_rule_text': f'{rule.feature} {operators[rule.operator]} {rule.value!r}',
        'diagnostic_rule_origin': 'explicit CLI arguments; confirm against scenario C2 rule',
        'internal_validation_membership_basis': 'saved clean-before-injection membership',
        'synthetic_coverage_basis': 'explicit rule applied to generated processed features',
        **summarize(fit, validation, membership, synthetic, rule),
        'privacy_status': 'NOT_EVALUATED', 'repair_package_approved': False,
        'benchmark_execution': False,
        'limitations': [
            'Development diagnostic; no automatic pass/fail quality threshold is applied.',
            'Class-balance differences alone do not establish generator collapse or utility.',
            'Slice coverage alone does not establish fidelity or safety.',
            'The diagnostic rule must match the intended processed C2 rule before contract use.',
            'Internal validation is from the repair role; no release/future outcomes were evaluated.',
            'No privacy pass, repair approval or novelty result is inferred.',
        ],
    }
    note = evidence_note(report).encode('utf-8')
    report['markdown_sha256'] = sha(note)
    checksum = save_json(directory / 'distribution_audit.json', report)
    with (directory / 'distribution_audit.md').open('xb') as file:
        file.write(note)
    for population, result in report['populations'].items():
        print('POPULATION:', population, result)
    for population, result in report['synthetic_target_comparisons'].items():
        print('SYNTHETIC TARGET COMPARISON:', population, result)
    print('PROCESSED RULE:', report['diagnostic_rule_text'])
    for population, result in report['processed_rule_coverage'].items():
        print('RULE COVERAGE:', population, result)
    print('REPORT:', directory / 'distribution_audit.json')
    print('EVIDENCE NOTE:', directory / 'distribution_audit.md')
    print('REPORT SHA256:', checksum)
    print('PRIVACY: NOT_EVALUATED; REPAIR APPROVED: False')


if __name__ == '__main__':
    main()
