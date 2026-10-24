"""Consolidate measured development evidence without issuing C3 approval."""
from _common import (
    parser, name, resolve_data_root, manifest_root, checked_json, checked_bytes,
    require, save_json, sha, utc_now, git_context,
)
from generate_development_repair import verify_generation


def main():
    p = parser(__doc__)
    p.add_argument('--preview-id', type=name, default='development-001')
    p.add_argument('--preparation-version', type=name, default='exact_slice-v1')
    p.add_argument('--manifest-version', type=name, default='v1')
    p.add_argument('--artifact-version', type=name, default='development-tvae-exact-v1')
    args = p.parse_args()
    root = resolve_data_root(args.data_root)
    directory = root / 'c4_artifacts' / 'offline_repair' / 'adult' / args.artifact_version
    preparation = (root / 'c4_artifacts' / 'repair_preparation' / 'adult'
                   / args.preview_id / args.preparation_version)
    for filename in ('readiness.json', 'readiness.sha256', 'C4_REPAIR_EVIDENCE.md'):
        if (directory / filename).exists():
            raise FileExistsError(f'Readiness output already exists: {directory / filename}')
    generation, generation_hash = verify_generation(
        directory, preparation, root, manifest_root(root, args.manifest_version))
    _, preparation_hash = checked_json(preparation)
    documents, hashes = {}, {}
    for stem in ('diagnostics', 'distribution_audit', 'privacy_diagnostics'):
        document, checksum = checked_json(directory, stem)
        require(document['generation_metadata_sha256'] == generation_hash,
                f'{stem}: generation link differs.')
        require(document['preparation_metadata_sha256'] == preparation_hash,
                f'{stem}: preparation link differs.')
        require(document['repair_package_approved'] is False,
                f'{stem}: unexpected approval in development evidence.')
        require(document['benchmark_execution'] is False,
                f'{stem}: unexpected benchmark-execution flag.')
        documents[stem], hashes[stem] = document, checksum
    utility = documents['diagnostics']
    distribution = documents['distribution_audit']
    privacy = documents['privacy_diagnostics']
    checked_bytes(directory / 'utility_probe.ubj', utility['probe_model_sha256'])
    for stem in ('distribution_audit', 'privacy_diagnostics'):
        checked_bytes(directory / f'{stem}.md', documents[stem]['markdown_sha256'])
        require(documents[stem]['synthetic_csv_sha256'] == generation['synthetic_csv_sha256'],
                f'{stem}: synthetic-data link differs.')
    checked_bytes(directory / 'privacy_distances.csv', privacy['distances_csv_sha256'])
    blockers = [
        'A justified, versioned privacy approval rule is not yet defined.',
        'The benchmark design and remaining upstream evidence procedures are still draft.',
        'Measured development diagnostics do not constitute an APPROVED C3 package.',
    ]
    lines = ['# C4 development repair evidence', '',
             f'Artifact: `{args.artifact_version}`', f'Created: {utc_now()}', '',
             'Status: UNVALIDATED. Candidate training through the approved-package workflow is not authorized.', '',
             '## Class distributions', '',
             '| Population | Rows | Class 0 | Class 1 | Class 1 (%) |',
             '|---|---:|---:|---:|---:|']
    for population, result in distribution['populations'].items():
        counts = result['target_counts']
        lines.append(f"| {population} | {result['rows']} | {counts['0']} | {counts['1']} | "
                     f"{100 * result['positive_fraction']:.4f} |")
    lines += ['', '## Slice coverage', '']
    for population, result in distribution['processed_rule_coverage'].items():
        lines.append(f"- {population}: {result['rule_members']}/{result['rows']} "
                     f"({100 * result['rule_coverage']:.4f}%).")
    lines += ['', '## Internal utility', '',
              '| Population | Rows | Champion accuracy | Probe accuracy | Margin (percentage points) |',
              '|---|---:|---:|---:|---:|']
    for population, result in utility['internal_utility_comparisons'].items():
        if result.get('champion_accuracy') is not None:
            lines.append(f"| {population} | {result['rows']} | {result['champion_accuracy']:.6f} | "
                         f"{result['probe_accuracy']:.6f} | {100 * result['accuracy_margin']:+.6f} |")
    lines += ['', f"Mean univariate fidelity: {utility['mean_univariate_fidelity']:.6f}.",
              'This is a descriptive feature-distribution score, not accuracy or overall repair validity.', '',
              '## Privacy-distance observations', '',
              '| Comparison | Minimum | 5th percentile | Median | Exact feature matches |',
              '|---|---:|---:|---:|---:|']
    for comparison, result in privacy['summaries'].items():
        lines.append(f"| {comparison} | {result['minimum']:.6f} | {result['p05']:.6f} | "
                     f"{result['median']:.6f} | {result['exact_feature_match_rows']} |")
    lines += ['', '## Interpretation limits', '',
              '- No privacy guarantee or statistical significance is inferred.',
              '- The utility probe is an offline development model, not an approved release candidate.',
              '- These observations are not release-validation or sealed-future outcomes.',
              '- Keep this original experiment even if a later version changes the generator.', '',
              '## Remaining blockers', '']
    lines.extend('- ' + item for item in blockers)
    lines += ['', '## Evidence checksums', '']
    for label, checksum in {'generation_metadata': generation_hash,
                            'preparation_metadata': preparation_hash, **hashes}.items():
        lines.append(f'- {label}: `{checksum}`')
    note = ('\n'.join(lines) + '\n').encode('utf-8')
    report = {
        'purpose': 'development_repair_readiness_snapshot',
        'created_at_utc': utc_now(), **git_context(),
        'artifact_version': args.artifact_version,
        'generation_metadata_sha256': generation_hash,
        'preparation_metadata_sha256': preparation_hash,
        'evidence_sha256': hashes, 'markdown_sha256': sha(note),
        'validation_status': 'UNVALIDATED', 'privacy_status': 'NOT_EVALUATED',
        'repair_package_approved': False, 'candidate_training_authorized': False,
        'benchmark_execution': False, 'blockers': blockers,
        'measured_evidence': documents,
    }
    checksum = save_json(directory / 'readiness.json', report)
    with (directory / 'C4_REPAIR_EVIDENCE.md').open('xb') as file:
        file.write(note)
    print('EVIDENCE FILE CHECKSUMS / ARTIFACT LINKS: PASS')
    print('REPORT:', directory / 'readiness.json')
    print('REPORT SHA256:', checksum)
    print('THESIS EVIDENCE:', directory / 'C4_REPAIR_EVIDENCE.md')
    print('STATUS: UNVALIDATED; PRIVACY: NOT_EVALUATED')
    print('APPROVED-PACKAGE CANDIDATE TRAINING AUTHORIZED: False')
    for item in blockers:
        print('BLOCKER:', item)


if __name__ == '__main__':
    main()
