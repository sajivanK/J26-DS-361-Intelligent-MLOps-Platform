"""Record a development mandatory-stop trace; no models are trained here."""
from _common import (
    parser, name, resolve_data_root, manifest_root, checked_json, checked_bytes,
    require, save_json, sha, utc_now, git_context,
)
from generate_development_repair import verify_generation
from c4.experiments.repair_admission import assess_repair_admission


def main():
    p = parser(__doc__)
    p.add_argument('--preview-id', type=name, default='development-001')
    p.add_argument('--preparation-version', type=name, default='exact_slice-v1')
    p.add_argument('--manifest-version', type=name, default='v1')
    p.add_argument('--artifact-version', type=name, default='development-tvae-target-strat-v1')
    p.add_argument('--upstream-version', type=name, default='mock-exact-v1')
    p.add_argument('--protocol-version', type=name, default='adult-offline-validation-draft-v1')
    args = p.parse_args()
    root = resolve_data_root(args.data_root)
    package = root / 'c4_artifacts' / 'offline_repair' / 'adult' / args.artifact_version
    preparation = (root / 'c4_artifacts' / 'repair_preparation' / 'adult'
                   / args.preview_id / args.preparation_version)
    upstream_directory = (root / 'c4_artifacts' / 'upstream_previews' / 'adult'
                          / args.preview_id / args.upstream_version)
    protocol_directory = package / 'protocol_assessments' / args.protocol_version
    output = package / 'admission_checks' / args.upstream_version / args.protocol_version
    if output.exists():
        raise FileExistsError(f'Admission trace already exists: {output}')
    generation, generation_hash = verify_generation(
        package, preparation, root, manifest_root(root, args.manifest_version))
    _, preparation_hash = checked_json(preparation)
    upstream, upstream_hash = checked_json(upstream_directory, 'evidence')
    assessment, assessment_hash = checked_json(protocol_directory, 'assessment')
    require(upstream.get('purpose') == 'engineering_mock_upstream_preview',
            'This runner is scoped to the engineering mock preview.')
    require(assessment['generation_metadata_sha256'] == generation_hash,
            'Assessment references a different repair artifact.')
    require(assessment['preparation_metadata_sha256'] == preparation_hash,
            'Assessment preparation differs.')
    checked_bytes(protocol_directory / 'protocol.json', assessment['protocol_sha256'])
    checked_bytes(protocol_directory / 'assessment.md', assessment['markdown_sha256'])
    preview_directory = (root / 'c4_artifacts' / 'scenario_previews' / 'adult'
                         / args.preview_id / 'release_validation')
    _, preview_hash = checked_json(preview_directory)
    require(upstream['preview_metadata_sha256'] == preview_hash,
            'Upstream evidence points to a different release preview.')
    result = assess_repair_admission(upstream, generation, assessment)
    report = {
        **result, 'created_at_utc': utc_now(), **git_context(),
        'generation_metadata_sha256': generation_hash,
        'preparation_metadata_sha256': preparation_hash,
        'upstream_evidence_sha256': upstream_hash,
        'protocol_assessment_sha256': assessment_hash,
        'candidate_training_executed': False,
        'release_evaluation_executed': False,
        'future_evaluation_executed': False,
        'benchmark_execution': False,
        'scope_note': 'Mandatory-stop development trace; not the integrated N1-N4 controller.',
    }
    lines = ['# C4 development repair admission', '',
             f"Package: `{args.artifact_version}`", '',
             f"Admission: **{result['admission_status']}**", '',
             f"Development preflight action: **{result['preflight_decision']}**", '',
             '## Failed checks', '']
    lines.extend('- ' + reason for reason in result['reasons'])
    lines += ['', 'No candidate training, release evaluation, future evaluation, or N1-N4 inference',
              'was performed by this script. This is an early-stop development trace.', '',
              'Producer-origin checks are specific to the real-evidence path. Controlled mock',
              'experiments may be used in a separately labelled simulation protocol; they cannot',
              'be relabelled as actual C1/C2 producer results.', '']
    note = '\n'.join(lines).encode('utf-8')
    report['markdown_sha256'] = sha(note)
    checksum = save_json(output / 'admission.json', report)
    with (output / 'admission.md').open('xb') as file:
        file.write(note)
    print('ADMISSION:', result['admission_status'])
    print('DEVELOPMENT PREFLIGHT DECISION:', result['preflight_decision'])
    for key in result['failed_checks']:
        print('FAILED CHECK:', key)
    print('CANDIDATE TRAINING EXECUTED: False')
    print('FUTURE EVALUATION EXECUTED: False')
    print('REPORT:', output / 'admission.json')
    print('REPORT SHA256:', checksum)
    print('FINAL N1-N4 CONTROLLER COMPLETE: False')


if __name__ == '__main__':
    main()
