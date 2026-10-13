"""Record the already-generated seed-123 pilot under its frozen screen."""
import json

from _common import parser, resolve_data_root, checked_json, checked_bytes, require, save_json, utc_now, git_context
from _scoped_benchmark import frozen_tail_screen


def main():
    args = parser(__doc__).parse_args()
    root = resolve_data_root(args.data_root)
    frozen_directory = root / 'c4_artifacts/protocols/offline_pilot/adult/adult-prospective-pilot-v1'
    package = root / 'c4_artifacts/offline_repair/adult/pilot-independent-tvae-v1'
    frozen, freeze_hash = checked_json(frozen_directory, 'freeze')
    protocol = json.loads(checked_bytes(frozen_directory / 'protocol.json', frozen['protocol_sha256']))
    generation, generation_hash = checked_json(package)
    context, context_hash = checked_json(package, 'pilot_context')
    claim, _ = checked_json(frozen_directory, 'attempt_claim')
    privacy, privacy_hash = checked_json(package, 'privacy_diagnostics')
    require(frozen['status'] == 'frozen_for_pilot_only', 'Wrong pilot freeze status.')
    require(context['freeze_sha256'] == freeze_hash and claim['freeze_sha256'] == freeze_hash,
            'Pilot does not link to the frozen attempt.')
    require(context['generation_metadata_sha256'] == generation_hash
            and privacy['generation_metadata_sha256'] == generation_hash, 'Generation links differ.')
    require(generation['preparation_metadata_sha256'] == frozen['preparation_metadata_sha256'],
            'Preparation differs from frozen pilot.')
    require(generation['artifact_version'] == protocol['generator_attempt']['artifact_version'],
            'Wrong pilot artifact.')
    for key in ('epochs', 'batch_size', 'n_rows', 'seed'):
        require(generation['generation_config'][key] == protocol['generator_attempt'][key],
                f'Pilot generation setting differs: {key}')
    checked_bytes(package / 'generator.pkl', generation['generator_sha256'])
    checked_bytes(package / 'synthetic_unvalidated.csv', generation['synthetic_csv_sha256'])
    checked_bytes(package / 'privacy_distances.csv', privacy['distances_csv_sha256'])
    summaries = privacy['summaries']
    result = frozen_tail_screen(protocol, summaries['synthetic_to_generator_fit'],
                                summaries['diagnosed_internal_validation_to_generator_fit'],
                                protocol_kind='pilot')
    report = {**result, 'purpose': 'frozen_pilot_screen_assessment', 'created_at_utc': utc_now(),
              **git_context(), 'freeze_sha256': freeze_hash, 'pilot_context_sha256': context_hash,
              'generation_metadata_sha256': generation_hash, 'privacy_report_sha256': privacy_hash,
              'simulation_admission': 'SCREEN_PASS_REQUIRES_REMAINING_CHECKS' if result['screen_status'] == 'PASS' else 'BLOCKED',
              'candidate_training_authorized': False, 'future_evaluation_authorized': False,
              'production_package_approved': False}
    checksum = save_json(package / 'frozen_pilot_assessment.json', report)
    print('FROZEN PILOT SCREEN:', result['screen_status'])
    print('TAIL DISTANCE RATIO:', result['tail_distance_ratio'])
    print('SIMULATION ADMISSION:', report['simulation_admission'])
    print('REPORT:', package / 'frozen_pilot_assessment.json')
    print('REPORT SHA256:', checksum)
    print('CANDIDATE / FUTURE AUTHORIZATION: False; ORIGINAL EVIDENCE PRESERVED')


if __name__ == '__main__':
    main()
