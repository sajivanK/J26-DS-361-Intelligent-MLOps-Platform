"""Apply the draft development-screen protocol to preserved repair evidence."""
import json

from _common import (
    parser, name, resolve_data_root, manifest_root, checked_json, checked_bytes,
    require, save_json, sha, utc_now, git_context, C4_ROOT,
)
from generate_development_repair import verify_generation
from _repair_validation import assess_privacy_screen


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate protocol key: {key}')
        result[key] = value
    return result


def main():
    p = parser(__doc__)
    p.add_argument('--preview-id', type=name, default='development-001')
    p.add_argument('--preparation-version', type=name, default='exact_slice-v1')
    p.add_argument('--manifest-version', type=name, default='v1')
    p.add_argument('--artifact-version', type=name, default='development-tvae-exact-v1')
    args = p.parse_args()
    config_path = C4_ROOT / 'configs' / 'repair_validation_protocol.json'
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes, object_pairs_hook=unique_object)
    require(config['schema_version'] == '1.0' and config['status'] == 'draft',
            'This implementation handles draft v1 screening only.')
    require(config['scope'] == 'offline_development_screen_only'
            and config['allows_c3_approval_export'] is False, 'Unsupported approval scope.')
    supported = {
        'metric': 'equal_weight_mixed_distance_fit_ranges_clipped',
        'features': 'input_features_only', 'scale_population': 'generator_fit_only',
        'synthetic_comparison': 'synthetic_to_generator_fit',
        'baseline_comparison': 'diagnosed_internal_validation_to_generator_fit',
        'tail_quantile': 0.05, 'maximum_exact_feature_match_rows': 0,
        'minimum_tail_distance_ratio': 1.0, 'zero_baseline_action': 'INDETERMINATE',
    }
    for key, expected in supported.items():
        require(config['privacy_screen'][key] == expected, f'Unsupported screen setting: {key}')
    require(config['data_restrictions']['source_role'] == 'repair', 'Wrong source role.')
    for key in ('use_release_validation_for_screen', 'use_sealed_future_for_screen',
                'refit_generator', 'overwrite_previous_evidence'):
        require(config['data_restrictions'][key] is False, f'Forbidden data setting: {key}')
    root = resolve_data_root(args.data_root)
    directory = root / 'c4_artifacts' / 'offline_repair' / 'adult' / args.artifact_version
    preparation = (root / 'c4_artifacts' / 'repair_preparation' / 'adult'
                   / args.preview_id / args.preparation_version)
    output = directory / 'protocol_assessments' / name(config['protocol_version'])
    if output.exists():
        raise FileExistsError(f'Assessment already exists: {output}')
    generation, generation_hash = verify_generation(
        directory, preparation, root, manifest_root(root, args.manifest_version))
    _, preparation_hash = checked_json(preparation)
    reports, hashes = {}, {}
    for stem in ('diagnostics', 'distribution_audit', 'privacy_diagnostics'):
        report, checksum = checked_json(directory, stem)
        require(report['generation_metadata_sha256'] == generation_hash,
                f'{stem}: generation metadata differs.')
        require(report['preparation_metadata_sha256'] == preparation_hash,
                f'{stem}: preparation metadata differs.')
        require(report['repair_package_approved'] is False
                and report['benchmark_execution'] is False, f'{stem}: unexpected approval/execution flag.')
        reports[stem], hashes[stem] = report, checksum
    privacy = reports['privacy_diagnostics']
    spec = privacy['distance_specification']
    require(spec['scale_population'] == 'generator_fit_only'
            and spec['target_included'] is False
            and spec['weights'] == 'equal_per_feature'
            and spec['continuous_rule'] == 'min(abs_difference / fit_range, 1)'
            and spec['constant_continuous_rule'] == '0_if_equal_else_1'
            and spec['categorical_rule'] == '0_if_equal_else_1', 'Distance protocol differs.')
    checked_bytes(directory / 'privacy_distances.csv', privacy['distances_csv_sha256'])
    checked_bytes(directory / 'utility_probe.ubj', reports['diagnostics']['probe_model_sha256'])
    synthetic_summary = privacy['summaries']['synthetic_to_generator_fit']
    baseline_summary = privacy['summaries']['diagnosed_internal_validation_to_generator_fit']
    require(synthetic_summary['reference_rows'] == baseline_summary['reference_rows'],
            'DCR comparisons must use the same fit-reference population size.')
    assessment = assess_privacy_screen(synthetic_summary, baseline_summary)
    distribution = reports['distribution_audit']
    scores = {
        'mean_univariate_fidelity': reports['diagnostics']['mean_univariate_fidelity'],
        'signed_internal_slice_utility': reports['diagnostics']['internal_utility_comparisons'][
            'internal_diagnosed_slice']['accuracy_margin'],
        'explicit_rule_target_coverage': distribution['processed_rule_coverage']['synthetic']['rule_coverage'],
        'target_total_variation_vs_fit': distribution['synthetic_target_comparisons'][
            'generator_fit']['target_total_variation'],
        'prescription_confidence': None,
    }
    lines = ['# Draft offline repair-validation assessment', '',
             f"Protocol: `{config['protocol_version']}`", '',
             f"**Exploratory screen: {assessment['screen_status']}**", '',
             assessment['reason'], '',
             f"Synthetic p05: {assessment['synthetic_p05']}",
             f"Held-out real p05: {assessment['baseline_p05']}",
             f"Ratio: {assessment['tail_distance_ratio']}", '',
             '## Measured evidence', '']
    lines.extend(f'- {key}: {value}' for key, value in scores.items())
    lines += ['', '## Scope', '', config['design_context'], '',
              config['privacy_screen']['rationale'], '',
              'No approval is exported. Original generation metadata and measurements are preserved.',
              'The screen is a design proposal; it is not an established C3 or literature threshold.',
              'Synthetic class imbalance and non-affected utility regression remain in the original evidence.', '',
              '## Requirements before benchmark approval', '']
    lines.extend('- ' + requirement for requirement in config['remaining_requirements'])
    note = ('\n'.join(lines) + '\n').encode('utf-8')
    report = {
        'purpose': 'exploratory_offline_validation_protocol_assessment',
        'created_at_utc': utc_now(), **git_context(),
        'protocol_version': config['protocol_version'], 'protocol_status': 'draft',
        'protocol_sha256': sha(config_bytes), 'protocol': config,
        'generation_metadata_sha256': generation_hash,
        'preparation_metadata_sha256': preparation_hash,
        'input_report_sha256': hashes, 'scores': scores,
        **assessment, 'benchmark_execution': False, 'markdown_sha256': sha(note),
    }
    output.mkdir(parents=True, exist_ok=False)
    with (output / 'protocol.json').open('xb') as file:
        file.write(config_bytes)
    checksum = save_json(output / 'assessment.json', report)
    with (output / 'assessment.md').open('xb') as file:
        file.write(note)
    print('PROTOCOL:', config['protocol_version'])
    print('PROTOCOL SHA256:', sha(config_bytes))
    print('EXPLORATORY SCREEN:', assessment['screen_status'])
    print('TAIL DISTANCE RATIO:', assessment['tail_distance_ratio'])
    print('REASON:', assessment['reason'])
    print('MEASURED SCORES:', scores)
    print('ASSESSMENT:', output / 'assessment.json')
    print('ASSESSMENT SHA256:', checksum)
    print('PRIVACY: NOT_EVALUATED; APPROVED: False')


if __name__ == '__main__':
    main()
