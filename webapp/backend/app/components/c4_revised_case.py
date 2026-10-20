"""Replay a pinned revised outer case without model or future-outcome reads.

The primary case is selected by identifier before inspecting its outcomes. N1
predictions and N2/N3/N4 results are saved evidence, not fresh model inference.
"""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path

VERSION = 'adult-hybrid-fresh-validation-v1'
CASE_ID = 'revised-outer-case'
SCENARIO_ID = 'adult-ns51001-r004-interaction-low-broadened_slice-label_corrupted_repair'
PROTOCOL_SHA256 = 'c6241cfe55680a767ecfd381294be0852bf4fced6b8c6cf2da4f4fa880dcf837'
BATCH_SHA256 = '13d55fc0c22b79404b1669748fc009ef6cde9febf9bc9b8eea16135d6046e33c'
METHODS = {
    'hybrid': 'hybrid_ridge_lightgbm_lightgbm',
    'lightgbm': 'same_external_training_lightgbm',
    'linear': 'same_external_training_linear',
    'constant': 'same_external_training_constant_quantiles',
}
TARGETS = ('future_global_margin', 'future_slice_margin', 'future_nonaffected_margin')
BOUNDARIES = dict(zip(TARGETS, (0., 0., -.02)))
GATES = ('repair_approved', 'privacy_check', 'diagnosis_stability', 'schema_compatible', 'rollback_ready')


class RevisedCaseUnavailable(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise ValueError(message)


def document(directory, stem):
    expected = (directory / (stem + '.sha256')).read_text(encoding='utf-8').strip()
    contents = checked(directory / (stem + '.json'), expected)
    payload = json.loads(contents)
    require(isinstance(payload, dict), 'Expected JSON object.')
    return payload, expected


def checked(path, expected):
    contents = path.read_bytes()
    require(hashlib.sha256(contents).hexdigest() == expected, 'Case artifact checksum differs.')
    return contents


def numeric(value):
    require(type(value) in (int, float) and math.isfinite(value), 'Invalid numeric evidence.')
    return float(value)


def feature_row(upstream, evidence):
    """Fixed original _scenario_meta.feature_row encoding, with no model imports."""
    values = {}
    def add(key, value):
        require(type(value) in (int, float, bool) and math.isfinite(float(value)), 'Invalid feature.')
        values[key] = float(value)
    c1, c2, c3 = (upstream[k] for k in ('c1_evidence', 'c2_diagnosis', 'c3_repair_package'))
    for key in ('batch_size', 'drift_detected', 'predicted_harm', 'interaction_suspected', 'detectors_fired'):
        add('c1_' + key, c1[key])
    lower, upper = c1['harm_interval']
    add('c1_harm_lower', lower); add('c1_harm_upper', upper); add('c1_harm_width', upper-lower)
    for section in ('detector_scores', 'prediction_stats'):
        for key in sorted(c1[section]): add('c1_' + section + '_' + key, c1[section][key])
    add('c1_affected_feature_count', len(c1['affected_features']))
    add('c1_warning_count', len(c1['warnings']))
    for key in ('slice_support', 'diagnosis_stability'): add('c2_' + key, c2[key])
    add('c2_ranked_feature_count', len(c2['affected_features_ranked']))
    for key in ('n_synthetic_rows', 'augmentation_ratio', 'fidelity_score', 'utility_score', 'target_coverage', 'prescription_confidence'):
        add('c3_' + key, c3[key])
    add('c3_repair_approved', c3['repair_package_status'] == 'APPROVED')
    add('c3_privacy_screen_pass', c3['privacy_status'] == 'PASS')
    for population in ('global', 'affected_slice', 'non_affected'):
        row = evidence['report']['comparisons'][population]
        add('c4_' + population + '_rows', row['rows'])
        for model in ('champion', 'candidate', 'candidate_minus_champion'):
            for metric in ('accuracy', 'macro_f1', 'auc'):
                value = row[model][metric]
                key = 'c4_' + population + '_' + model + '_' + metric
                if metric == 'auc': add(key + '_missing', value is None)
                add(key, 0. if metric == 'auc' and value is None else value)
    return dict(sorted(values.items()))


def verify_forecast(forecast, calibration, safety):
    require(calibration['alpha'] == .05 and calibration['target_columns'] == list(TARGETS)
            and calibration['nonempty_groups'] == 20 and calibration['rank'] == 20
            and calibration['fallback'] is False and calibration['auto_deploy_supported'] is True,
            'Unexpected frozen N2 calibration.')
    adjustment = numeric(calibration['adjustment'])
    require(adjustment >= 0., 'Negative adjustment.')
    require(set(safety['dimensions']) == set(TARGETS) and safety['auto_deploy_supported'] is True
            and safety['release_authorized'] is False, 'N3 scope differs.')
    statuses = []
    names = {t+'__'+q for t in TARGETS for q in ('lower', 'median', 'upper')}
    require(all(set(forecast[stage]) == names for stage in ('raw', 'processed', 'calibrated')),
            'Forecast column inventory differs.')
    for target in TARGETS:
        raw = [numeric(forecast['raw'][target+'__'+q]) for q in ('lower', 'median', 'upper')]
        # Original fixed postprocessing sorts and clips all quantile predictions.
        ordered = sorted(max(-1., min(1., v)) for v in raw)
        processed = [numeric(forecast['processed'][target+'__'+q]) for q in ('lower', 'median', 'upper')]
        require(processed == ordered, 'N1 postprocessing differs.')
        lower, median, upper = (numeric(forecast['calibrated'][target+'__'+q]) for q in ('lower', 'median', 'upper'))
        require(lower == max(-1., processed[0]-adjustment) and upper == min(1., processed[2]+adjustment)
                and median == processed[1], 'N1/N2 interval binding differs.')
        dim = safety['dimensions'][target]
        boundary = BOUNDARIES[target]
        status = 'SAFE' if lower >= boundary else 'UNSAFE' if upper < boundary else 'UNCERTAIN'
        require(dim['lower'] == lower and dim['upper'] == upper and dim['boundary'] == boundary
                and dim['status'] == status, 'N3 safety differs from frozen bounds.')
        statuses.append(status)
    signal = 'ESCALATE' if 'UNSAFE' in statuses else 'AUTO_DEPLOY' if all(s == 'SAFE' for s in statuses) else 'HUMAN_REVIEW'
    require(safety['release_signal'] == signal, 'N3 signal differs.')


def replay(case_id=CASE_ID, method='hybrid', family='adaptive', *, data_root=None):
    # Request errors remain distinct from artifact failures.
    if case_id != CASE_ID: raise KeyError(case_id)
    if method not in METHODS or family not in ('global', 'adaptive'):
        raise ValueError('Unknown revised method or policy family.')
    root = data_root if data_root is not None else os.environ.get('C4_DATA_ROOT')
    if not root:
        raise RevisedCaseUnavailable('Revised case evidence is unavailable. Configure C4_DATA_ROOT in the worker.')
    try:
        return _replay(Path(root), method, family)
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        raise RevisedCaseUnavailable('Revised case evidence is missing or failed verification. No revised decision was produced.') from exc


def _replay(root, method, family):
    bundle = root / 'c4_artifacts/forecasting_experiments/adult' / VERSION
    protocol, protocol_hash = document(bundle, 'validation_protocol')
    batch, batch_hash = document(bundle / 'outer-forecast-freezer-v1/completion-v1', 'forecast_batch')
    require(protocol_hash == PROTOCOL_SHA256 and protocol['primary_method'] == METHODS['hybrid']
            and batch_hash == BATCH_SHA256, 'Unexpected frozen protocol or batch.')
    records = batch['records']
    require(batch['cases'] == len(records) == 56 and len({r['scenario_id'] for r in records}) == 56
            and min(r['scenario_id'] for r in records) == SCENARIO_ID
            and batch['forecasts_frozen_before_future'] is True
            and batch['selected_policy_decisions_frozen_before_future'] is True
            and batch['future_evaluation_executed'] is False
            and batch['production_release_authorized'] is False, 'Invalid pre-future batch.')
    record = next(r for r in records if r['scenario_id'] == SCENARIO_ID)
    require(record['split_role'] == 'outer_test', 'Not an outer case.')
    run = root / 'c4_artifacts/benchmark_runs/adult' / VERSION / SCENARIO_ID
    freeze, freeze_hash = document(run / 'forecast_freeze-v1', 'freeze')
    snapshot, snapshot_hash = document(run / 'pre_release_freeze-v1', 'freeze')
    envelope, envelope_hash = document(run / 'pre_release_freeze-v1', 'upstream_simulation')
    evidence, evidence_hash = document(root / 'c4_artifacts/pre_release/adult' / SCENARIO_ID, 'evidence')
    metadata, metadata_hash = document(run / 'scoped_candidate_validation-v1', 'metadata')
    validation, validation_hash = document(run / 'scoped_candidate_validation-v1', 'validation')
    n2, n2_hash = document(bundle / 'n2-calibration-v1/completion-v1', 'calibration')
    policies, policy_hash = document(bundle / 'policy-selection-v1/completion-v1', 'selection')
    require(freeze_hash == record['forecast_freeze_sha256']
            and snapshot_hash == freeze['decision_freeze_sha256'] == record['decision_freeze_sha256']
            and snapshot['scenario'] == freeze['scenario'] == validation['scenario']
            and snapshot['scenario']['scenario_id'] == metadata['scenario_id'] == evidence['scenario_id'] == SCENARIO_ID
            and snapshot['scenario']['split_role'] == 'outer_test', 'Case identity binding differs.')
    require(n2_hash == freeze['N2_calibration_sha256'] == batch['N2_calibration_sha256'] == record['N2_calibration_sha256'] == policies['N2_calibration_sha256']
            and policy_hash == freeze['policy_selection_sha256'] == batch['policy_selection_sha256'] == record['policy_selection_sha256']
            and n2['model_bindings'] == policies['model_bindings'] == freeze['model_bindings'] == batch['model_bindings']
            and set(n2['methods']) == set(policies['methods']) == set(METHODS.values()), 'Model, N2 or policy lineage differs.')
    require(envelope_hash == snapshot['upstream_simulation_sha256']
            and evidence_hash == snapshot['pre_release_evidence_sha256'] == validation['pre_release_evidence_sha256']
            and metadata_hash == validation['candidate_metadata_sha256']
            and metadata['model_sha256'] == evidence['candidate_model_sha256']
            and metadata['champion_model_sha256'] == evidence['champion_model_sha256']
            and metadata['artifact_version'] == evidence['candidate_version']
            and metadata['repair_csv_sha256'] == evidence['repair_csv_sha256'], 'Candidate/pre-release binding differs.')
    for stem, expected in [('metadata', metadata_hash), ('validation', validation_hash)]:
        suffix = '/scoped_candidate_validation-v1/' + stem + '.json'
        links = [v for k, v in snapshot['dependencies'].items() if k.replace('\\', '/').endswith(suffix)]
        require(links == [expected], 'Candidate record not bound by snapshot.')
    require(snapshot['complete_decision_evidence_frozen'] is True
            and snapshot['future_evaluation_executed'] is False and snapshot['release_authorized'] is False
            and snapshot['production_package_approved'] is False
            and freeze['forecasts_frozen_before_future'] is True
            and freeze['selected_policy_decisions_frozen_before_future'] is True
            and freeze['future_evaluation_executed'] is False and freeze['production_release_authorized'] is False
            and envelope['scope'] == 'controlled_offline_simulation' and envelope['production_package_approved'] is False
            and evidence['evaluation_partition'] == 'release_validation' and evidence['slice_basis'] == 'c2_diagnosed_slice'
            and validation['candidate_reload_verified'] is True and validation['future_evaluation_executed'] is False,
            'Replay scope differs from pre-future evidence.')
    raw = {}
    for stem, state_key, record_key in [('features', 'features_sha256', 'feature_sha256'),
                                        ('forecasts', 'forecasts_sha256', 'forecasts_sha256'),
                                        ('controller_decisions', 'controller_decisions_sha256', 'controller_decisions_sha256')]:
        require(freeze[state_key] == record[record_key], 'Batch/case evidence link differs.')
        raw[stem] = json.loads(checked(run / 'forecast_freeze-v1' / (stem+'.json'), freeze[state_key]))
    feature = raw['features']
    values = feature_row(envelope['upstream'], evidence)
    require(feature['scenario_id'] == SCENARIO_ID and len(feature['columns']) == len(feature['values']) == 75
            and feature['columns'] == list(values) and feature['values'] == list(values.values())
            and feature['basis'] == 'pre_release_diagnosed_populations_only_no_future_targets', 'Feature reconstruction differs.')
    signals = raw['controller_decisions']
    up = envelope['upstream']; c3 = up['c3_repair_package']
    gates = dict(repair_approved=c3['repair_package_status'] == 'APPROVED', privacy_check=c3['privacy_status'] == 'PASS',
                 diagnosis_stability=up['c2_diagnosis']['diagnosis_stability'] >= .50,
                 schema_compatible=snapshot['runtime_checks']['schema_compatible'], rollback_ready=snapshot['runtime_checks']['rollback_ready'])
    require(set(signals['hard_gates']) == set(GATES) and signals['hard_gates'] == gates
            and all(type(v) is bool for v in gates.values())
            and signals['future_targets_used'] is False and signals['production_release_authorized'] is False,
            'Gate or decision scope differs.')
    method_id = METHODS[method]
    saved = signals['methods'][method_id]; forecast = raw['forecasts'][method_id]
    verify_forecast(forecast, n2['methods'][method_id], saved['N3'])
    entry = saved['policies'][family]; selected = policies['methods'][method_id][family]
    require(entry['selected_policy'] == selected['selected_policy']
            and entry['selection_status'] == selected['status']
            and entry['selected_policy']['kind'] == 'review_only', 'Selected policy differs.')
    result = entry['result']
    failed = [key for key in GATES if gates[key] is not True]
    expected = 'ESCALATE' if failed or saved['N3']['release_signal'] == 'ESCALATE' else 'HUMAN_REVIEW'
    require(result['decision'] == expected and result['release_authorized'] is False
            and (failed or result['n3'] == saved['N3']), 'Frozen policy result differs.')
    # Displayed context is a heuristic from the same pre-release vector.
    quality = .35*values['c2_diagnosis_stability'] + .25*max(0., 1.-10.*values['c1_harm_width']) + .25*values['c3_fidelity_score'] + .15*max(0., min(1., values['c3_utility_score']+.1))
    context = dict(quality=quality, context='LOW' if quality >= .75 else 'MEDIUM' if quality >= .50 else 'HIGH', quality_is_calibrated_probability=False)
    require(signals['evidence_context'] == context, 'Evidence context differs.')
    details = dict(scenario=snapshot['scenario'], champion_model_id=metadata['champion_version'], candidate_model_id=metadata['artifact_version'],
                   pre_release_recorded_at=snapshot['created_at_utc'], candidate_evaluation=evidence['report']['comparisons'],
                   upstream_evidence=up, source_declarations=envelope['source_declarations'], privacy_interpretation=envelope['privacy_interpretation'],
                   production_package_approved=False, rollback_scope=snapshot['runtime_checks']['rollback_scope'],
                   pre_release_evidence_sha256=evidence_hash, upstream_simulation_sha256=envelope_hash,
                   candidate_metadata_sha256=metadata_hash, displayed_evidence_reconstructed=True)
    return dict(mode='verified_revised_recorded_replay', upstream_source='simulated', evidence_verified=True,
                experiment_version=VERSION, inference_executed=False, candidate_retrained=False,
                replay_note='Saved pre-future hybrid/baseline predictions and selected-policy decisions; no new model inference, training or deployment.',
                controller_freeze_sha256=freeze_hash, controller_freeze_label='Revised forecast freeze SHA256',
                input_file_sha256=freeze['features_sha256'], hard_gates=gates, decision_time_features=values,
                case_details=details, n2_calibration=n2['methods'][method_id], candidate_evaluation=details['candidate_evaluation'],
                candidate_evaluation_note='Observed pre-release evaluation on diagnosed populations, separate from future targets.',
                provenance=dict(protocol_sha256=protocol_hash, forecast_batch_sha256=batch_hash, forecast_freeze_sha256=freeze_hash,
                                decision_freeze_sha256=snapshot_hash, N2_calibration_sha256=n2_hash, policy_selection_sha256=policy_hash,
                                candidate_metadata_sha256=metadata_hash),
                result={**result, 'n3':saved['N3'], 'evidence_id':SCENARIO_ID, 'method':method, 'method_id':method_id,
                        'family':family, 'failed_gates':failed, 'evidence_context':context,
                        'selected_policy_kind':entry['selected_policy']['kind'], 'selected_policy_id':entry['selected_policy']['policy_id'],
                        'N1_executed':False, 'N2_executed':False, 'N1_replayed':True, 'N2_replayed':True,
                        'n1_forecasts':forecast['processed'], 'n2_intervals':forecast['calibrated'],
                        'future_targets_used':False, 'production_release_authorized':False})
