"""Conformal-fold selection and frozen-N1 verification; no model fitting."""
from pathlib import Path

from _common import C4_ROOT, checked_bytes, checked_json, require
from _training_group import corrupt_labels

VERSION = 'adult-simulation-pp1-v2'
FOLD = 'conformal_calibration'
N1_SPEC_HASH = '437ae92c313020697a35e8b1372442d327df22c9189aa6392afaeb2e354d98c3'
N1_RUN_HASH = 'c9bcedd6f74c47d856eead5a3dcfc5edb37605312318cebc695cf12fe0d05545'


def verify_n1(root):
    directory = Path(root) / 'c4_artifacts/forecasting/adult' / VERSION / 'adult-n1-training-v1'
    spec, spec_hash = checked_json(directory, 'specification')
    run, run_hash = checked_json(directory / 'n1-fit-v1')
    require(spec_hash == N1_SPEC_HASH and run_hash == N1_RUN_HASH,
            'N1 specification or fitted-model version differs from the verified checkpoint.')
    require(run['specification_sha256'] == spec_hash and run['n2_calibrated'] is False and
            run['outer_test_evaluated'] is False and run['release_authorized'] is False,
            'N1 stage/provenance differs.')
    require(run['internal_grouped_validation_complete'] is True and
            run['model_reload_prediction_equality'] is True, 'N1 completion evidence differs.')
    binding = spec['binding']
    require(run['training_metadata_sha256'] == binding['training_metadata_sha256'], 'N1 training snapshot link differs.')
    for relative, expected in binding['code_sha256'].items():
        checked_bytes(C4_ROOT / relative, expected)
    for filename, expected in run['files'].items():
        checked_bytes(directory / 'n1-fit-v1' / filename, expected)
    return {'specification_sha256': spec_hash, 'run_metadata_sha256': run_hash,
            'protocol_freeze_sha256': binding['protocol_freeze_sha256']}


def conformal_groups(manifest):
    groups = manifest['groups']
    mapping = {g['group_id']: g for g in groups}
    require(len(mapping) == len(groups), 'Duplicate group IDs in frozen manifest.')
    selected = [g['group_id'] for g in groups if g['split_role'] == FOLD]
    require(len(selected) == 20, 'Exactly 20 frozen conformal groups are required.')
    return selected  # Preserve frozen manifest order; never sort by results.


def conformal_group_cases(manifest, group_id):
    require(group_id in conformal_groups(manifest), 'Only a frozen conformal-calibration group is allowed.')
    rows = [r for r in manifest['scenarios'] if r['group_id'] == group_id]
    require(len(rows) == 37 and all(r['split_role'] == FOLD for r in rows),
            'Exactly 37 conformal cases in the requested group are required.')
    require(len({r['scenario_id'] for r in rows}) == 37, 'Duplicate scenario IDs.')
    require(sum(bool(r['is_no_drift_control']) for r in rows) == 1, 'One no-drift control is required.')
    drifted = [r for r in rows if not r['is_no_drift_control']]
    expected = {(m, s, d, q) for m in ('marginal_multi','interaction','combined') for s in (.2,.5,.8)
                for d in ('exact_slice','broadened_slice') for q in ('targeted_repair','label_corrupted_repair')}
    actual = {(r['mechanism'],r['magnitude'],r['diagnosis_condition'],r['repair_condition']) for r in drifted}
    require(actual == expected, 'Conformal group differs from the frozen factorial design.')
    control = next(r for r in rows if r['is_no_drift_control'])
    require(control['mechanism'] == 'none' and control['magnitude'] == 0 and
            control['diagnosis_condition'] == 'exact_slice' and control['repair_condition'] == 'targeted_repair',
            'No-drift control differs from frozen design.')
    return sorted(rows, key=lambda r:(r['shared_base_package_id'], r['repair_condition'] != 'targeted_repair', r['scenario_id']))


def batch_groups(manifest, number):
    require(type(number) is int and 1 <= number <= 5, 'Batch number must be 1 through 5.')
    return conformal_groups(manifest)[(number-1)*4:number*4]
