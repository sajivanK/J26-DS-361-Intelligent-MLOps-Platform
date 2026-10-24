"""Select only admitted cases from the completed original training group."""
import io
from pathlib import Path
import numpy as np
import pandas as pd
from _common import require, checked_json, checked_bytes
from _training_group import corrupt_labels

GROUP = 'adult-ns123-r003'


def admitted_rows(group_report):
    require(group_report['group_id'] == GROUP and group_report['cases'] == 37,
            'The completed original 37-case training group is required.')
    records = group_report['records']
    require(len(records) == 37 and len({r['scenario_id'] for r in records}) == 37,
            'Group records are incomplete or duplicated.')
    require(all(r['group_id'] == GROUP and r['split_role'] == 'forecast_train' for r in records),
            'Group contains a forbidden scenario fold.')
    rows = [r for r in records if r['simulation_admission'] == 'READY_FOR_SCOPED_CANDIDATE_VALIDATION']
    require(all(r['mechanism'] == 'interaction' for r in rows),
            'This group pipeline supports the currently admitted interaction conditions only.')
    return sorted(rows, key=lambda r: r['scenario_id'])


def group_context(root, scenario_id):
    directory = root / 'c4_artifacts/benchmark_runs/adult/adult-simulation-pp1-v2/group_admissions' / GROUP / 'v1'
    report, checksum = checked_json(directory, 'group_admissions')
    matches = [r for r in admitted_rows(report) if r['scenario_id'] == scenario_id]
    require(len(matches) == 1, 'Requested case is not admitted in the fixed completed training group.')
    record = matches[0]
    # Resolve explicitly rather than relying on report-relative layout.
    run = root / 'c4_artifacts/benchmark_runs/adult/adult-simulation-pp1-v2' / scenario_id
    admission, admission_hash = checked_json(run, 'admission')
    require(admission_hash == record['admission_sha256'] and admission['scenario']['split_role'] == 'forecast_train',
            'Admission checksum or fold differs.')
    return report, checksum, record


def verify_corruption(root, directory, generation):
    """Check paired variant against its frozen base and retain dependency hashes."""
    links = {}
    variant = generation.get('corruption_variant')
    if not variant: return links
    base_dir = directory.parent / variant['base_artifact_version']
    base_md, base_hash = checked_json(base_dir)
    require(base_hash == variant['base_metadata_sha256'], 'Corruption base metadata differs.')
    base_bytes = checked_bytes(base_dir / 'synthetic_unvalidated.csv', variant['base_csv_sha256'])
    require(base_md['synthetic_csv_sha256'] == variant['base_csv_sha256'] and
            base_md['generator_sha256'] == generation['generator_sha256'], 'Paired generator/CSV link differs.')
    checked_bytes(base_dir / 'generator.pkl', base_md['generator_sha256'])
    mask_bytes = checked_bytes(directory / 'corruption_mask.csv', variant['mask_sha256'])
    actual_bytes = checked_bytes(directory / 'synthetic_unvalidated.csv', generation['synthetic_csv_sha256'])
    base = pd.read_csv(io.BytesIO(base_bytes), float_precision='round_trip')
    actual = pd.read_csv(io.BytesIO(actual_bytes), float_precision='round_trip')
    expected, expected_mask = corrupt_labels(base, fraction=variant['fraction'], seed=variant['seed'])
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    pd.testing.assert_frame_equal(pd.read_csv(io.BytesIO(mask_bytes)), expected_mask, check_exact=True)
    require(variant['generator_refitted'] is False and variant['quality_scores_reused'] is False,
            'Corrupted repair condition does not follow the frozen paired protocol.')
    links = {str(base_dir / 'metadata.json'): base_hash,
             str(base_dir / 'synthetic_unvalidated.csv'): variant['base_csv_sha256'],
             str(base_dir / 'generator.pkl'): base_md['generator_sha256'],
             str(directory / 'corruption_mask.csv'): variant['mask_sha256']}
    return links
