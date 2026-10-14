"""Recover a verified saved candidate after a native long-path reload failure; never retrain."""
import io
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from _common import (parser, resolve_data_root, checked_json, checked_bytes, require, sha,
                     save_json, utc_now, git_context)
from _general_training import group_context
from _admitted_training import verify_corruption
from _portable_model_io import load_candidate, load_champion, save_pre_release_evidence
from c4.models import load_candidate as original_load_candidate
from c4.contracts.models import C2Diagnosis
from c4.data import FEATURE_COLUMNS
from c4.data.split_manifest import load_adult_split_manifest, verify_clean_release_source
from save_role_previews import verify_preview

SCENARIO = 'adult-ns2026-r004-marginal_multi-high-broadened_slice-label_corrupted_repair'


def main():
    p = parser(__doc__)
    p.add_argument('--finish', action='store_true', help='Finish verified saved artifacts without fitting a model.')
    args = p.parse_args()
    root = resolve_data_root(args.data_root)
    report, report_hash, record = group_context(root, SCENARIO)
    run = root / 'c4_artifacts/benchmark_runs/adult/adult-simulation-pp1-v2' / SCENARIO
    directory = run / 'scoped_candidate_validation-v1'
    require(not (directory / 'validation.json').exists(), 'Candidate validation is already complete.')
    claim, claim_hash = checked_json(directory, 'claim')
    admission, admission_hash = checked_json(run, 'admission')
    require(claim['scenario_id'] == SCENARIO and claim['admission_sha256'] == admission_hash,
            'Original training claim differs.')
    from _common import C4_ROOT
    for relative, checksum in claim['source_sha256'].items(): checked_bytes(C4_ROOT / relative, checksum)
    frozen_dir = root / 'c4_artifacts/benchmark_protocols/adult/adult-simulation-pp1-v2'
    frozen, freeze_hash = checked_json(frozen_dir, 'freeze')
    require(claim['freeze_sha256'] == admission['freeze_sha256'] == report['freeze_sha256'] == freeze_hash,
            'Protocol links differ.')
    protocol = json.loads(checked_bytes(frozen_dir / 'protocol.json', frozen['protocol_sha256']))
    md_bytes = (directory / 'metadata.json').read_bytes()
    metadata = json.loads(md_bytes)
    metadata_hash = sha(md_bytes)
    require(metadata_hash == 'f82ee902633ac75d0c5c911d22f637f31b5f534931744bf9d5b99211bfd4abd0',
            'Metadata differs from the inspected failed attempt.')
    require(metadata['model_sha256'] == '3416519b9e2990b6418594d3217fa782a89d8df855213b145271448992ad9867',
            'Model differs from the inspected failed attempt.')
    sidecar = directory / 'metadata.sha256'
    if sidecar.exists(): require(sidecar.read_text().strip() == metadata_hash, 'Existing metadata checksum differs.')
    checked_bytes(directory / 'candidate.ubj', metadata['model_sha256'])
    require(metadata['role'] == 'candidate' and metadata['scope'] == 'controlled_offline_simulation' and
            metadata['scenario_id'] == SCENARIO and metadata['admission_sha256'] == admission_hash,
            'Saved candidate scope or provenance differs.')
    require(metadata['release_validation_used_for_training'] is False and metadata['sealed_future_used_for_training'] is False,
            'Saved candidate training roles differ.')
    package = root / 'c4_artifacts/offline_repair/adult' / admission['artifact_version']
    generation, generation_hash = checked_json(package)
    require(generation_hash == admission['generation_metadata_sha256'] == metadata['generation_metadata_sha256'],
            'Generation link differs.')
    corruption_links = verify_corruption(root, package, generation)
    checked_bytes(package / 'synthetic_unvalidated.csv', metadata['repair_csv_sha256'])
    require(metadata['repair_csv_sha256'] == generation['synthetic_csv_sha256'], 'Repair checksum differs.')
    champion_dir = root / 'c4_artifacts/champion/adult' / protocol['source_versions']['champion']
    champion, champion_md = load_champion(champion_dir)
    require(metadata['champion_model_sha256'] == champion_md['model_sha256'] == frozen['champion_model_sha256'],
            'Champion link differs.')
    require(metadata['training_config'] == champion_md['training_config'], 'Training configuration differs.')
    reference_bytes = checked_bytes(root / 'uci_adult_data/reference.csv', metadata['reference_csv_sha256'])
    require(metadata['reference_csv_sha256'] == champion_md['reference_csv_sha256'], 'Reference provenance differs.')
    reference = pd.read_csv(io.BytesIO(reference_bytes))
    require(metadata['reference_rows'] == len(reference) and metadata['repair_rows'] == generation['synthetic_rows'] and
            metadata['training_rows'] == len(reference) + generation['synthetic_rows'], 'Training row counts differ.')
    # load_candidate checks the existing model against metadata without needing its missing sidecar.
    candidate, _ = load_candidate(directory)
    sample = reference.loc[:, list(FEATURE_COLUMNS)].head(10)
    with tempfile.TemporaryDirectory(prefix='c4-recovery-reload-') as temporary:
        for filename in ('candidate.ubj', 'metadata.json'):
            (Path(temporary) / filename).write_bytes((directory / filename).read_bytes())
        restored, _ = original_load_candidate(temporary)
        require(np.array_equal(candidate.predict(sample), restored.predict(sample)) and
                np.allclose(candidate.predict_proba(sample), restored.predict_proba(sample)), 'Saved candidate reload differs.')
    manifest_dir = root / 'c4_artifacts/splits/adult' / protocol['source_versions']['split_manifest']
    checked_bytes(manifest_dir / 'manifest.json', frozen['split_manifest_sha256'])
    source = load_adult_split_manifest(manifest_dir, data_root=root)
    preview_dir = root / 'c4_artifacts/scenario_previews/adult' / admission['preview_id'] / 'release_validation'
    verify_preview(preview_dir, root, manifest_dir, 'release_validation')
    preview, preview_hash = checked_json(preview_dir)
    test_bytes = checked_bytes(root / 'uci_adult_data/test.csv', source['source_checksums']['test_csv_sha256'])
    ids = source['partitions']['release_validation']['source_row_ids']
    selected = set(ids)
    original_role = pd.read_csv(io.BytesIO(test_bytes), skiprows=lambda n:n>0 and n-1 not in selected)
    require(len(original_role) == len(ids), 'Source record layout differs.')
    original_role.index = sorted(selected); original_role = original_role.loc[ids]
    require(original_role.to_csv(index=True, lineterminator='\n').encode() ==
            checked_bytes(preview_dir / 'clean.csv', preview['files']['clean.csv']), 'Preview serialization differs.')
    clean = pd.read_csv(io.BytesIO(test_bytes)).loc[ids].copy()
    from _all_mechanism_transforms import canonical_release_frames
    from c4.scenarios.config import ScenarioConfig
    from c4.scenarios.slices import SliceRule
    row = admission['scenario']
    true = protocol['true_slice']
    rule = SliceRule(name=true['name'], feature=true['feature'], operator=true['operator'],
                     value=true['resolved_threshold'])
    config = ScenarioConfig(mechanism=row['mechanism'], magnitude=row['magnitude'], features=tuple(row['features']),
        interaction_pairs=tuple(tuple(p) for p in row['interaction_pairs']), seed=row['drift_seeds']['release_validation'])
    clean, evaluated = canonical_release_frames(original_role, clean,
        checked_bytes(preview_dir / 'clean.csv', preview['files']['clean.csv']),
        checked_bytes(preview_dir / 'transformed.csv', preview['files']['transformed.csv']),
        config=config, true_rule=rule)
    verify_clean_release_source(clean, source, data_root=root)
    evidence_dir = root / 'c4_artifacts/pre_release/adult' / SCENARIO
    require(not evidence_dir.exists(), 'Pre-release evidence already exists; do not overwrite it.')
    recovery_sources = {str(Path(__file__)): sha(Path(__file__).read_bytes()),
        str(Path(__file__).parent / '_portable_model_io.py'): sha((Path(__file__).parent / '_portable_model_io.py').read_bytes())}
    print('SAVED MODEL / METADATA / ORIGINAL CLAIM / INPUT LINKS: PASS')
    print('TRAINING ROWS:', metadata['training_rows'])
    print('RETRAINING REQUIRED: False')
    if not args.finish:
        print('PLAN ONLY. Use --finish to save pre-release evidence from the verified model without retraining.')
        return
    for path, expected in recovery_sources.items(): checked_bytes(path, expected)
    if not sidecar.exists():
        with sidecar.open('x', encoding='utf-8') as file: file.write(metadata_hash + '\n')
    artifact = save_pre_release_evidence(champion_directory=champion_dir, candidate_directory=directory,
        split_manifest_directory=manifest_dir, clean_release_validation=clean, release_validation=evaluated,
        diagnosis=C2Diagnosis.model_validate(admission['simulated_c2_diagnosis']), rule_feature_space='processed',
        scenario_id=SCENARIO, data_root=root)
    evidence, evidence_hash = checked_json(evidence_dir, 'evidence')
    require(evidence_hash == artifact.evidence_sha256, 'Saved evidence checksum differs.')
    result = {'purpose': 'scoped_simulation_candidate_and_pre_release_metrics', 'scenario': admission['scenario'],
        'created_at_utc': utc_now(), **git_context(), 'group_admissions_sha256': report_hash,
        'freeze_sha256': freeze_hash, 'admission_sha256': admission_hash,
        'candidate_metadata_sha256': metadata_hash, 'release_preview_metadata_sha256': preview_hash,
        'pre_release_evidence_sha256': evidence_hash, 'pre_release_directory': str(evidence_dir),
        'corruption_dependencies': corruption_links, 'candidate_training_executed': True,
        'candidate_reload_verified': True, 'production_package_approved': False, 'future_evaluation_executed': False,
        'complete_decision_evidence_frozen': False, 'recovery_without_retraining': True,
        'original_training_claim_sha256': claim_hash, 'original_model_and_metadata_unchanged': True,
        'recovery_script_sha256': sha(Path(__file__).read_bytes()),
        'recovery_source_sha256': recovery_sources,
        'model_io_mode': 'verified_bytearray_and_byte_identical_short_temporary_mirrors',
        'original_candidate_directory': str(directory), 'original_champion_directory': str(champion_dir)}
    checksum = save_json(directory / 'validation.json', result)
    print('CANDIDATE VALIDATION RECOVERED WITHOUT RETRAINING: PASS')
    print('REPORT:', directory / 'validation.json')
    print('REPORT SHA256:', checksum)
    checked_bytes(directory / 'candidate.ubj', metadata['model_sha256'])
    checked_bytes(directory / 'metadata.json', metadata_hash)
    print('ORIGINAL MODEL / METADATA UNCHANGED: PASS')
    print('FUTURE EVALUATION: False')


if __name__ == '__main__': main()
