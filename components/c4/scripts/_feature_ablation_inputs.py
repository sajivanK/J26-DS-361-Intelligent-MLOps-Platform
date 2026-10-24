"""Bind A1-A3 to existing training data and pre-outer evaluation definitions."""
from pathlib import Path
from dataclasses import asdict
from _common import C4_ROOT, checked_json, checked_bytes, require, sha
from _evaluation_protocol import VERSION, PROTOCOL_VERSION, validate_definition, verify_manifest
from run_n1_benchmark import inputs
from c4.forecasting.benchmark_feature_ablation import VARIANTS, ablation_columns

EVALUATION_HASH = 'e899365a05130a45a2f5c3ce513e99e1064605c5c98ecc985140fdc64eaab06e'
SPEC_VERSION = 'n1-feature-ablations-v1'


def prepare_inputs(root):
    directory = root / 'c4_artifacts/evaluation_protocols/adult' / VERSION / PROTOCOL_VERSION
    frozen, evaluation_hash = checked_json(directory, 'freeze')
    require(evaluation_hash == EVALUATION_HASH and frozen['outer_execution_authorized'] is False, 'Wrong frozen evaluation definitions.')
    definition, definition_hash = checked_json(directory, 'definition')
    require(definition_hash == frozen['definition_sha256'], 'Evaluation definition checksum differs.')
    validate_definition(definition)
    for relative, digest in frozen['source_sha256'].items(): checked_bytes(C4_ROOT / relative, digest)
    base = root / 'c4_artifacts/benchmark_protocols/adult' / VERSION
    original, original_hash = checked_json(base, 'freeze')
    manifest, manifest_hash = checked_json(base, 'scenario_manifest')
    require(original_hash == frozen['original_protocol_freeze_sha256'] and
            manifest_hash == original['scenario_manifest_sha256'], 'Scenario protocol differs.')
    _, outer_ids = verify_manifest(manifest)
    runs = root / 'c4_artifacts/benchmark_runs/adult' / VERSION
    require(not any((runs / sid).exists() for sid in outer_ids), 'Prepare comparison models before outer execution.')
    cfg, features, targets, index, original_binding = inputs(root, C4_ROOT / 'configs/n1_benchmark_config.json')
    n1base = root / 'c4_artifacts/forecasting/adult' / VERSION / original_binding['configuration']['specification_version']
    n1spec, n1spec_hash = checked_json(n1base, 'specification')
    require(n1spec_hash == '437ae92c313020697a35e8b1372442d327df22c9189aa6392afaeb2e354d98c3'
            and n1spec['binding'] == original_binding, 'Original N1 inputs, code, environment or specification changed.')
    recipes = {r['id']: r for r in definition['ablations'][:3]}
    require(tuple(recipes) == VARIANTS, 'Frozen feature ablation inventory differs.')
    require(recipes['A1']['drop_rule'] == 'all columns beginning c1_'
            and recipes['A2']['drop_rule'] == 'exact column c2_diagnosis_stability'
            and recipes['A3']['drop_columns'] == ['c3_fidelity_score','c3_utility_score','c3_target_coverage','c3_prescription_confidence'],
            'Frozen feature removal rules differ.')
    columns = {}
    for variant in VARIANTS:
        kept, dropped = ablation_columns(features.columns, variant)
        varying = [c for c in kept if features[c].nunique(dropna=False) > 1]
        columns[variant] = {'input_columns': kept, 'removed_columns': dropped,
            'selected_columns_final_fit': varying, 'constant_columns_final_fit': [c for c in kept if c not in varying]}
    paths = [C4_ROOT / 'scripts/run_n1_feature_ablations.py', Path(__file__),
             C4_ROOT / 'src/c4/forecasting/benchmark_feature_ablation.py']
    binding = {'evaluation_specification_sha256': evaluation_hash, 'original_n1_specification_sha256': n1spec_hash,
        'training_metadata_sha256': original_binding['training_metadata_sha256'],
        'original_n1_binding': original_binding, 'parameters': asdict(cfg), 'method': 'lightgbm',
        'variants': columns, 'variant_scope': 'N1 predictors only; gates and context formula retained',
        'source_sha256': {str(p.relative_to(C4_ROOT)): sha(p.read_bytes()) for p in paths},
        'calibration_or_policy_data_used_for_fitting': False, 'outer_test_accessed': False}
    return cfg, features, targets, index, binding
