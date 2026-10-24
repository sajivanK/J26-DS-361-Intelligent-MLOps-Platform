"""Save engineering-only fake C1/C2 evidence for the existing release preview."""
import importlib.util
import io
import json

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from _common import (
    parser, name, resolve_data_root, manifest_root, checked_json, checked_bytes,
    require, sha, save_json, git_context, utc_now, REPO_ROOT, C4_ROOT,
)
from save_role_previews import verify_preview
from c4.models import load_champion
from c4.data import FEATURE_COLUMNS
from c4.scenarios.slices import SliceRule
from c4.scenarios.feature_metadata import load_adult_feature_metadata
from c4.experiments.upstream_adapters import adapt_evidence_pair


def main():
    p = parser(__doc__)
    p.add_argument('--manifest-version', type=name, default='v1')
    p.add_argument('--champion-version', type=name, default='v1')
    p.add_argument('--evidence-version', type=name, default='mock-exact-v1')
    args = p.parse_args()
    root = resolve_data_root(args.data_root)
    # This smoke fixture is explicitly scoped to the existing marginal-multi pilot.
    preview = root / 'c4_artifacts' / 'scenario_previews' / 'adult' / 'development-001' / 'release_validation'
    output = root / 'c4_artifacts' / 'upstream_previews' / 'adult' / 'development-001' / args.evidence_version
    if output.exists():
        raise FileExistsError(f'Upstream preview already exists: {output}')
    verify_preview(preview, root, manifest_root(root, args.manifest_version), 'release_validation')
    metadata, metadata_hash = checked_json(preview)

    def read_frame(filename):
        return pd.read_csv(io.BytesIO(checked_bytes(
            preview / filename, metadata['files'][filename])), index_col=0)

    clean, transformed = read_frame('clean.csv'), read_frame('transformed.csv')
    champion, champion_metadata = load_champion(
        root / 'c4_artifacts' / 'champion' / 'adult' / args.champion_version)
    require(champion_metadata['feature_columns'] == list(FEATURE_COLUMNS), 'Champion feature schema differs.')
    y = clean['target'].to_numpy()
    before = champion.predict(clean.loc[:, list(FEATURE_COLUMNS)])
    after = champion.predict(transformed.loc[:, list(FEATURE_COLUMNS)])
    clean_accuracy = float(np.mean(before == y))
    transformed_accuracy = float(np.mean(after == y))
    drop_pp = 100.0 * (clean_accuracy - transformed_accuracy)
    feature_metadata = load_adult_feature_metadata()
    ks = {feature: float(ks_2samp(clean[feature], transformed[feature]).statistic)
          for feature in FEATURE_COLUMNS
          if feature_metadata.feature_types[feature] == 'continuous'}
    engineering_ground_truth = {
        'drift_type': 'marginal_multi', 'actual_accuracy_drop': drop_pp,
        'batch_size': len(clean), 'injected_features': ['education_num', 'hours_per_week'],
        'per_feature_ks': ks,
    }
    helper_path = REPO_ROOT / 'shared' / 'contracts' / 'c1_fake_output.py'
    helper_bytes = helper_path.read_bytes()
    spec = importlib.util.spec_from_file_location('c4_shared_fake_c1_smoke', helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    require(helper_path.read_bytes() == helper_bytes, 'Shared fake builder changed during loading.')
    c1 = helper.build_fake_c1_output(
        engineering_ground_truth, seed=42, model_id=f'adult-xgb-{args.champion_version}',
        reference_dataset_uri=str(root / 'uci_adult_data' / 'reference.csv'),
        current_dataset_uri=str(preview / 'transformed.csv'))
    # Preserve the helper's output, including fake interval/constants/extent.
    fixture_path = C4_ROOT / 'tests' / 'fixtures' / 'valid_upstream_input.json'
    fixture_bytes = fixture_path.read_bytes()
    c2 = json.loads(fixture_bytes)['c2_diagnosis']
    rule = SliceRule.model_validate(metadata['transformation']['true_slice'])
    operators = {'le': '<=', 'gt': '>', 'eq': '=='}
    c2.update({
        'affected_slice_rule': f'{rule.feature} {operators[rule.operator]} {rule.value!r}',
        'slice_support': float(rule.mask(clean).mean()),
        'root_cause_category': 'marginal', 'root_cause_feature': 'education_num',
        'affected_features_ranked': ['education_num', 'hours_per_week'],
        'diagnosis_version': args.evidence_version,
    })
    envelope = adapt_evidence_pair(c1, c2, clean, c1_origin='mock', c2_origin='mock')
    envelope.update({
        'purpose': 'engineering_mock_upstream_preview', 'created_at_utc': utc_now(), **git_context(),
        'preview_metadata_sha256': metadata_hash,
        'champion_model_sha256': champion_metadata['model_sha256'],
        'fake_c1_builder_sha256': sha(helper_bytes),
        'c2_fixture_sha256': sha(fixture_bytes),
        'feature_metadata_sha256': feature_metadata.source_sha256,
        'research_training_authorized': False, 'benchmark_execution': False,
        'field_origins': {
            'c1': 'shared answer-key fake builder; not measured C1 engine output',
            'paired_champion_drop_pp': 'measured clean versus transformed release-preview accuracy',
            'c1_interval': 'fake helper formula with fixed historical constants; not validated coverage',
            'c1_detector_prediction_fields': 'shared simulated values; KS inputs measured on this preview',
            'c2_rule': 'offline true-slice rule copied for engineering exact-diagnosis fixture',
            'c2_slice_support': 'computed from clean current-role membership',
            'c2_diagnosis_stability': 'unchanged fixture value; not measured diagnosis stability',
        },
        'engineering_measurements': {'clean_champion_accuracy': clean_accuracy,
                                     'transformed_champion_accuracy': transformed_accuracy,
                                     'paired_accuracy_drop_pp': drop_pp},
        'limitations': [
            'Engineering smoke test only; oracle-informed fake harm is not a research predictor.',
            'The helper uses historical P/N/Q constants, not calibrated values for this champion/split.',
            'Its combined extent mapping and interaction detection are preserved, not claimed correct.',
            'No C3 package, confidence, privacy PASS, approval, or final release decision is fabricated.',
            'No future outcomes are evaluated; research benchmark remains draft.',
        ],
    })
    checksum = save_json(output / 'evidence.json', envelope)
    print('C1 / C2 CONTRACT VALIDATION: PASS')
    print('C1 ENGINE:', envelope['c1_evidence']['engine_version'])
    print('SOURCE DECLARATIONS:', envelope['source_declarations'])
    print('DIAGNOSED ROWS:', envelope['diagnosed_rows'])
    print('NON-AFFECTED ROWS:', envelope['non_affected_rows'])
    print('COMPUTED SUPPORT:', envelope['computed_slice_support'])
    print('REPORT:', output / 'evidence.json')
    print('REPORT SHA256:', checksum)
    print('RESEARCH TRAINING AUTHORIZED: False')
    print('CANDIDATE TRAINING AUTHORIZED: False')


if __name__ == '__main__':
    main()
