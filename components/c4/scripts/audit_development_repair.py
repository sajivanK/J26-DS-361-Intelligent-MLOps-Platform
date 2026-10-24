"""Measure or verify internal development diagnostics; never approve a package."""
from pathlib import Path
import io
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import accuracy_score, f1_score
from _common import (parser, resolve_data_root, name, manifest_root, checked_json,
                     checked_bytes, require, sha, save_json, utc_now, git_context)
from generate_development_repair import verify_generation
from c4.models import ChampionConfig, load_champion, train_champion
from c4.data import FEATURE_COLUMNS
from c4.scenarios.feature_metadata import load_adult_feature_metadata


def main():
    p = parser(__doc__)
    p.add_argument('--preview-id', type=name, default='development-001')
    p.add_argument('--preparation-version', type=name, default='exact_slice-v1')
    p.add_argument('--manifest-version', type=name, default='v1')
    p.add_argument('--artifact-version', type=name, default='development-tvae-exact-v1')
    p.add_argument('--champion-version', type=name, default='v1')
    p.add_argument('--verify-only', action='store_true')
    args = p.parse_args()
    root = resolve_data_root(args.data_root)
    directory = root / 'c4_artifacts' / 'offline_repair' / 'adult' / args.artifact_version
    preparation = root / 'c4_artifacts' / 'repair_preparation' / 'adult' / args.preview_id / args.preparation_version
    manifest_directory = manifest_root(root, args.manifest_version)
    generation, generation_hash = verify_generation(directory, preparation, root, manifest_directory)
    prepared, preparation_hash = checked_json(preparation)
    if args.verify_only:
        report, checksum = checked_json(directory, 'diagnostics')
        require(report['generation_metadata_sha256'] == generation_hash, 'Diagnostic generation link differs.')
        require(report['preparation_metadata_sha256'] == preparation_hash, 'Diagnostic preparation link differs.')
        require(report['repair_package_approved'] is False, 'Development diagnostics cannot authorize repair.')
        checked_bytes(directory / 'utility_probe.ubj', report['probe_model_sha256'])
        print('DIAGNOSTICS CHECKSUM / INPUT LINKS: PASS')
        print('REPORT:', directory / 'diagnostics.json')
        print('REPORT SHA256:', checksum)
        print('MEAN UNIVARIATE FIDELITY:', report['mean_univariate_fidelity'])
        for population, result in report['internal_utility_comparisons'].items():
            print(population, result)
        return
    for filename in ('diagnostics.json', 'diagnostics.sha256', 'utility_probe.ubj'):
        if (directory / filename).exists():
            raise FileExistsError(f'Output exists: {directory / filename}; use --verify-only.')
    synthetic = pd.read_csv(io.BytesIO(checked_bytes(
        directory / 'synthetic_unvalidated.csv', generation['synthetic_csv_sha256'])))
    validation = pd.read_csv(io.BytesIO(checked_bytes(
        preparation / 'internal_validation.csv', prepared['files']['internal_validation.csv'])), index_col=0)
    membership = pd.read_csv(io.BytesIO(checked_bytes(
        preparation / 'validation_diagnosed_membership.csv',
        prepared['files']['validation_diagnosed_membership.csv'])), index_col=0)['diagnosed_slice_member']
    require(membership.index.equals(validation.index), 'Validation membership is misaligned.')
    require(pd.api.types.is_bool_dtype(membership.dtype) and not membership.isna().any(), 'Invalid membership values.')
    diagnosed_validation = validation.loc[membership]
    require(not diagnosed_validation.empty, 'No diagnosed internal-validation rows.')
    feature_metadata = load_adult_feature_metadata()
    require(feature_metadata.source_sha256 == generation['feature_metadata_sha256'], 'Feature metadata changed.')
    fidelity = {}
    for feature in FEATURE_COLUMNS:
        real, generated = diagnosed_validation[feature], synthetic[feature]
        if feature_metadata.feature_types[feature] == 'continuous':
            distance = float(ks_2samp(real, generated).statistic)
            method = '1_minus_ks'
        else:
            categories = pd.Index(pd.unique(pd.concat([real, generated])))
            real_probs = real.value_counts(normalize=True).reindex(categories, fill_value=0)
            generated_probs = generated.value_counts(normalize=True).reindex(categories, fill_value=0)
            distance = float(0.5 * np.abs(real_probs - generated_probs).sum())
            method = '1_minus_total_variation'
        fidelity[feature] = {'method': method, 'score': 1.0 - distance}
    mean_fidelity = float(np.mean([item['score'] for item in fidelity.values()]))
    champion, champion_metadata = load_champion(root / 'c4_artifacts' / 'champion' / 'adult' / args.champion_version)
    reference_bytes = checked_bytes(root / 'uci_adult_data' / 'reference.csv', champion_metadata['reference_csv_sha256'])
    reference = pd.read_csv(io.BytesIO(reference_bytes))
    print('Training offline utility probe...', flush=True)
    probe = train_champion(pd.concat([reference, synthetic], ignore_index=True),
                          ChampionConfig(**champion_metadata['training_config']))
    features = validation.loc[:, list(FEATURE_COLUMNS)]
    target = validation['target'].to_numpy()
    champion_predictions, probe_predictions = champion.predict(features), probe.predict(features)
    masks = {'internal_global': np.ones(len(validation), dtype=bool),
             'internal_diagnosed_slice': membership.to_numpy(dtype=bool),
             'internal_non_affected': ~membership.to_numpy(dtype=bool)}
    comparisons = {}
    for population, mask in masks.items():
        if not mask.any():
            comparisons[population] = {'rows': 0, 'metrics': None}
            continue
        before = float(accuracy_score(target[mask], champion_predictions[mask]))
        after = float(accuracy_score(target[mask], probe_predictions[mask]))
        comparisons[population] = {
            'rows': int(mask.sum()), 'champion_accuracy': before, 'probe_accuracy': after,
            'accuracy_margin': after - before,
            'champion_macro_f1': float(f1_score(target[mask], champion_predictions[mask], labels=[0, 1], average='macro', zero_division=0)),
            'probe_macro_f1': float(f1_score(target[mask], probe_predictions[mask], labels=[0, 1], average='macro', zero_division=0)),
        }
    probe_path = directory / 'utility_probe.ubj'
    probe.save_model(probe_path)
    report = {
        'purpose': 'development_repair_diagnostics', 'created_at_utc': utc_now(), **git_context(),
        'generation_metadata_sha256': generation_hash, 'preparation_metadata_sha256': preparation_hash,
        'champion_model_sha256': champion_metadata['model_sha256'], 'probe_model_sha256': sha(probe_path.read_bytes()),
        'probe_training_config': champion_metadata['training_config'], 'reference_rows': len(reference),
        'synthetic_rows': len(synthetic), 'diagnosed_validation_rows': len(diagnosed_validation),
        'mean_univariate_fidelity': mean_fidelity, 'per_feature_fidelity': fidelity,
        'internal_utility_comparisons': comparisons, 'privacy_status': 'NOT_EVALUATED',
        'repair_package_approved': False, 'benchmark_execution': False,
        'limitations': ['Development experiment; benchmark design remains draft.',
                        'Univariate fidelity does not establish joint or conditional fidelity.',
                        'Internal utility is not release-validation or future performance.',
                        'No privacy pass or C3 approval is inferred.'],
    }
    checksum = save_json(directory / 'diagnostics.json', report)
    print('MEAN UNIVARIATE FIDELITY:', mean_fidelity)
    for population, result in comparisons.items():
        print(population, result)
    print('REPORT:', directory / 'diagnostics.json')
    print('REPORT SHA256:', checksum)
    print('PRIVACY: NOT_EVALUATED')
    print('REPAIR PACKAGE APPROVED: False')


if __name__ == '__main__':
    main()
