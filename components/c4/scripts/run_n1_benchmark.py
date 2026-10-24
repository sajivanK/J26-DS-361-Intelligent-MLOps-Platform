"""Freeze the N1 specification, then fit only the audited forecasting fold."""
import io
import json
import platform
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from _common import C4_ROOT, checked_bytes, checked_json, git_context, name, parser, require, resolve_data_root, save_json, sha, utc_now
from c4.forecasting.benchmark_n1 import BenchmarkN1Config, BenchmarkQuantileForecaster, TARGETS, forecast_metrics, grouped_folds, validate_tables

VERSION = 'adult-simulation-pp1-v2'


def environment():
    return {'python': platform.python_version(), **{p: version(p) for p in ('numpy', 'pandas', 'scikit-learn', 'scipy', 'joblib', 'lightgbm')}}


def inputs(root, config_path):
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    require(config['raw_quantiles'] == [.05, .5, .95] and config['hyperparameter_search'] is False,
            'Unexpected quantiles or parameter-search scope.')
    cfg = BenchmarkN1Config(**config['parameters'])
    directory = root / 'c4_artifacts/meta_datasets/adult' / VERSION / config['training_snapshot_version']
    metadata, digest = checked_json(directory)
    require(digest == config['training_snapshot_metadata_sha256'], 'Wrong audited training snapshot.')
    require(metadata['fold'] == 'forecast_train' and metadata['rows'] == 106 and len(metadata['groups']) == 32,
            'Unexpected training inventory.')
    frozen_dir = root / 'c4_artifacts/benchmark_protocols/adult' / VERSION
    frozen, freeze_digest = checked_json(frozen_dir, 'freeze')
    manifest, manifest_digest = checked_json(frozen_dir, 'scenario_manifest')
    require(freeze_digest == metadata['protocol_freeze_sha256'] and manifest_digest == frozen['scenario_manifest_sha256'],
            'Frozen protocol/manifest link differs.')
    protocol_bytes = checked_bytes(frozen_dir / 'protocol.json', frozen['protocol_sha256'])
    protocol = json.loads(protocol_bytes)
    require(protocol['N2']['raw_quantiles'] == config['raw_quantiles'], 'N1 quantiles differ from frozen N2 protocol.')
    groups = manifest['groups']
    require(len({g['group_id'] for g in groups}) == len(groups), 'Duplicate manifest groups.')
    training_groups = {g['group_id'] for g in groups if g['split_role'] == 'forecast_train'}
    require(training_groups == set(metadata['groups']) and len(training_groups) == 32, 'Training group inventory differs.')
    tables = {}
    for filename, expected in metadata['files'].items():
        contents = checked_bytes(directory / filename, expected)
        if filename.endswith('.csv'):
            tables[filename] = pd.read_csv(io.BytesIO(contents), float_precision='round_trip')
    features = tables['features.csv'].set_index('scenario_id')
    targets = tables['targets.csv'].set_index('scenario_id')
    index = tables['scenario_index.csv']
    validate_tables(features, targets, index)
    require(features.shape == (106, 75) and index.group_id.nunique() == 29, 'Training support differs.')
    require(features.columns.tolist() == metadata['feature_columns'] and set(index.group_id) <= training_groups,
            'Training feature schema or group assignment differs.')
    admission = tables['admission_table.csv']
    require(len(admission) == 1184 and admission.scenario_id.is_unique, 'Admission inventory differs.')
    require(pd.api.types.is_bool_dtype(admission.verified_target_row_available.dtype), 'Admission flags must be Boolean.')
    admitted = set(admission.loc[admission.verified_target_row_available, 'scenario_id'])
    require(admitted == set(features.index), 'Targets must correspond exactly to admitted cases.')
    # Recheck the audited dependency bytes; no outcomes are parsed or reevaluated here.
    for path, expected in metadata['source_links'].items():
        checked_bytes(Path(path), expected)
    folds = []
    for number, (train, valid) in enumerate(grouped_folds(index, cfg.cv_splits), 1):
        folds.append({'fold': number, 'train_ids': index.iloc[train].scenario_id.tolist(),
                      'validation_ids': index.iloc[valid].scenario_id.tolist(),
                      'train_groups': sorted(index.iloc[train].group_id.unique().tolist()),
                      'validation_groups': sorted(index.iloc[valid].group_id.unique().tolist())})
    selected = features.columns[features.nunique(dropna=False) > 1].tolist()
    require(len(selected) == 52, 'Varying feature count differs.')
    source_files = [Path(__file__), C4_ROOT / 'src/c4/forecasting/benchmark_n1.py', config_path]
    binding = {'configuration': config, 'configuration_sha256': sha(config_bytes),
               'training_metadata_sha256': digest, 'protocol_freeze_sha256': freeze_digest,
               'training_files': metadata['files'], 'raw_columns': features.columns.tolist(),
               'selected_columns_final_fit': selected,
               'constant_columns_final_fit': [c for c in features if c not in selected],
               'internal_folds': folds, 'environment': environment(),
               'code_sha256': {str(p.relative_to(C4_ROOT)): sha(p.read_bytes()) for p in source_files}}
    return cfg, features, targets, index, binding


def main():
    p = parser(__doc__)
    p.add_argument('--freeze-spec', action='store_true')
    p.add_argument('--fit', action='store_true')
    p.add_argument('--run-version', type=name, default='n1-fit-v1')
    args = p.parse_args()
    require(not (args.freeze_spec and args.fit), 'Freeze and fitting are separate invocations.')
    root = resolve_data_root(args.data_root)
    cfg, features, targets, index, binding = inputs(root, C4_ROOT / 'configs/n1_benchmark_config.json')
    base = root / 'c4_artifacts/forecasting/adult' / VERSION / binding['configuration']['specification_version']
    print('VERIFIED TRAINING ROWS:', len(features), 'NONEMPTY GROUPS:', index.group_id.nunique())
    print('FEATURES: 75 raw; 52 varying; 23 constants excluded from final fit')
    print('METHODS: Quantile LightGBM + regularized linear quantile regression; 9 models each')
    print('INTERNAL VALIDATION: 5 grouped folds; preprocessing refitted within each fold')
    print('CONFORMAL / POLICY / OUTER-TEST DATA: EXCLUDED')
    if args.freeze_spec:
        require(not base.exists(), 'Specification already exists; preserve the frozen version.')
        base.mkdir(parents=True, exist_ok=False)
        digest = save_json(base / 'specification.json', {'purpose': 'frozen_n1_training_specification',
            'created_at_utc': utc_now(), **git_context(), 'binding': binding,
            'model_fitting_executed': False, 'release_authorized': False})
        print('SPECIFICATION:', base / 'specification.json')
        print('SPECIFICATION SHA256:', digest)
        print('NO MODEL FITTING EXECUTED')
        return
    if not args.fit:
        print('PLAN ONLY. Next use --freeze-spec; then a separate invocation with --fit.')
        return
    frozen, spec_hash = checked_json(base, 'specification')
    require(frozen['binding'] == binding, 'Inputs, code, environment or configuration changed after specification freeze.')
    output = base / args.run_version
    require(not output.exists(), 'Run version exists; preserve it and inspect before recovery.')
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / 'claim.json', {'created_at_utc': utc_now(), **git_context(), 'specification_sha256': spec_hash})
    files = {}
    metrics = {}
    fold_records = []
    try:
        for method in ('lightgbm', 'linear'):
            chunks = []
            for number, (train, valid) in enumerate(grouped_folds(index, cfg.cv_splits), 1):
                print('FITTING:', method, 'GROUPED FOLD:', number, flush=True)
                model = BenchmarkQuantileForecaster(method, cfg).fit(features.iloc[train], targets.iloc[train], index.iloc[train])
                chunks.append(model.predict(features.iloc[valid]))
                fold_records.append({'method': method, 'fold': number,
                    'selected_columns': model.selected_columns, 'constant_columns': model.constant_columns})
            oof = pd.concat(chunks).loc[features.index]
            metrics[method] = forecast_metrics(targets, oof, index.group_id)
            csv_path = output / (method + '_grouped_oof.csv')
            csv_path.write_bytes(oof.to_csv(index=True, lineterminator='\n').encode())
            files[csv_path.name] = sha(csv_path.read_bytes())
            print('FITTING:', method, 'FINAL TRAINING MODEL', flush=True)
            final = BenchmarkQuantileForecaster(method, cfg).fit(features, targets, index)
            model_path = output / (method + '.joblib')
            joblib.dump(final, model_path)
            saved = model_path.read_bytes()
            files[model_path.name] = sha(saved)
            restored = joblib.load(io.BytesIO(saved))
            pd.testing.assert_frame_equal(final.predict(features), restored.predict(features), check_exact=True)
        files['metrics.json'] = save_json(output / 'metrics.json', {'evaluation': 'internal_grouped_oof_only', 'methods': metrics})
        files['fold_transforms.json'] = save_json(output / 'fold_transforms.json', {'folds': fold_records})
        lines = ['# C4 N1 research evidence', '',
            'Controlled simulated-upstream benchmark; final outer-test evaluation is pending.',
            '', '106 admitted scenarios from 29 nonempty groups; 32 training groups audited.',
            '1,184 admission cases retained; 1,078 blocked cases have no imputed targets.',
            '75 raw features; final-fit constant filtering selects 52. Inner folds refit filtering and scaling.',
            'Raw quantiles: 0.05, 0.50, 0.95. Predictions retain raw values and sorted/clipped values.',
            'Equal total fitting weight per nonempty replication group; no hyperparameter search.',
            'Shared source observations limit independence claims. No conformal coverage or production approval is asserted.',
            '', '| Method | Target | Equal-group MAE | RMSE | R2 | Median pinball |',
            '|---|---|---:|---:|---:|---:|']
        for method, report in metrics.items():
            for target, record in report.items():
                m = record['equal_group_weighted']
                lines.append(f'| {method} | {target} | {m["mae"]} | {m["rmse"]} | {m["r2"]} | {m["pinball"]["0.5"]} |')
        lines += ['', 'Full scenario-weighted and equal-group metrics, raw and processed pinball losses, and crossing counts are in metrics.json.',
                  f'Specification SHA256: `{spec_hash}`', '', 'N2 / N3 / policy / N4 / outer-test evaluation: pending.', '']
        note = output / 'N1_RESEARCH_EVIDENCE.md'
        note.write_bytes('\n'.join(lines).encode())
        files[note.name] = sha(note.read_bytes())
        digest = save_json(output / 'metadata.json', {'purpose': 'frozen_n1_benchmark_models',
            'created_at_utc': utc_now(), **git_context(), 'specification_sha256': spec_hash,
            'training_metadata_sha256': binding['training_metadata_sha256'], 'files': files,
            'source_scope': binding['configuration']['source_scope'], 'rows': 106, 'nonempty_groups': 29,
            'internal_grouped_validation_complete': True, 'model_reload_prediction_equality': True,
            'n2_calibrated': False, 'outer_test_evaluated': False, 'release_authorized': False})
        print('N1 MODELS / GROUPED OOF / RELOAD VERIFICATION: PASS')
        print('OUTPUT:', output)
        print('METADATA SHA256:', digest)
    except Exception as error:
        save_json(output / 'failure.json', {'created_at_utc': utc_now(), 'error_type': type(error).__name__,
                                          'message': str(error), 'preserve_partial_artifacts': True})
        raise


if __name__ == '__main__':
    main()
