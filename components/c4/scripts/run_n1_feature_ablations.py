"""Fit A1-A3 with frozen N1 settings on the original 106 training scenarios only."""
import io
import json
from pathlib import Path
import joblib
import pandas as pd
from _common import parser, resolve_data_root, require, name, checked_json, checked_bytes, save_json, sha, utc_now, git_context
from _feature_ablation_inputs import VERSION, SPEC_VERSION, prepare_inputs
from c4.forecasting.benchmark_feature_ablation import VARIANTS, project_features
from c4.forecasting.benchmark_n1 import BenchmarkQuantileForecaster, forecast_metrics, grouped_folds


def main():
    p = parser(__doc__)
    modes = p.add_mutually_exclusive_group()
    modes.add_argument('--freeze-spec', action='store_true')
    modes.add_argument('--fit', action='store_true')
    p.add_argument('--run-version', type=name, default='fit-v1')
    args = p.parse_args(); root = resolve_data_root(args.data_root)
    cfg, features, targets, index, binding = prepare_inputs(root)
    print('FROZEN EVALUATION SPECIFICATION / ORIGINAL N1 TRAINING BINDINGS: PASS')
    print('TRAINING: 106 ROWS / 29 NONEMPTY GROUPS; ORIGINAL 32 ASSIGNED GROUPS RETAINED')
    print('VARIANTS: A1 direct C1 predictors; A2 diagnosis-stability predictor; A3 four C3 quality predictors')
    for variant, columns in binding['variants'].items():
        print(variant, 'REMOVED:',len(columns['removed_columns']), 'INPUTS:',len(columns['input_columns']),
              'FINAL VARYING:',len(columns['selected_columns_final_fit']))
    print('FITTING: LIGHTGBM ONLY; SAME 5 GROUPED FOLDS / PARAMETERS / GROUP WEIGHTS')
    print('NO TVAE / CANDIDATE RETRAINING / CALIBRATION-POLICY FITTING / OUTER ACCESS')
    base = root / 'c4_artifacts/comparison_models/adult' / VERSION / SPEC_VERSION
    if args.freeze_spec:
        require(not base.exists(), 'Ablation specification exists; preserve it.')
        git = git_context(); require(not git['working_tree_dirty'], 'Commit the ablation implementation before freezing.')
        digest = save_json(base / 'specification.json', {'purpose':'frozen_n1_feature_ablation_specification',
            'created_at_utc':utc_now(), **git, 'binding':binding, 'model_fitting_executed':False, 'release_authorized':False})
        print('SPECIFICATION:',base / 'specification.json'); print('SPECIFICATION SHA256:',digest)
        print('NO FITTING EXECUTED; next use --fit in a separate invocation.'); return
    if not args.fit:
        print('PLAN ONLY. Commit/push, then --freeze-spec, then --fit.'); return
    frozen, spec_hash = checked_json(base, 'specification')
    require(frozen['binding'] == binding, 'Ablation inputs/source/environment changed after freeze.')
    output = base / args.run_version
    require(not output.exists(), 'Ablation run exists; preserve partial artifacts and diagnose before a fresh attempt.')
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / 'claim.json', {'purpose':'A1_A2_A3_training_only_refits','created_at_utc':utc_now(), **git_context(),
        'specification_sha256':spec_hash,'outcome_based_variant_selection':False,'original_N1_refitted':False})
    files = {}; metrics = {}; folds = []
    try:
        for variant in VARIANTS:
            projected = project_features(features, variant, features.columns)
            chunks = []
            for number, (train, valid) in enumerate(grouped_folds(index, cfg.cv_splits), 1):
                print('FITTING:',variant,'GROUPED FOLD:',number,flush=True)
                model = BenchmarkQuantileForecaster('lightgbm', cfg).fit(projected.iloc[train],targets.iloc[train],index.iloc[train])
                chunks.append(model.predict(projected.iloc[valid]))
                folds.append({'variant':variant,'fold':number,'selected_columns':model.selected_columns,
                    'constant_columns':model.constant_columns,'train_groups':sorted(index.iloc[train].group_id.unique().tolist()),
                    'validation_groups':sorted(index.iloc[valid].group_id.unique().tolist())})
            oof = pd.concat(chunks).loc[features.index]
            metrics[variant] = forecast_metrics(targets,oof,index.group_id)
            contents = oof.to_csv(index=True,lineterminator='\n').encode()
            filename = variant+'_grouped_oof.csv'; (output / filename).write_bytes(contents); files[filename]=sha(contents)
            print('FITTING:',variant,'FINAL TRAINING MODEL',flush=True)
            final = BenchmarkQuantileForecaster('lightgbm',cfg).fit(projected,targets,index)
            expected = binding['variants'][variant]
            require(final.input_columns == expected['input_columns'] and final.selected_columns == expected['selected_columns_final_fit'],
                    'Ablation feature filtering differs.')
            filename = variant+'.joblib'; joblib.dump(final,output / filename)
            saved = (output / filename).read_bytes(); files[filename]=sha(saved)
            restored = joblib.load(io.BytesIO(saved))
            pd.testing.assert_frame_equal(final.predict(projected),restored.predict(projected),check_exact=True)
        files['metrics.json']=save_json(output / 'metrics.json', {'evaluation':'internal_grouped_OOF_training_only',
            'method':'lightgbm','variants':metrics,'not_outer_test_evidence':True})
        files['fold_transforms.json']=save_json(output / 'fold_transforms.json',{'folds':folds})
        files['feature_projections.json']=save_json(output / 'feature_projections.json',binding['variants'])
        # Recheck immutable sources, training inputs and environment before saving the completed marker.
        require(prepare_inputs(root)[-1] == binding, 'Inputs changed during ablation fitting.')
        note='\n'.join(['# A1-A3 forecasting feature ablations','',
            '106 forecast-training scenarios, 29 nonempty of 32 assigned training groups.',
            'Three new LightGBM forecasters; original N1 models unchanged. Same hyperparameters and grouped folds.',
            'A1 drops direct C1 predictors; A2 drops diagnosis stability; A3 drops four C3 quality predictors.',
            'Hard gates, context definition and upstream construction are retained: these are forecasting feature ablations.',
            'No conformal/policy/outer rows used to fit models or filter constants.',
            'Each variant still requires its own N2 and policy calibration before outer evaluation.',
            'Full internal OOF metrics are in metrics.json; no outer-test or deployment benefit established.',''])
        (output / 'C4_FEATURE_ABLATION_EVIDENCE.md').write_bytes(note.encode()); files['C4_FEATURE_ABLATION_EVIDENCE.md']=sha(note.encode())
        digest=save_json(output / 'metadata.json', {'purpose':'completed_N1_feature_ablation_models',
            'created_at_utc':utc_now(), **git_context(),'specification_sha256':spec_hash,
            'evaluation_specification_sha256':binding['evaluation_specification_sha256'],
            'original_n1_specification_sha256':binding['original_n1_specification_sha256'],
            'training_metadata_sha256':binding['training_metadata_sha256'],'variants':list(VARIANTS),
            'rows':106,'nonempty_groups':29,'method':'lightgbm','files':files,'model_reload_prediction_equality':True,
            'original_N1_refitted':False,'N2_calibrated':False,'policy_calibrated':False,'outer_test_evaluated':False,
            'release_authorized':False})
        for filename,expected in files.items():checked_bytes(output / filename,expected)
        print('A1-A3 MODELS / GROUPED OOF / RELOAD EQUALITY / CHECKSUMS: PASS')
        print('OUTPUT:',output);print('METADATA SHA256:',digest)
        print('NEXT: variant N2 and policy calibration; outer groups remain untouched.')
    except Exception as error:
        save_json(output / 'failure.json',{'created_at_utc':utc_now(),'exception_type':type(error).__name__,
            'message':str(error),'preserve_partial_artifacts':True,'automatic_retry':False})
        raise


if __name__=='__main__':main()
