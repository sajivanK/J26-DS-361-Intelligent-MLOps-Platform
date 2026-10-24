"""Verify saved training/pooled-conformal/policy checkpoints before comparison preparation."""
import io
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from _common import C4_ROOT,checked_json,checked_bytes,require,sha
from _controller_artifacts import load_controller
from _feature_ablation_inputs import prepare_inputs,EVALUATION_HASH
from _evaluation_protocol import VERSION,PROTOCOL_VERSION
from _policy_pipeline import verify_context_support
from c4.forecasting.benchmark_feature_ablation import VARIANTS,project_features
from c4.forecasting.benchmark_n1 import TARGETS
from c4.risk.benchmark_policy import evidence_context

ABLATION_SPEC_HASH='f1a46eeed02a0b7b7a693bab4d6e4a4035f759791ff07466be4845264e2fc27d'
ABLATION_RUN_HASH='ab1ef7b2cfebbe762a3037f0dca883f43ef33d168d23824cd829e95528073385'
CALIBRATION_SNAPSHOT_HASH='963e374608ac0b4090a2fe24b83d8f271055208a0a343878b2dcb107e4dc7613'
POLICY_SNAPSHOT_HASH='d29e2f073c6dfe35241ebd0835cddccff6be08bc9faba1255dd6a2a06e93c4fd'
SPEC_VERSION='comparison-calibration-v1'


def load_inputs(root):
    cfg,train_features,_,train_index,abinding=prepare_inputs(root)
    base=root/'c4_artifacts/comparison_models/adult'/VERSION/'n1-feature-ablations-v1'
    spec,spec_hash=checked_json(base,'specification')
    require(spec_hash==ABLATION_SPEC_HASH and spec['binding']==abinding,'A1-A3 frozen training specification differs.')
    md,run_hash=checked_json(base/'fit-v1')
    require(run_hash==ABLATION_RUN_HASH and md['specification_sha256']==spec_hash and md['rows']==106
            and md['variants']==list(VARIANTS) and md['model_reload_prediction_equality'] is True,'A1-A3 completed models differ.')
    for filename,digest in md['files'].items():checked_bytes(base/'fit-v1'/filename,digest)
    controller,binding,controller_hash=load_controller(root)
    models={'ORIGINAL':controller.models['lightgbm']}
    for variant in VARIANTS:
        models[variant]=joblib.load(io.BytesIO(checked_bytes(base/'fit-v1'/(variant+'.joblib'),md['files'][variant+'.joblib'])))
        model=models[variant]
        require(model.input_columns==abinding['variants'][variant]['input_columns']
                and model.selected_columns==abinding['variants'][variant]['selected_columns_final_fit']
                and model.method=='lightgbm' and model.config==cfg
                and model.training_scenario_ids==train_features.index.tolist()
                and model.training_group_ids==sorted(train_index.group_id.unique().tolist()),'Ablation model training provenance differs.')
    meta_root=root/'c4_artifacts/meta_datasets/adult'/VERSION
    tables={};source_links={}
    for role,name,expected,rows,groups in [('calibration','conformal-plus-extension-audit-v2',CALIBRATION_SNAPSHOT_HASH,150,37),
                                         ('policy','policy-all12-audit-v1',POLICY_SNAPSHOT_HASH,44,11)]:
        directory=meta_root/name;meta,digest=checked_json(directory)
        require(digest==expected and meta['rows']==rows and meta['nonempty_groups']==groups,'Saved development snapshot differs.')
        source_links[str(directory/'metadata.json')]=digest
        for path,value in meta['source_links'].items():checked_bytes(path,value);source_links[path]=value
        for filename,value in meta['files'].items():checked_bytes(directory/filename,value);source_links[str(directory/filename)]=value
        def frame(filename):
            return pd.read_csv(io.BytesIO(checked_bytes(directory/filename,meta['files'][filename])),float_precision='round_trip')
        features=frame('features.csv').set_index('scenario_id');targets=frame('targets.csv').set_index('scenario_id');index=frame('scenario_index.csv')
        require(features.index.is_unique and features.index.equals(targets.index) and features.index.tolist()==index.scenario_id.tolist()
                and index.group_id.nunique()==groups and targets.columns.tolist()==list(TARGETS)
                and features.columns.tolist()==binding['raw_columns'],'Snapshot schema/identity differs.')
        roles=['conformal_calibration','conformal_extension'] if role=='calibration' else ['policy_calibration']
        require(index.split_role.isin(roles).all() and index.groupby('group_id').split_role.nunique().eq(1).all(),'Snapshot fold differs.')
        require(np.isfinite(features.to_numpy(dtype=float)).all() and np.isfinite(targets.to_numpy(dtype=float)).all()
                and (np.abs(targets.to_numpy(dtype=float))<=1).all(),'Invalid development values.')
        tables[role]={'features':features,'targets':targets,'index':index,
                     'original_forecasts':frame('lightgbm_frozen_forecasts.csv').set_index('scenario_id')}
    for left,right in [('calibration','policy')]:
        a=tables[left]['index'];b=tables[right]['index']
        require(not set(a.scenario_id)&set(b.scenario_id) and not set(a.group_id)&set(b.group_id),'Calibration-policy overlap.')
    for table in tables.values():
        require(not set(table['index'].scenario_id)&set(train_index.scenario_id)
                and not set(table['index'].group_id)&set(train_index.group_id),'Forecast training-development overlap.')
    support,support_hash=verify_context_support(root)
    gates={}
    for sid in tables['policy']['features'].index:
        directory=root/'c4_artifacts/benchmark_runs/adult'/VERSION/sid/'pre_release_freeze-v1'
        frozen,freeze_hash=checked_json(directory,'freeze');signals,signal_hash=checked_json(directory,'policy_forecasts')
        require(signal_hash==frozen['policy_forecasts_sha256'] and signals['future_targets_included'] is False,
                'Policy pre-release gates/signals differ.')
        require(evidence_context(tables['policy']['features'].loc[sid].to_dict())==signals['evidence_context'],'Pre-release evidence context differs.')
        source_links[str(directory/'freeze.json')]=freeze_hash;source_links[str(directory/'policy_forecasts.json')]=signal_hash
        gates[sid]=signals['explicit_hard_gates']
    evaldir=root/'c4_artifacts/evaluation_protocols/adult'/VERSION/PROTOCOL_VERSION
    evaluation,evaluation_hash=checked_json(evaldir,'freeze');definition,definition_hash=checked_json(evaldir,'definition')
    require(evaluation_hash==EVALUATION_HASH and definition_hash==evaluation['definition_sha256'],'Evaluation specification differs.')
    pdir=root/'c4_artifacts/policy_protocols/adult'/VERSION/'policy-spec-v1'
    policy_definition,policy_definition_hash=checked_json(pdir,'definition')
    source_links[str(pdir/'definition.json')]=policy_definition_hash
    source_paths=[Path(__file__),C4_ROOT/'scripts/prepare_comparison_artifacts.py',
                  C4_ROOT/'src/c4/evaluation/comparison_calibration.py',C4_ROOT/'src/c4/evaluation/comparison_decisions.py']
    original=controller.calibrations['lightgbm']
    original_calibration={'comparison':'ORIGINAL','mode':'joint','active_targets':list(TARGETS),'joint_alpha':.05,
        'nonempty_groups':original.nonempty_groups,'rank':original.rank,'fallback':original.fallback,
        'auto_deploy_supported':original.auto_deploy_supported,'adjustments':dict.fromkeys(TARGETS,original.adjustment),
        'unit':'one max score per nonempty replication group','production_release_authorized':False}
    artifact_binding={'evaluation_specification_sha256':EVALUATION_HASH,'controller_freeze_sha256':controller_hash,
        'ablation_specification_sha256':spec_hash,'ablation_run_metadata_sha256':run_hash,
        'calibration_snapshot_metadata_sha256':CALIBRATION_SNAPSHOT_HASH,'policy_snapshot_metadata_sha256':POLICY_SNAPSHOT_HASH,
        'context_support_sha256':support_hash,'model_input_columns':{m:model.input_columns for m,model in models.items()},
        'original_calibration':original_calibration,'original_selected_policies':{
            'PROPOSED':binding['selected_policies']['lightgbm']['adaptive'],'B8':binding['selected_policies']['lightgbm']['global']},
        'unsupported_contexts':support['unsupported_contexts'],'policy_selection_rule':policy_definition['selection'],
        'source_sha256':{str(p.relative_to(C4_ROOT)):sha(p.read_bytes()) for p in source_paths},
        'source_links':source_links,'outer_test_accessed':False,'N1_fitting_executed':False}
    return models,tables,gates,artifact_binding


def fixed_predictions(models,features):
    predictions={}
    for variant,model in models.items():
        projected=features if variant=='ORIGINAL' else project_features(features,variant,features.columns)
        predictions[variant]=model.predict(projected)
    return predictions
