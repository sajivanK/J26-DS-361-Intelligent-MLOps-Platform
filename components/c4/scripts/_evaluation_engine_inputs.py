"""Bind development evaluation to completed comparisons and frozen definitions."""
import io
import json
from pathlib import Path
import pandas as pd
from _common import C4_ROOT,checked_json,checked_bytes,require,sha
from _evaluation_protocol import VERSION,PROTOCOL_VERSION,verify_manifest,validate_definition
from c4.evaluation.comparison_decisions import METHOD_IDS

PREPARED_HASH='3b51935a3f5dbe3a359edabe113c872d433699e11bfa590f4ee747da3ff917eb'
PREPARATION_SPEC_HASH='5c16362b6abffac01e6409c5e90e68bd5267a332289730b564402beb0234ae55'
EVALUATION_HASH='e899365a05130a45a2f5c3ce513e99e1064605c5c98ecc985140fdc64eaab06e'
ENGINE_FILES=['scripts/_evaluation_engine_inputs.py','scripts/evaluate_development_comparisons.py',
              'scripts/freeze_comparison_evaluation_ready.py','src/c4/evaluation/pilot_evaluation_metrics.py',
              'src/c4/evaluation/pilot_group_statistics.py']


def engine_sources():return {p:sha((C4_ROOT/p).read_bytes()) for p in ENGINE_FILES}


def load_inputs(root):
    protocol_dir=root/'c4_artifacts/evaluation_protocols/adult'/VERSION/PROTOCOL_VERSION
    frozen,freeze_hash=checked_json(protocol_dir,'freeze')
    definition,definition_hash=checked_json(protocol_dir,'definition')
    require(freeze_hash==EVALUATION_HASH and definition_hash==frozen['definition_sha256'],'Evaluation definitions differ.')
    validate_definition(definition)
    for relative,digest in frozen['source_sha256'].items():checked_bytes(C4_ROOT/relative,digest)
    manifest_dir=root/'c4_artifacts/benchmark_protocols/adult'/VERSION
    base,base_hash=checked_json(manifest_dir,'freeze');manifest,manifest_hash=checked_json(manifest_dir,'scenario_manifest')
    require(base_hash==frozen['original_protocol_freeze_sha256'] and manifest_hash==base['scenario_manifest_sha256'],'Scenario protocol differs.')
    _,outer_ids=verify_manifest(manifest)
    runs=root/'c4_artifacts/benchmark_runs/adult'/VERSION
    require(not any((runs/sid).exists() for sid in outer_ids),'Complete development evaluation readiness before outer execution.')
    directory=root/'c4_artifacts/comparison_calibrations/adult'/VERSION/'comparison-calibration-v1/prepared-v1'
    md,digest=checked_json(directory)
    spec,spec_hash=checked_json(directory.parent,'specification')
    require(digest==PREPARED_HASH and spec_hash==PREPARATION_SPEC_HASH and md['specification_sha256']==spec_hash
            and md['evaluation_specification_sha256']==EVALUATION_HASH and md['method_ids']==list(METHOD_IDS), 'Completed comparison checkpoint differs.')
    for relative,value in spec['binding']['source_sha256'].items():checked_bytes(C4_ROOT/relative,value)
    for path,value in spec['binding']['source_links'].items():checked_bytes(path,value)
    for filename,value in md['files'].items():checked_bytes(directory/filename,value)
    runtime,_=checked_json(directory,'comparison_runtime')
    require(runtime['method_ids']==list(METHOD_IDS) and runtime['selected_policies']==checked_json(directory,'policy_selections')[0]['selected_policies'], 'Comparison runtime differs.')
    policy_dir=root/'c4_artifacts/meta_datasets/adult'/VERSION/'policy-all12-audit-v1'
    policy,p_hash=checked_json(policy_dir)
    require(p_hash==spec['binding']['policy_snapshot_metadata_sha256'],'Policy snapshot differs.')
    def table(path,filename,metadata):
        return pd.read_csv(io.BytesIO(checked_bytes(path/filename,metadata['files'][filename])),float_precision='round_trip')
    data={'features':table(policy_dir,'features.csv',policy).set_index('scenario_id'),
        'targets':table(policy_dir,'targets.csv',policy).set_index('scenario_id'),
        'index':table(policy_dir,'scenario_index.csv',policy),
        'decisions':table(directory,'development_decisions.csv',md),
        'forecasts':{v:table(directory,'policy_'+v+'_forecasts.csv',md).set_index('scenario_id') for v in ('ORIGINAL','A1','A2','A3')},
        'runtime':runtime,'groups':policy['groups'],'definition':definition,'manifest':manifest,
        'admission_cases':policy['admission_cases'],'blocked_cases':policy['blocked_cases']}
    require(data['features'].index.equals(data['targets'].index) and len(data['targets'])==44
            and len(data['groups'])==12 and data['index'].split_role.eq('policy_calibration').all(),'Policy inventory differs.')
    require(len(data['decisions'])==748 and not data['decisions'].duplicated(['comparison','scenario_id']).any()
            and set(data['decisions'].comparison)==set(METHOD_IDS),'Development decision inventory differs.')
    gates={}
    for sid in data['targets'].index:
        snap,snap_hash=checked_json(runs/sid/'pre_release_freeze-v1','freeze')
        signals,signal_hash=checked_json(runs/sid/'pre_release_freeze-v1','policy_forecasts')
        require(signal_hash==snap['policy_forecasts_sha256'],'Pre-release gate binding differs.')
        gates[sid]=signals['explicit_hard_gates']
    data['gates']=gates
    binding={'prepared_comparisons_metadata_sha256':digest,'preparation_specification_sha256':spec_hash,
        'evaluation_specification_sha256':freeze_hash,'policy_snapshot_metadata_sha256':p_hash,
        'controller_freeze_sha256':spec['binding']['controller_freeze_sha256'],'engine_source_sha256':engine_sources()}
    return data,binding
