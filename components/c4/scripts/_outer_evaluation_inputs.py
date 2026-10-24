"""Read the pinned recovered outer snapshot without refitting or selecting anything."""
import io
import json
from pathlib import Path
import numpy as np
import pandas as pd
from _common import C4_ROOT, checked_json, checked_bytes, require, sha
from _outer_pipeline import (VERSION, inventory, runs, verified_phase, verify_freeze_dependencies,
                             verify_pre_release_registry)
from c4.evaluation.comparison_decisions import METHOD_IDS
from c4.evaluation.pilot_evaluation_metrics import validate_outcomes

SNAPSHOT_HASH = 'f7e7c41e1a7e54e509dfd7d70069359b73bbf3e1a44a08875b0b09313d8ff895'
EXECUTION_HASH = 'e88d5702380b872e66b5b07e8cd9179a3863b379ad49377bde193a71803ab08d'
REGISTRY_HASH = '8630e0ad7e39b2ca2b9b4921fe531f6964e1ef48f2dc14de9c1350a5629b2493'
PREPARED_HASH = '3b51935a3f5dbe3a359edabe113c872d433699e11bfa590f4ee747da3ff917eb'
REPORT_VERSION = 'outer-comparison-evaluation-v1'
SOURCE_FILES = ('scripts/_outer_evaluation_inputs.py','scripts/evaluate_outer_comparisons.py',
                'scripts/export_outer_result_plots.py')


def sources():
    return {p:sha((C4_ROOT/p).read_bytes()) for p in SOURCE_FILES}


def validate_inventory(data):
    features,targets,index,groups = (data[k] for k in ('features','targets','index','groups'))
    validate_outcomes(targets,index,groups)
    require(len(groups)==16 and len(targets)==56 and len(features.columns)==75 and
            features.index.equals(targets.index) and index.split_role.eq('outer_test').all() and
            index.group_id.nunique()==14 and features.columns.is_unique and
            np.isfinite(features.to_numpy(dtype=float)).all(), 'Wrong outer feature/fold/group inventory.')
    require(index.explicit_hard_gates_json.map(lambda s:isinstance(json.loads(s),dict)).all(), 'Missing saved hard gates.')
    decisions = data['decisions']
    require(len(decisions)==56*len(METHOD_IDS) and not decisions.duplicated(['comparison','scenario_id']).any() and
            set(decisions.comparison)==set(METHOD_IDS) and set(decisions.scenario_id)==set(targets.index) and
            decisions.decision.isin(['AUTO_DEPLOY','HUMAN_REVIEW','ESCALATE']).all(), 'Wrong outer method-decision inventory.')
    mapping = index.set_index('scenario_id').group_id.to_dict()
    require(all(mapping[sid]==gid for sid,gid in zip(decisions.scenario_id,decisions.group_id)), 'Decision group changed.')
    admissions = data['admissions'];summary=data['group_summary']
    require(len(admissions)==592 and admissions.scenario_id.is_unique and admissions.split_role.eq('outer_test').all() and
            set(admissions.group_id)==set(groups) and admissions.groupby('group_id').size().eq(37).all(),
            'All 592 original outer admissions must be retained.')
    mask=admissions.verified_target_row_available
    require(pd.api.types.is_bool_dtype(mask) and int(mask.sum())==56 and
            set(admissions.loc[mask,'scenario_id'])==set(targets.index) and
            admissions.loc[mask,'simulation_admission'].eq('READY_FOR_SCOPED_CANDIDATE_VALIDATION').all() and
            admissions.loc[~mask,'simulation_admission'].eq('BLOCKED_ESCALATE').all(),
            'Blocked/admitted target accounting differs.')
    require(summary.group_id.tolist()==groups and summary.target_rows.tolist()==
            [int(index.group_id.eq(g).sum()) for g in groups] and summary.blocked.sum()==536 and
            summary.all_blocked.sum()==2, 'All-blocked assigned group accounting differs.')
    for label,forecasts in data['forecasts'].items():
        require(forecasts.index.equals(targets.index), 'Forecast identities/order differ: '+label)


def load_inputs(root):
    root=Path(root)
    with verified_phase(root,'audit') as phase:
        require(phase['sha256']==EXECUTION_HASH,'Outer execution version differs.')
        directory=root/'c4_artifacts/meta_datasets/adult'/VERSION/'outer-all16-outcome-audit-recovered-v2'
        metadata,digest=checked_json(directory)
        require(digest==SNAPSHOT_HASH and metadata['purpose']=='verified_outer_snapshot' and
                metadata['outer_execution_freeze_sha256']==EXECUTION_HASH and
                metadata['outer_pre_release_registry_sha256']==REGISTRY_HASH and
                all(metadata[k] is False for k in ('N1_refitted','N2_refitted','policy_reselected',
                                                   'blocked_targets_imputed','production_release_authorized')),
                'Recovered snapshot binding/state differs.')
        for filename,expected in metadata['files'].items():checked_bytes(directory/filename,expected)
        for path,expected in metadata['source_links'].items():checked_bytes(path,expected)
        recovery=metadata['recovery'];original=Path(recovery['original_snapshot_directory'])
        _,original_hash=checked_json(original)
        require(original_hash==recovery['original_metadata_sha256'] and
                recovery['original_snapshot_modified'] is False and recovery['all_saved_tables_copied_byte_identically'] is True,
                'Snapshot recovery provenance differs.')
        for filename,expected in recovery['original_actual_file_sha256'].items():checked_bytes(original/filename,expected)
        manifest,groups,reports,selected=inventory(root)
        registry,registry_hash=verify_pre_release_registry(root)
        require(registry_hash==REGISTRY_HASH,'Pre-future decision registry differs.')
        def table(filename,indexed=False):
            frame=pd.read_csv(io.BytesIO(checked_bytes(directory/filename,metadata['files'][filename])),float_precision='round_trip')
            return frame.set_index('scenario_id') if indexed else frame
        prepared=root/'c4_artifacts/comparison_calibrations/adult'/VERSION/'comparison-calibration-v1/prepared-v1'
        md,md_hash=checked_json(prepared)
        runtime,runtime_hash=checked_json(prepared,'comparison_runtime')
        require(md_hash==PREPARED_HASH and runtime_hash==md['files']['comparison_runtime.json'] and
                runtime['method_ids']==list(METHOD_IDS), 'Prepared comparison runtime differs.')
        definition_dir=root/'c4_artifacts/evaluation_protocols/adult'/VERSION/'pp1-evaluation-spec-v1'
        definition,definition_hash=checked_json(definition_dir,'definition')
        data={'features':table('features.csv',True),'targets':table('targets.csv',True),
              'index':table('scenario_index.csv'),'decisions':table('decisions.csv'),
              'admissions':table('admissions.csv'),'group_summary':table('group_summary.csv'),
              'forecasts':{v:table('forecasts_'+v+'.csv',True) for v in ('ORIGINAL','A1','A2','A3','linear')},
              'groups':groups,'runtime':runtime,'definition':definition,'manifest':manifest,
              'admission_cases':592,'blocked_cases':536}
        validate_inventory(data)
        require(data['targets'].index.tolist()==[r['scenario_id'] for r in selected], 'Original admitted case order differs.')
        gates={}
        for sid in data['targets'].index:
            snapshot,freeze_hash=checked_json(runs(root)/sid/'pre_release_freeze-v1','freeze')
            verify_freeze_dependencies(snapshot)
            payload,payload_hash=checked_json(runs(root)/sid/'pre_release_freeze-v1','comparison_forecasts')
            original_n1,_=checked_json(runs(root)/sid/'pre_release_freeze-v1','n1_forecasts')
            outcome,outcome_hash=checked_json(runs(root)/sid/'future_research-v1','outcomes')
            index_row=data['index'].set_index('scenario_id').loc[sid]
            require(freeze_hash==registry['scenario_freeze_sha256'][sid]==index_row.decision_freeze_sha256 and
                    payload_hash==registry['comparison_forecast_sha256'][sid] and
                    payload['future_targets_included'] is False and payload['runtime']==runtime and
                    data['features'].loc[sid].to_dict()==payload['decision_time_features'] and
                    data['targets'].loc[sid].to_dict()==outcome['primary_targets'] and
                    outcome_hash==index_row.outcomes_sha256, 'Saved outer evidence/result binding differs.')
            gates[sid]=payload['explicit_hard_gates']
            require(json.loads(index_row.explicit_hard_gates_json)==gates[sid], 'Saved gate snapshot differs.')
            for variant in ('ORIGINAL','A1','A2','A3','linear'):
                expected=original_n1['forecasts']['linear'] if variant=='linear' else payload['forecasts'][variant]
                require(data['forecasts'][variant].loc[sid].to_dict()==expected,'Saved frozen forecasts differ.')
            for method in METHOD_IDS:
                saved=data['decisions'].loc[data['decisions'].scenario_id.eq(sid)&data['decisions'].comparison.eq(method)].iloc[0]
                require(saved.decision==payload['decisions'][method]['decision'] and
                        saved.reason==payload['decisions'][method]['reason'], 'Pre-future method decision differs.')
        data['gates']=gates
        binding={'recovered_snapshot_metadata_sha256':digest,'outer_execution_freeze_sha256':EXECUTION_HASH,
                 'outer_pre_release_registry_sha256':registry_hash,'prepared_comparisons_metadata_sha256':md_hash,
                 'evaluation_definition_sha256':definition_hash,'source_sha256':sources(),
                 'models_refitted':False,'policies_reselected':False,'thresholds_changed':False}
        return data,binding
