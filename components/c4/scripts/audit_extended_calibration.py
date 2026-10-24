"""Audit all 20 extension groups and combine immutable evidence; no training or future reevaluation."""
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

from _common import C4_ROOT, checked_bytes, checked_json, git_context, name, parser, require, resolve_data_root, save_json, sha, utc_now
from _extension_group import VERSION, EXTENSION_HASH, load_extension, extension_groups, verify_n1
from _extension_pipeline import extension_support, select_admitted_cases, verify_freeze_dependencies
from _scenario_meta import TARGET_COLUMNS, feature_row, validated_targets
from c4.contracts.models import UpstreamInput

BATCH_HASH = '7466c98fb8220f48028bb167a88b5cee4dc0f8d9e74de6212951d8b1098d1950'


def combine_tables(features, targets, indices):
    require(features and len(features)==len(targets)==len(indices), 'Missing extension tables.')
    columns = features[0].columns.tolist()
    require(len(columns)==75 and not set(TARGET_COLUMNS)&set(columns), 'Invalid decision-time feature schema.')
    for f,t,i in zip(features,targets,indices):
        require(f.columns.tolist()==columns and t.columns.tolist()==list(TARGET_COLUMNS), 'Feature/target schema differs.')
        require(f.index.equals(t.index) and f.index.tolist()==i.scenario_id.tolist(), 'Feature/target/scenario order differs.')
        require(i.split_role.isin(['conformal_calibration','conformal_extension']).all(), 'Nonextension data cannot enter this snapshot.')
        require(np.isfinite(f.to_numpy(dtype=float)).all() and np.isfinite(t.to_numpy(dtype=float)).all(), 'Nonfinite input.')
    f,t,i = pd.concat(features),pd.concat(targets),pd.concat(indices,ignore_index=True)
    require(f.index.is_unique and t.index.is_unique and i.scenario_id.is_unique, 'Duplicate scenario IDs.')
    return f,t,i


def main():
    p=parser(__doc__)
    p.add_argument('--snapshot-version',type=name,default='conformal-plus-extension-audit-v2')
    p.add_argument('--save',action='store_true')
    args=p.parse_args()
    root=resolve_data_root(args.data_root)
    n1=verify_n1(root)
    frozen_dir=root/'c4_artifacts/benchmark_protocols/adult'/VERSION
    frozen,freeze_hash,protocol,manifest,manifest_hash,extension_hash=load_extension(root)
    split_manifest,split_hash=checked_json(root/'c4_artifacts/splits/adult'/protocol['source_versions']['split_manifest'],'manifest')
    require(split_hash==frozen['split_manifest_sha256'],'Adult partition manifest differs.')
    future_ids=split_manifest['partitions']['sealed_future']['source_row_ids']
    groups=extension_groups(manifest)
    runs=root/'c4_artifacts/benchmark_runs/adult'/VERSION
    meta_root=root/'c4_artifacts/meta_datasets/adult'/VERSION
    source_links={}
    def document(directory,stem='metadata'):
        payload,digest=checked_json(directory,stem)
        source_links[str(directory/(stem+'.json'))]=digest
        return payload,digest
    def table(directory,filename,metadata):
        contents=checked_bytes(directory/filename,metadata['files'][filename])
        source_links[str(directory/filename)]=sha(contents)
        return pd.read_csv(io.BytesIO(contents),float_precision='round_trip')
    batch,batch_hash=document(runs/'extension_candidate_batches/extension-candidate-all20-v1','batch_results')
    require(batch_hash==BATCH_HASH and batch['group_ids_in_fixed_order']==groups and batch['meta_rows']==86 and
            batch['blocked_cases']==654 and batch['n1_binding']==n1, 'Completed batch differs from reported checkpoint.')
    batch_records={r['group_id']:r for r in batch['records']}
    require(len(batch_records)==len(batch['records'])==20,'Batch group IDs differ.')
    features,targets,indices=[],[],[]
    admission_rows,group_rows=[],[]
    forecasts={method:[] for method in ('lightgbm','linear')}
    snapshots=[]
    for group in groups:
        group_dir=runs/'group_admissions'/group
        report,report_hash=document(group_dir/'v1','group_admissions')
        require(report['freeze_sha256']==freeze_hash and report['n1_binding']==n1, 'Admission protocol/N1 differs.')
        selected=select_admitted_cases(manifest,report,group)
        selected_ids={r['scenario_id'] for r in selected}
        record=batch_records[group]
        require(record['admission_sha256']==report_hash and record['admitted']==len(selected) and
                record['blocked']==37-len(selected), 'Batch admission accounting differs.')
        for raw in report['records']:
            admission,digest=document(runs/raw['scenario_id'],'admission')
            require(digest==raw['admission_sha256'] and admission['scenario']['group_id']==group and
                    admission['scenario']['split_role']=='conformal_extension' and
                    admission['simulation_admission']==raw['simulation_admission'], 'Raw admission differs.')
            admission_rows.append({**raw,'verified_target_row_available':raw['scenario_id'] in selected_ids})
        group_rows.append({'group_id':group,'cases':37,'rows':len(selected),'blocked':37-len(selected),
                           'status':'ALL_BLOCKED' if not selected else 'COMPLETE'})
        if not selected:
            require(record['meta_rows']==0 and record['status']=='ADMISSION_ONLY_ALL_BLOCKED', 'All-blocked group differs.')
            print('VERIFIED GROUP:',group,'META ROWS: 0 BLOCKED: 37; NO TARGETS IMPUTED')
            continue
        pipeline,pipeline_hash=document(group_dir/'extension-pipeline-v1','group_pipeline')
        require(pipeline_hash==record['pipeline_sha256'] and pipeline['completed_scenarios']==len(selected) and
                pipeline['group_admissions_sha256']==report_hash, 'Pipeline link/count differs.')
        directory=meta_root/(group+'-extension-v1')
        metadata,metadata_hash=document(directory)
        require(metadata_hash==record['meta_metadata_sha256'] and metadata['group_id']==group and
                metadata['fold']=='conformal_extension' and metadata['group_admissions_sha256']==report_hash and
                metadata['group_pipeline_sha256']==pipeline_hash, 'Group meta links differ.')
        for filename,expected in metadata['files'].items():
            checked_bytes(directory/filename,expected)
            source_links[str(directory/filename)]=expected
        f=table(directory,'features.csv',metadata).set_index('scenario_id')
        t=table(directory,'targets.csv',metadata).set_index('scenario_id')
        i=table(directory,'scenario_index.csv',metadata)
        require(set(f.index)==selected_ids and len(f)==metadata['rows']==len(selected) and
                i.group_id.eq(group).all(), 'Scenario/group membership differs.')
        for row in selected:
            sid=row['scenario_id']; run=runs/sid
            ix=i.loc[i.scenario_id.eq(sid)]
            require(len(ix)==1,'Scenario index is missing/duplicated.')
            ix=ix.iloc[0]
            snapshot,snapshot_hash=document(run/'pre_release_freeze-v1','freeze')
            verify_freeze_dependencies(snapshot); snapshots.append(snapshot)
            outcomes,outcome_hash=document(run/'future_research-v1','outcomes')
            envelope,envelope_hash=document(run/'pre_release_freeze-v1','upstream_simulation')
            evidence,evidence_hash=document(root/'c4_artifacts/pre_release/adult'/sid,'evidence')
            require(snapshot_hash==ix.decision_freeze_sha256==outcomes['decision_freeze_sha256'] and
                    outcome_hash==ix.outcomes_sha256 and envelope_hash==ix.upstream_simulation_sha256==snapshot['upstream_simulation_sha256'] and
                    evidence_hash==ix.pre_release_evidence_sha256==snapshot['pre_release_evidence_sha256'] and
                    ix.admission_sha256==row['admission_sha256'], 'Scenario artifact links differ.')
            require(snapshot['scenario']==outcomes['scenario'] and outcomes['scenario']['group_id']==group and
                    outcomes['scenario']['split_role']=='conformal_extension', 'Future scenario/fold differs.')
            expected_f=feature_row(UpstreamInput.model_validate(envelope['upstream']).model_dump(mode='json'),evidence)
            expected_t=validated_targets(outcomes)
            require(f.columns.tolist()==list(expected_f) and np.allclose(f.loc[sid].to_numpy(dtype=float),list(expected_f.values()),rtol=0,atol=1e-15),
                    'Features differ from original decision-time reconstruction.')
            require(t.columns.tolist()==list(TARGET_COLUMNS) and np.allclose(t.loc[sid].to_numpy(dtype=float),list(expected_t.values()),rtol=0,atol=1e-15),
                    'Targets differ from original outcomes.')
            future=run/'future_research-v1'
            for filename,expected in outcomes['files'].items():
                checked_bytes(future/filename,expected)
                source_links[str(future/filename)]=expected
            predicted=table(future,'future_predictions.csv',outcomes)
            membership=table(future,'future_membership.csv',outcomes)
            require(predicted.source_row_id.is_unique and predicted.source_row_id.tolist()==membership.source_row_id.tolist()==outcomes['source_row_ids']==future_ids,
                    'Future row identities differ.')
            require(pd.api.types.is_bool_dtype(membership.true_slice_member.dtype),'True membership must be Boolean.')
            masks=[np.ones(len(predicted),dtype=bool),membership.true_slice_member.to_numpy(),~membership.true_slice_member.to_numpy()]
            for target,mask in zip(TARGET_COLUMNS,masks):
                require(mask.any(),'Empty target population.')
                y=predicted.loc[mask,'target'].to_numpy()
                margin=float(np.mean(predicted.loc[mask,'candidate_prediction'].to_numpy()==y)-np.mean(predicted.loc[mask,'champion_prediction'].to_numpy()==y))
                require(abs(margin-expected_t[target])<=1e-15,'Paired future-target recomputation differs.')
            fixed,fixed_hash=document(run/'pre_release_freeze-v1','n1_forecasts')
            require(fixed_hash==snapshot['n1_forecasts_sha256'] and fixed['scenario_id']==sid and fixed['n1_binding']==n1 and
                    fixed['future_targets_included'] is False and fixed['N2_calibrated'] is False and
                    fixed['decision_time_features']==expected_f and fixed['feature_columns']==list(expected_f),
                    'N1 forecasts were not frozen from the same decision-time features.')
            for method in forecasts:
                forecasts[method].append({'scenario_id':sid,**fixed['forecasts'][method]})
        features.append(f);targets.append(t);indices.append(i)
        print('VERIFIED GROUP:',group,'META ROWS:',len(f),'BLOCKED:',37-len(f))
    f,t,index=combine_tables(features,targets,indices)
    require(f.shape==(86,75) and index.group_id.nunique()==20 and len(admission_rows)==740,'Combined inventory differs.')
    original_dir=meta_root/'conformal-all20-audit-v1'
    original_md,original_hash=document(original_dir)
    require(original_hash=='53e76636be10a25525e72ce1bab8b19473a3e861ba5f66b6a0e99a06f128c4ad'
            and original_md['n1_binding']==n1 and original_md['rows']==64 and original_md['nonempty_groups']==17,
            'Original audited calibration checkpoint differs.')
    checked_bytes(C4_ROOT/'scripts/audit_conformal_snapshot.py',original_md['source_code_sha256'])
    for path,expected in original_md['source_links'].items():
        checked_bytes(path,expected);source_links[path]=expected
    for filename,expected in original_md['files'].items():
        checked_bytes(original_dir/filename,expected);source_links[str(original_dir/filename)]=expected
    original_f=table(original_dir,'features.csv',original_md).set_index('scenario_id')
    original_t=table(original_dir,'targets.csv',original_md).set_index('scenario_id')
    original_i=table(original_dir,'scenario_index.csv',original_md)
    require(original_i.split_role.eq('conformal_calibration').all() and
            not set(original_i.group_id)&set(index.group_id), 'Original/extension calibration groups overlap.')
    extension_forecasts={method:pd.DataFrame(rows).set_index('scenario_id').loc[f.index] for method,rows in forecasts.items()}
    f,t,index=combine_tables([original_f,f],[original_t,t],[original_i,index])
    require(f.shape==(150,75) and index.group_id.nunique()==37 and
            index.groupby('group_id').split_role.nunique().eq(1).all(),'Combined extension inventory or group-fold assignment differs.')
    combined_forecasts={}
    for method,frame in extension_forecasts.items():
        previous=table(original_dir,method+'_frozen_forecasts.csv',original_md).set_index('scenario_id')
        require(previous.columns.tolist()==frame.columns.tolist(),'Frozen forecast schemas differ.')
        combined_forecasts[method]=pd.concat([previous,frame]).loc[f.index]
    previous_admissions=table(original_dir,'admission_table.csv',original_md).to_dict('records')
    previous_groups=table(original_dir,'group_summary.csv',original_md).to_dict('records')
    admission_rows=previous_admissions+admission_rows
    group_rows=previous_groups+group_rows
    groups=original_md['groups']+groups
    require(len(admission_rows)==1480 and len(groups)==40 and len(set(groups))==40,
            'Combined admission/group inventory differs.')
    n1_directory=root/'c4_artifacts/forecasting/adult'/VERSION/'adult-n1-training-v1'
    spec,_=checked_json(n1_directory,'specification')
    config=spec['binding']['configuration']
    training_directory=meta_root/config['training_snapshot_version']
    training_md,training_hash=document(training_directory)
    require(training_hash==spec['binding']['training_metadata_sha256'],'N1 training provenance differs.')
    training_index=table(training_directory,'scenario_index.csv',training_md)
    require(not set(index.scenario_id)&set(training_index.scenario_id) and not set(index.group_id)&set(training_index.group_id),
            'Forecast-training and extension groups/scenarios overlap.')
    require(f.columns.tolist()==spec['binding']['raw_columns'],'N1 raw feature schema differs.')
    forecast_frames=combined_forecasts
    print('COMBINED CALIBRATION ROWS: 150; ASSIGNED GROUPS: 40; NONEMPTY: 37; ALL BLOCKED: 3')
    print('ADMISSION RECORDS: 1480; BLOCKED: 1330; DECISION-TIME FEATURES: 75')
    print('RECONSTRUCTION / PAIRED TARGETS / FROZEN N1 FORECASTS / TRAINING DISJOINTNESS: PASS')
    print('ORIGINAL N2 PRESERVED; N2-v2 CALIBRATION: PENDING')
    if not args.save:
        print('PLAN ONLY. No models fitted, future observations reevaluated, or snapshot written.')
        return
    output=meta_root/args.snapshot_version
    require(not output.exists(),'Combined extension snapshot exists; preserve it.')
    require(verify_n1(root)==n1,'N1 changed during audit.')
    for snapshot in snapshots:verify_freeze_dependencies(snapshot)
    for path,expected in source_links.items():checked_bytes(path,expected)
    output.mkdir(parents=True,exist_ok=False)
    files={}
    tables={'features.csv':(f,True),'targets.csv':(t,True),'scenario_index.csv':(index,False),
            'admission_table.csv':(pd.DataFrame(admission_rows),False),'group_summary.csv':(pd.DataFrame(group_rows),False),
            **{method+'_frozen_forecasts.csv':(frame,True) for method,frame in forecast_frames.items()}}
    for filename,(frame,include_index) in tables.items():
        data=frame.to_csv(index=include_index,lineterminator='\n').encode()
        (output/filename).write_bytes(data);files[filename]=sha(data)
    note='\n'.join(['# Combined original and extended calibration audit','',
        'Original: 64 target rows, 17 nonempty groups, 3 all-blocked groups.',
        'Extension: 86 target rows, 20 nonempty groups. All 20 predefined groups completed.',
        'Combined: 150 target rows, 37 nonempty of 40 assigned groups, 1480 admissions, 1330 blocked.',
        'Each group remains in its original role. No training, policy or outer-test groups included.',
        'The extension was introduced after the original support limitation; this is not the original frozen experiment.',
        'No classifiers were reevaluated and N1 was not refitted. Original N2/N3 artifacts remain unchanged.',
        'Finite rank support alone does not establish useful widths, exchangeability or independent 95% coverage.',''])
    (output/'C4_EXTENDED_CALIBRATION_AUDIT.md').write_bytes(note.encode())
    files['C4_EXTENDED_CALIBRATION_AUDIT.md']=sha(note.encode())
    digest=save_json(output/'metadata.json',{'purpose':'verified_combined_extended_conformal_snapshot',
        'created_at_utc':utc_now(),**git_context(),'rows':150,'groups':groups,'nonempty_groups':37,'all_blocked_groups':3,
        'folds':['conformal_calibration','conformal_extension'],'original_rows':64,'extension_rows':86,
        'admission_cases':1480,'blocked_cases':1330,'feature_columns':f.columns.tolist(),'target_columns':list(TARGET_COLUMNS),
        'n1_binding':n1,'protocol_freeze_sha256':freeze_hash,'extension_freeze_sha256':EXTENSION_HASH,
        'original_snapshot_metadata_sha256':original_hash,'source_links':source_links,'files':files,
        'source_code_sha256':sha(Path(__file__).read_bytes()),'N1_refitted':False,'future_evaluations_repeated':False,
        'N2_calibrated':False,'release_authorized':False})
    print('SAVED EXTENSION SNAPSHOT:',output)
    print('METADATA SHA256:',digest)


if __name__=='__main__':main()
