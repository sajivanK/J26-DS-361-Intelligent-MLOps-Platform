"""Complete the fixed policy fold in two phases: all pre-release freezes, then futures."""
import subprocess
import sys
from pathlib import Path
from _common import C4_ROOT,checked_json,checked_bytes,git_context,name,parser,require,resolve_data_root,save_json,sha,utc_now
from _policy_pipeline import VERSION,POLICY_SPEC_HASH,policy_inventory,group_context,verify_freeze_dependencies,verify_context_support,verify_policy_spec
from run_policy_group_admissions import existing_record
from _policy_group import policy_group_cases


def main():
    p=parser(__doc__)
    p.add_argument('--all-groups',action='store_true',required=True)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--batch-version',type=name,default='policy-candidate-all12-v1')
    args=p.parse_args();root=resolve_data_root(args.data_root)
    manifest,groups,reports,selected=policy_inventory(root)
    runs=root/'c4_artifacts/benchmark_runs/adult'/VERSION
    # Verify the complete admission record set against its saved repair artifacts.
    for gid in groups:
        report=reports[gid]['report']
        planned=policy_group_cases(manifest,gid)
        actual=[existing_record(root,row,report['freeze_sha256']) for row in planned]
        require(all(r is not None for r in actual) and actual==report['records'],'Policy repair/admission artifacts differ.')
    print('POLICY ADMISSIONS VERIFIED: 12 GROUPS / 444 CASES / 44 ADMITTED / 400 BLOCKED')
    print('NONEMPTY GROUPS: 11; ALL-BLOCKED GROUPS: 1; N1 AND N2-v2 FIXED')
    print('PHASE A: 44 candidate validations + pre-release N1/N2/all-policy signal freezes')
    print('PHASE B: whole-fold context freeze -> future outcomes -> group meta verification')
    print('NO TVAE FITTING; NO N1/N2 REFITTING; NO OUTER-TEST ACCESS')
    if not args.execute:
        print('PLAN ONLY. Use --execute; no candidate training, future reads or artifacts written.')
        return
    output=runs/'policy_candidate_batches'/args.batch_version
    require(not output.exists(),'Batch attempt exists; preserve it and diagnose before a fresh attempt.')
    filenames=['run_policy_candidate_batch.py','_policy_pipeline.py','validate_policy_candidate.py','freeze_policy_case.py',
        'freeze_policy_context_support.py','evaluate_policy_case_future.py','run_policy_pipeline.py',
        'build_policy_group_meta_dataset.py','_policy_group.py','_scenario_meta.py','_admitted_training.py',
        '_all_mechanism_transforms.py','_portable_model_io.py','_future_research.py']
    hashes={str(C4_ROOT/'scripts'/f):sha((C4_ROOT/'scripts'/f).read_bytes()) for f in filenames}
    output.mkdir(parents=True,exist_ok=False)
    save_json(output/'claim.json',{'purpose':'two_phase_fixed_policy_candidate_batch','created_at_utc':utc_now(),**git_context(),
        'policy_specification_sha256':POLICY_SPEC_HASH,'group_ids_in_fixed_order':groups,
        'scenario_ids_in_fixed_order':[r['scenario_id'] for r in selected],
        'group_admission_sha256':{g:r['sha256'] for g,r in reports.items()},'source_sha256':hashes,
        'policy_parameters_selected':False,'automatic_retry':False,'release_authorized':False})
    def stage(identity,label,script,arguments):
        verify_policy_spec(root)
        for path,digest in hashes.items():checked_bytes(path,digest)
        log=output/(sha(identity.encode())[:12]+'-'+label+'.log')
        print('RUNNING:',identity,label,flush=True)
        with log.open('x',encoding='utf-8') as file:
            result=subprocess.run([sys.executable,str(C4_ROOT/'scripts'/script),'--data-root',str(root),*arguments],
                stdout=file,stderr=subprocess.STDOUT,text=True)
        require(result.returncode==0,'Stage failed; preserve artifacts and inspect '+str(log))
    current=None;records=[]
    try:
        for row in selected:
            current=row['scenario_id'];group_context(root,current)
            run=runs/current
            for label,script,directory,stem,flag in (
                ('validation','validate_policy_candidate.py','scoped_candidate_validation-v1','validation','--execute'),
                ('freeze','freeze_policy_case.py','pre_release_freeze-v1','freeze','--freeze')):
                target=run/directory
                if (target/(stem+'.json')).exists():
                    checked_json(target,stem);print('REUSE VERIFIED STAGE:',current,label,flush=True)
                else:
                    require(not target.exists(),'Incomplete claimed stage: '+str(target))
                    stage(current,label,script,['--scenario-id',current,flag])
            snapshot,_=checked_json(run/'pre_release_freeze-v1','freeze');verify_freeze_dependencies(snapshot)
            print('PRE-RELEASE COMPLETE:',current,flush=True)
        stage('all-policy-groups','context-support','freeze_policy_context_support.py',['--freeze'])
        support,support_hash=verify_context_support(root)
        print('ALL PRE-RELEASE EVIDENCE / CONTEXT SUPPORT FROZEN; FUTURE PHASE STARTING',flush=True)
        for current in groups:
            report=reports[current]['report'];count=report['scoped_admissible_cases']
            if count==0:
                records.append({'group_id':current,'admitted':0,'blocked':37,'meta_rows':0,
                    'admission_sha256':reports[current]['sha256'],'status':'ADMISSION_ONLY_ALL_BLOCKED','future_targets_imputed':False})
                print('ALL BLOCKED:',current,'preserved; no future targets generated',flush=True);continue
            stage(current,'pipeline','run_policy_pipeline.py',['--group-id',current,'--execute'])
            pipeline,pipeline_hash=checked_json(runs/'group_admissions'/current/'policy-pipeline-v1','group_pipeline')
            require(pipeline['completed_scenarios']==count and pipeline['group_admissions_sha256']==reports[current]['sha256'],
                    'Policy pipeline links/counts differ.')
            stage(current,'meta-verify','build_policy_group_meta_dataset.py',['--group-id',current])
            meta=root/'c4_artifacts/meta_datasets/adult'/VERSION/(current+'-policy-v1')
            if not meta.exists():stage(current,'meta-save','build_policy_group_meta_dataset.py',['--group-id',current,'--save'])
            md,md_hash=checked_json(meta)
            require(md['fold']=='policy_calibration' and md['group_id']==current and md['rows']==count and
                    md['group_pipeline_sha256']==pipeline_hash,'Policy meta links/counts differ.')
            for filename,digest in md['files'].items():checked_bytes(meta/filename,digest)
            records.append({'group_id':current,'admitted':count,'blocked':37-count,'meta_rows':count,
                'admission_sha256':reports[current]['sha256'],'pipeline_sha256':pipeline_hash,
                'meta_metadata_sha256':md_hash,'status':'COMPLETE'})
            print('COMPLETE:',current,'META ROWS:',count,'BLOCKED:',37-count,flush=True)
        verify_context_support(root)
        for path,digest in hashes.items():checked_bytes(path,digest)
        digest=save_json(output/'batch_results.json',{'purpose':'completed_two_phase_policy_candidate_batch',
            'created_at_utc':utc_now(),**git_context(),'policy_specification_sha256':POLICY_SPEC_HASH,
            'policy_context_support_sha256':support_hash,'group_ids_in_fixed_order':groups,'records':records,
            'group_count':12,'nonempty_groups':11,'all_blocked_groups':1,'admission_cases':444,'meta_rows':44,'blocked_cases':400,
            'N1_refitted':False,'N2_refitted':False,'policy_calibrated':False,'outer_test_evaluated':False,
            'release_authorized':False,'case_log_sha256':{p.name:sha(p.read_bytes()) for p in output.glob('*.log')}})
        print('POLICY GROUPS COMPLETE: 12; META ROWS: 44; BLOCKED CASES RETAINED: 400')
        print('REPORT:',output/'batch_results.json');print('REPORT SHA256:',digest)
        print('GLOBAL/N4 SELECTION / OUTER TEST / PRODUCTION RELEASE: PENDING')
    except Exception as error:
        save_json(output/'failure.json',{'created_at_utc':utc_now(),'identity':current,'exception_type':type(error).__name__,
            'message':str(error),'completed_group_records':records,'automatic_retry':False})
        raise


if __name__=='__main__':main()
