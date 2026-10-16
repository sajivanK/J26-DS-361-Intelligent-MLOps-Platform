"""Freeze and compute the predefined pooled original-plus-extension calibration."""
import io
import json
from pathlib import Path
import pandas as pd
from _common import C4_ROOT, checked_bytes, checked_json, git_context, parser, require, resolve_data_root, save_json, sha, utc_now
from _extension_group import EXTENSION_HASH, VERSION, load_extension, verify_n1
from c4.conformal.benchmark_n2 import TARGETS, JointCalibration, apply_joint_calibration, calibration_diagnostics
from c4.conformal.benchmark_n2_extension import fit_extended_calibration


def main():
    p=parser(__doc__)
    p.add_argument('--freeze-spec',action='store_true')
    p.add_argument('--save',action='store_true')
    args=p.parse_args()
    require(not(args.freeze_spec and args.save),'Freeze specification and calibrate in separate invocations.')
    root=resolve_data_root(args.data_root)
    n1=verify_n1(root)
    frozen,base_hash,protocol,manifest,manifest_hash,extension_hash=load_extension(root)
    directory=root/'c4_artifacts/meta_datasets/adult'/VERSION/'conformal-plus-extension-audit-v2'
    md,md_hash=checked_json(directory)
    require(md['purpose']=='verified_combined_extended_conformal_snapshot' and md['n1_binding']==n1 and
            md['extension_freeze_sha256']==EXTENSION_HASH and md['rows']==150 and md['nonempty_groups']==37 and
            md['all_blocked_groups']==3 and md['admission_cases']==1480 and md['blocked_cases']==1330,
            'Combined audited inventory differs.')
    checked_bytes(C4_ROOT/'scripts/audit_extended_calibration.py',md['source_code_sha256'])
    for path,digest in md['source_links'].items():checked_bytes(path,digest)
    tables={}
    for filename,digest in md['files'].items():
        data=checked_bytes(directory/filename,digest)
        if filename.endswith('.csv'):tables[filename]=pd.read_csv(io.BytesIO(data),float_precision='round_trip')
    index=tables['scenario_index.csv']
    targets=tables['targets.csv'].set_index('scenario_id')
    require(index.scenario_id.tolist()==targets.index.tolist() and index.group_id.nunique()==37 and
            index.groupby('group_id').split_role.nunique().eq(1).all() and
            index.split_role.value_counts().to_dict()=={'conformal_extension':86,'conformal_calibration':64},
            'Target/index identities, fold counts or assignments differ.')
    code_paths=[Path(__file__),C4_ROOT/'src/c4/conformal/benchmark_n2_extension.py',
                C4_ROOT/'scripts/audit_extended_calibration.py',C4_ROOT/'src/c4/conformal/benchmark_n2.py']
    code_hashes={str(p.relative_to(C4_ROOT)):sha(p.read_bytes()) for p in code_paths}
    definition={'purpose':'pooled_calibration_support_extension_specification','n1_binding':n1,
        'extension_freeze_sha256':EXTENSION_HASH,'protocol_freeze_sha256':base_hash,
        'snapshot_metadata_sha256':md_hash,'table_sha256':md['files'],'source_sha256':code_hashes,
        'N2_definition':protocol['N2'],'pooling':'all original and extension nonempty groups once; group max over variants and targets',
        'assigned_groups':40,'nonempty_groups':37,'rows':150,'alpha':.05,'rank':37,
        'original_result_preserved':True,'policy_and_outer_test_excluded':True,
        'evaluation':'calibration-set diagnostics only; independent validation pending'}
    output=root/'c4_artifacts/conformal/adult'/VERSION/'n2-joint-v2'
    print('COMBINED SNAPSHOT / ORIGINAL N2 / EXTENSION / FROZEN N1: PASS')
    print('CALIBRATION: 150 SCENARIOS; 37 NONEMPTY GROUPS; ALPHA: 0.05; RANK: 37')
    if args.freeze_spec:
        require(not output.exists(),'N2-v2 version already exists; preserve it.')
        require(not git_context()['working_tree_dirty'],'Commit reviewed code before freezing N2-v2 specification.')
        output.mkdir(parents=True,exist_ok=False)
        digest=save_json(output/'specification.json',{'definition':definition,'created_at_utc':utc_now(),**git_context()})
        print('SPECIFICATION:',output/'specification.json')
        print('SPECIFICATION SHA256:',digest)
        print('NO CALIBRATION COMPUTED; next use --save in a separate invocation.')
        return
    if not args.save:
        print('PLAN ONLY. Use --freeze-spec then --save. No adjustment computed or artifacts written.')
        return
    spec,spec_hash=checked_json(output,'specification')
    require(spec['definition']==definition,'Calibration inputs/code changed after specification freeze.')
    require(not(output/'metadata.json').exists() and not(output/'claim.json').exists(),'Calibration already claimed; preserve it.')
    save_json(output/'claim.json',{'specification_sha256':spec_hash,'created_at_utc':utc_now(),**git_context(),
        'N1_refitted':False,'future_reevaluated':False,'release_authorized':False})
    results={}; files={}
    for method in ('lightgbm','linear'):
        forecasts=tables[method+'_frozen_forecasts.csv'].set_index('scenario_id')
        calibration,scores,groups=fit_extended_calibration(targets,forecasts,index)
        require(calibration.nonempty_groups==37 and calibration.rank==37 and not calibration.fallback,
                'Pooled support differs from the frozen audit.')
        intervals=apply_joint_calibration(calibration,forecasts)
        restored=JointCalibration.from_dict(json.loads(json.dumps(calibration.to_dict())))
        pd.testing.assert_frame_equal(intervals,apply_joint_calibration(restored,forecasts),check_exact=True)
        diagnostics=calibration_diagnostics(targets,intervals,index)
        widths={t:diagnostics['targets'][t]['mean_width_fraction'] for t in TARGETS}
        results[method]={'calibration':calibration.to_dict(),'diagnostics':diagnostics,'mean_widths_fraction':widths}
        for label,frame in (('scores',scores),('group_scores',groups),('intervals',intervals)):
            filename=method+'_'+label+'.csv'
            data=frame.to_csv(index=True,lineterminator='\n').encode()
            with (output/filename).open('xb') as file:file.write(data)
            files[filename]=sha(data)
        print('METHOD:',method,'FINITE ADJUSTMENT:',calibration.adjustment)
        print('MEAN INTERVAL WIDTHS (FRACTION):',json.dumps(widths,sort_keys=True))
        print('IN-SAMPLE JOINT COVERAGE:',diagnostics['scenario_joint_coverage'],
              '; INDEPENDENT COVERAGE VALIDATION: False')
    files['calibration.json']=save_json(output/'calibration.json',{
        'purpose':'frozen_pooled_n2_group_calibration','status':'FINITE_ADJUSTMENT_ESTIMATED',
        'methods':{m:r['calibration'] for m,r in results.items()},'specification_sha256':spec_hash,
        'n1_binding':n1,'extension_freeze_sha256':EXTENSION_HASH,'snapshot_metadata_sha256':md_hash,
        'finite_adjustment_estimated':True,'useful_interval_performance_established':False,
        'independent_95_percent_coverage_validated':False,'automatic_release_authorized':False})
    files['calibration_diagnostics.json']=save_json(output/'calibration_diagnostics.json',{
        'methods':results,'evaluation':'in_sample_calibration_only_not_outer_test',
        'limitation':'Extension chosen after original support limitation; shared data do not establish exchangeability.'})
    note='\n'.join(['# N2-v2 calibration-support extension','',
        'Original 64 scenarios/17 nonempty groups plus 86 extension scenarios/20 groups: 150 rows in 37 groups.',
        'All 20 predefined extension groups completed; 1330 blocked cases across both sets retained without target imputation.',
        'Rank ceil((37+1)*0.95)=37. Adjustment is the maximum group score separately for each fixed N1 method.',
        'Original N2 insufficient-support result and original N3 assessments remain unchanged.',
        'Finite adjustment is numerical support, not proof of useful intervals, safe automation or independent 95% coverage.',
        'All coverage and width results here are calibration-set diagnostics; untouched outer-test evaluation remains pending.',
        '',json.dumps(results,indent=2,sort_keys=True,allow_nan=False),''])
    with (output/'N2_V2_RESEARCH_EVIDENCE.md').open('xb') as file:file.write(note.encode())
    files['N2_V2_RESEARCH_EVIDENCE.md']=sha(note.encode())
    require(verify_n1(root)==n1,'N1 changed during calibration.')
    checked_bytes(directory/'metadata.json',md_hash)
    for path,digest in md['source_links'].items():checked_bytes(path,digest)
    for filename,digest in md['files'].items():checked_bytes(directory/filename,digest)
    for relative,digest in code_hashes.items():checked_bytes(C4_ROOT/relative,digest)
    digest=save_json(output/'metadata.json',{'purpose':'verified_pooled_n2_extension_calibration',
        'created_at_utc':utc_now(),**git_context(),'specification_sha256':spec_hash,'n1_binding':n1,
        'extension_freeze_sha256':EXTENSION_HASH,'snapshot_metadata_sha256':md_hash,
        'rows':150,'nonempty_groups':37,'assigned_groups':40,'rank':37,'files':files,'source_sha256':code_hashes,
        'finite_adjustment_estimated':True,'N1_refitted':False,'future_reevaluated':False,
        'outer_test_evaluated':False,'release_authorized':False})
    print('SCORES / GROUP MAX / FINITE RANK / JSON RELOAD: PASS')
    print('OUTPUT:',output)
    print('METADATA SHA256:',digest)
    print('N3-v2 / GLOBAL-N4 POLICY / INDEPENDENT EVALUATION: PENDING; RELEASE AUTHORIZED: False')


if __name__=='__main__':main()
