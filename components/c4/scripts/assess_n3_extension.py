"""Apply unchanged N3 boundaries to pinned N2-v2 intervals, without observed labels."""
import io
from pathlib import Path
import numpy as np
import pandas as pd
from _common import C4_ROOT, checked_bytes, checked_json, git_context, parser, require, resolve_data_root, save_json, sha, utc_now
from _conformal_group import VERSION, verify_n1
from c4.risk.benchmark_n3 import BOUNDARIES, POPULATIONS, TARGETS, assess_safety

N2_HASH='b79af74ecd63029e34461e6a9632225fb50a033e405fc5ffe8683b9da0d32934'
SPEC_HASH='6e34b025c7dccb47b14846348e96f1d94de3d914e866fe0a51dd26e9d8bedbf0'
ORIGINAL_N3_SPEC='505a1d3b2804a99bfc8437d4d54f7fb0d0711a66149b012b86d45a808bcf35bd'


def assess_interval_table(frame, *, supported):
    require(type(supported) is bool,'N2 support must be an explicit boolean.')
    require(not frame.empty and frame.index.is_unique and frame.index.name=='scenario_id',
            'Intervals require unique nonempty scenario IDs.')
    columns={t+'__'+q for t in TARGETS for q in ('lower','median','upper')}
    require(frame.columns.is_unique and set(frame.columns)==columns,'Exactly three lower/median/upper interval sets required.')
    values=frame.to_numpy(dtype=float)
    require(np.isfinite(values).all() and (np.abs(values)<=1).all(),'Intervals must be finite margin fractions in [-1,1].')
    for t in TARGETS:
        require((frame[t+'__lower']<=frame[t+'__median']).all() and
                (frame[t+'__median']<=frame[t+'__upper']).all(),'Interval/median order differs.')
    records=[]
    for sid,row in frame.iterrows():
        result=assess_safety({t:[row[t+'__lower'],row[t+'__upper']] for t in TARGETS},
                             auto_deploy_supported=supported)
        records.append({'scenario_id':sid,**result})
    counts={'scenarios':len(records),**{signal:sum(r['release_signal']==signal for r in records)
                                      for signal in ('AUTO_DEPLOY','HUMAN_REVIEW','ESCALATE')}}
    dimensions={t:{state:sum(r['dimensions'][t]['status']==state for r in records)
                   for state in ('SAFE','UNCERTAIN','UNSAFE')} for t in TARGETS}
    return records,counts,dimensions


def main():
    p=parser(__doc__)
    p.add_argument('--freeze-spec',action='store_true')
    p.add_argument('--save',action='store_true')
    args=p.parse_args()
    require(not(args.freeze_spec and args.save),'Freeze and save in separate invocations.')
    root=resolve_data_root(args.data_root)
    n1=verify_n1(root)
    source=root/'c4_artifacts/conformal/adult'/VERSION/'n2-joint-v2'
    md,md_hash=checked_json(source)
    require(md_hash==N2_HASH and md['purpose']=='verified_pooled_n2_extension_calibration' and
            md['n1_binding']==n1 and md['rows']==150 and md['nonempty_groups']==37 and md['rank']==37 and
            md['finite_adjustment_estimated'] is True and md['release_authorized'] is False,'Wrong N2-v2 checkpoint.')
    for relative,digest in md['source_sha256'].items():checked_bytes(C4_ROOT/relative,digest)
    n2spec,n2spec_hash=checked_json(source,'specification')
    require(n2spec_hash==SPEC_HASH==md['specification_sha256'],'N2-v2 specification differs.')
    calibration,cal_hash=checked_json(source,'calibration')
    require(cal_hash==md['files']['calibration.json'] and calibration['n1_binding']==n1 and
            calibration['specification_sha256']==SPEC_HASH and calibration['finite_adjustment_estimated'] is True and
            calibration['automatic_release_authorized'] is False,'N2-v2 calibration links differ.')
    old_dir=root/'c4_artifacts/safety/adult'/VERSION/'n3-safety-v1'
    old,old_hash=checked_json(old_dir,'specification')
    require(old_hash==ORIGINAL_N3_SPEC and old['definition']['boundaries']==BOUNDARIES and
            old['definition']['population_basis']==POPULATIONS,'Original N3 boundaries/populations changed.')
    for relative,digest in old['definition']['source_sha256'].items():checked_bytes(C4_ROOT/relative,digest)
    frames={};input_hashes={'calibration.json':cal_hash,'specification.json':n2spec_hash}
    for method in ('lightgbm','linear'):
        c=calibration['methods'][method]
        require(c['alpha']==.05 and c['nonempty_groups']==37 and c['rank']==37 and
                c['fallback'] is False and c['auto_deploy_supported'] is True and
                c['adjustment'] is not None and np.isfinite(c['adjustment']) and c['adjustment']>=0,
                'N2-v2 numerical support differs.')
        filename=method+'_intervals.csv';input_hashes[filename]=md['files'][filename]
        frame=pd.read_csv(io.BytesIO(checked_bytes(source/filename,input_hashes[filename])),
                          index_col=0,float_precision='round_trip')
        require(len(frame)==150,'Expected 150 original-plus-extension interval rows.')
        assess_interval_table(frame,supported=True)
        frames[method]=frame
    require(frames['lightgbm'].index.equals(frames['linear'].index),'Method scenario IDs/order differ.')
    code_hashes={str(path.relative_to(C4_ROOT)):sha(path.read_bytes()) for path in
                 (Path(__file__),C4_ROOT/'src/c4/risk/benchmark_n3.py')}
    definition={'purpose':'n3_extension_safety_specification','n2_metadata_sha256':md_hash,
        'original_n3_specification_sha256':old_hash,'n1_binding':n1,'input_sha256':input_hashes,
        'boundaries':BOUNDARIES,'population_basis':POPULATIONS,'source_sha256':code_hashes,
        'units':'candidate_minus_champion_accuracy_fraction',
        'rules':{'SAFE':'lower >= boundary','UNSAFE':'upper < boundary','UNCERTAIN':'otherwise'},
        'aggregation':'any UNSAFE -> ESCALATE; all SAFE and numerical N2 support -> AUTO_DEPLOY signal; otherwise HUMAN_REVIEW',
        'scope':'calibration-set signal characterization only; not held-out policy evaluation',
        'slice_limitation':'True injected slice, not diagnosed slice; no diagnosed-slice safety guarantee claimed.',
        'observed_future_labels_as_inputs':False,'policy_calibrated':False,'production_release_authorized':False}
    output=root/'c4_artifacts/safety/adult'/VERSION/'n3-safety-v2'
    print('N1 / N2-v2 / ORIGINAL N3 BOUNDARIES / INTERVAL CHECKSUMS: PASS')
    print('SCENARIOS: 150 PER METHOD; BOUNDARIES: global=0; true injected slice=0; complement=-0.02')
    if args.freeze_spec:
        require(not output.exists(),'N3-v2 already exists; preserve it.')
        require(not git_context()['working_tree_dirty'],'Commit reviewed code before freezing.')
        output.mkdir(parents=True,exist_ok=False)
        digest=save_json(output/'specification.json',{'definition':definition,'created_at_utc':utc_now(),**git_context()})
        print('SPECIFICATION:',output/'specification.json')
        print('SPECIFICATION SHA256:',digest)
        print('Next use --save in a separate invocation.')
        return
    if not args.save:
        print('PLAN ONLY. Use --freeze-spec then --save; no observed labels read or results written.')
        return
    spec,spec_hash=checked_json(output,'specification')
    require(spec['definition']==definition,'N3 inputs/code changed after freeze.')
    require(not(output/'metadata.json').exists() and not(output/'claim.json').exists(),'N3-v2 already claimed; preserve it.')
    save_json(output/'claim.json',{'specification_sha256':spec_hash,'created_at_utc':utc_now(),**git_context()})
    files={};counts={};dimensions={}
    for method,frame in frames.items():
        records,counts[method],dimensions[method]=assess_interval_table(frame,supported=True)
        files[method+'_assessment.json']=save_json(output/(method+'_assessment.json'),{
            'method':method,'specification_sha256':spec_hash,'records':records,
            'counts':counts[method],'dimension_counts':dimensions[method],
            'evaluation_scope':'calibration-set signals only; independent policy performance pending'})
        print('METHOD:',method,'SIGNALS:',counts[method])
        print('DIMENSION COUNTS:',dimensions[method])
    note='\n'.join(['# N3-v2 interval safety assessment','',
        'Same boundaries as original N3: global 0, true injected slice 0, true slice complement -0.02.',
        '150 calibration scenarios assessed separately for both fixed N1 methods; these are not 300 independent scenarios.',
        'Signals use N2-v2 intervals only. No observed future labels read by this assessment.',
        'An AUTO_DEPLOY signal, if present, is not a production authorization; hard gates and final policy remain necessary.',
        'No unsafe-deployment rate, policy effectiveness or independent coverage claim can be derived from these signal counts.',
        f'Signal counts: {counts}',f'Dimension counts: {dimensions}',''])
    with (output/'N3_V2_RESEARCH_EVIDENCE.md').open('xb') as file:file.write(note.encode())
    files['N3_V2_RESEARCH_EVIDENCE.md']=sha(note.encode())
    require(verify_n1(root)==n1,'N1 changed during assessment.')
    checked_bytes(source/'metadata.json',md_hash)
    checked_bytes(old_dir/'specification.json',old_hash)
    for filename,digest in input_hashes.items():checked_bytes(source/filename,digest)
    for relative,digest in code_hashes.items():checked_bytes(C4_ROOT/relative,digest)
    digest=save_json(output/'metadata.json',{'purpose':'verified_n3_extended_calibration_assessment',
        'created_at_utc':utc_now(),**git_context(),'specification_sha256':spec_hash,
        'n2_metadata_sha256':md_hash,'original_n3_specification_sha256':old_hash,'n1_binding':n1,
        'files':files,'counts':counts,'dimension_counts':dimensions,'source_sha256':code_hashes,
        'observed_future_labels_read':False,'policy_calibrated':False,'outer_test_evaluated':False,'release_authorized':False})
    print('OUTPUT:',output)
    print('METADATA SHA256:',digest)
    print('GLOBAL / N4 POLICY AND INDEPENDENT EVALUATION PENDING; RELEASE AUTHORIZED: False')


if __name__=='__main__':main()
