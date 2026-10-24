"""Compute frozen joint group scores; persist the required insufficient-support fallback."""
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

from _common import C4_ROOT, checked_bytes, checked_json, git_context, name, parser, require, resolve_data_root, save_json, sha, utc_now
from _conformal_group import VERSION, verify_n1
from c4.conformal.benchmark_n2 import TARGETS, JointCalibration, apply_joint_calibration, calibration_diagnostics, fit_joint_calibration


def main():
    p=parser(__doc__)
    p.add_argument('--snapshot-version',type=name,default='conformal-all20-audit-v1')
    p.add_argument('--calibration-version',type=name,default='n2-joint-v1')
    p.add_argument('--save',action='store_true')
    args=p.parse_args()
    root=resolve_data_root(args.data_root)
    n1=verify_n1(root)
    directory=root/'c4_artifacts/meta_datasets/adult'/VERSION/args.snapshot_version
    metadata,metadata_hash=checked_json(directory)
    require(metadata['purpose']=='verified_combined_conformal_snapshot' and metadata['fold']=='conformal_calibration' and
            metadata['rows']==64 and metadata['nonempty_groups']==17 and metadata['all_blocked_groups']==3 and
            metadata['admission_cases']==740 and metadata['blocked_cases']==676 and metadata['n1_binding']==n1,
            'Wrong audited conformal inventory or N1 version.')
    checked_bytes(C4_ROOT/'scripts/audit_conformal_snapshot.py',metadata['source_code_sha256'])
    for path,expected in metadata['source_links'].items():checked_bytes(path,expected)
    tables={}
    for filename,expected in metadata['files'].items():
        data=checked_bytes(directory/filename,expected)
        if filename.endswith('.csv'):tables[filename]=pd.read_csv(io.BytesIO(data),float_precision='round_trip')
    index=tables['scenario_index.csv']
    targets=tables['targets.csv'].set_index('scenario_id')
    require(index.scenario_id.tolist()==targets.index.tolist() and index.group_id.nunique()==17 and
            index.split_role.eq('conformal_calibration').all(),'Conformal target/index fold differs.')
    frozen_dir=root/'c4_artifacts/benchmark_protocols/adult'/VERSION
    frozen,freeze_hash=checked_json(frozen_dir,'freeze')
    require(freeze_hash==metadata['protocol_freeze_sha256']==n1['protocol_freeze_sha256'],'Frozen protocol differs.')
    protocol=json.loads(checked_bytes(frozen_dir/'protocol.json',frozen['protocol_sha256']))
    definition=protocol['N2']
    require(definition['alpha']==.05 and definition['raw_quantiles']==[.05,.5,.95] and
        definition['joint_score']=='max_over_three_targets_of_max_0_lower_minus_y_y_minus_upper' and
        definition['calibration_unit']=='replication_group' and
        definition['group_score']=='max_joint_score_across_admitted_variants_within_group' and
        definition['quantile_rank']=='ceil((nonempty_calibration_groups_plus_1)*0.95)' and
        definition['too_few_nonempty_groups']=='uninformative_margin_bounds_and_disable_AUTO_DEPLOY_no_informative_95_percent_claim',
        'N2 implementation does not match the frozen protocol.')
    results={}
    for method in ('lightgbm','linear'):
        forecasts=tables[method+'_frozen_forecasts.csv'].set_index('scenario_id')
        calibration,scores,grouped=fit_joint_calibration(targets,forecasts,index)
        require(calibration.nonempty_groups==17 and calibration.rank==18 and calibration.fallback,
                'Frozen small-sample fallback differs.')
        intervals=apply_joint_calibration(calibration,forecasts)
        pd.testing.assert_frame_equal(intervals,apply_joint_calibration(JointCalibration.from_dict(json.loads(json.dumps(calibration.to_dict()))),forecasts),check_exact=True)
        diagnostics=calibration_diagnostics(targets,intervals,index)
        results[method]={'calibration':calibration.to_dict(),'scores':scores,'group_scores':grouped,
                         'intervals':intervals,'diagnostics':diagnostics}
        print('METHOD:',method,'SCENARIOS: 64; NONEMPTY GROUPS: 17; REQUIRED RANK: 18')
        print('STATUS: INSUFFICIENT_NONEMPTY_GROUPS_FALLBACK; FINITE ADJUSTMENT: NONE')
        print('INTERVALS: [-1, 1] FOR ALL THREE MARGINS; AUTO_DEPLOY SUPPORTED: False')
        print('IN-SAMPLE CALIBRATION JOINT COVERAGE:',diagnostics['scenario_joint_coverage'],
              '(trivial full-range containment; not independent coverage validation)')
    print('N1 REFITTED: False; CANDIDATE RETRAINING / FUTURE REEVALUATION: False')
    if not args.save:
        print('PLAN ONLY. Use --save to persist scores, fallback calibration and evidence.')
        return
    output=root/'c4_artifacts/conformal/adult'/VERSION/args.calibration_version
    require(not output.exists(),'Calibration version exists; never overwrite it.')
    output.mkdir(parents=True,exist_ok=False)
    source_hashes={str(p.relative_to(C4_ROOT)):sha(p.read_bytes()) for p in
        (Path(__file__),C4_ROOT/'src/c4/conformal/benchmark_n2.py',C4_ROOT/'scripts/audit_conformal_snapshot.py')}
    save_json(output/'claim.json',{'created_at_utc':utc_now(),**git_context(),'conformal_snapshot_metadata_sha256':metadata_hash,
        'n1_binding':n1,'protocol_freeze_sha256':freeze_hash,'source_sha256':source_hashes,
        'calibration_definition':definition,'outcome_based_rule_changes':False})
    files={}
    calibrations={}
    diagnostics={}
    for method,result in results.items():
        calibrations[method]=result['calibration'];diagnostics[method]=result['diagnostics']
        for label in ('scores','group_scores','intervals'):
            filename=method+'_'+label+'.csv'
            data=result[label].to_csv(index=True,lineterminator='\n').encode()
            (output/filename).write_bytes(data);files[filename]=sha(data)
    files['calibration.json']=save_json(output/'calibration.json',{'purpose':'frozen_n2_joint_group_calibration',
        'status':'INSUFFICIENT_NONEMPTY_GROUPS_FALLBACK','methods':calibrations,'definition':definition,
        'n1_binding':n1,'conformal_snapshot_metadata_sha256':metadata_hash,
        'finite_adjustment_estimated':False,'informative_95_percent_intervals':False,
        'automatic_release_authorized':False,'outer_test_evaluated':False})
    files['calibration_diagnostics.json']=save_json(output/'calibration_diagnostics.json',{'methods':diagnostics,
        'note':'In-sample full-range containment is not an independent 95% coverage validation.'})
    group_summary=tables['group_summary.csv']
    note='\n'.join(['# C4 N2 joint conformal evidence','','Status: INSUFFICIENT_NONEMPTY_GROUPS_FALLBACK.',
        'Both fixed N1 methods used the same 64 conformal scenarios in 17 nonempty groups; three all-blocked groups were retained without replacement.',
        'Per-target score = max(0, lower-y, y-upper); per-scenario joint score = max over three targets; per-group score = max over admitted variants.',
        'Frozen alpha=0.05; rank=ceil((17+1)*0.95)=18. Seventeen group scores cannot supply the eighteenth order statistic.',
        'No finite common conformal adjustment was estimated. The predefined fallback sets all three bounds to [-1,1] and disables AUTO_DEPLOY.',
        'Interval width is 2.0 accuracy-margin fractions (200 percentage points); it is deliberately uninformative.',
        'Calibration-set coverage is 1.0 because the bounds span the complete target domain. This does not validate useful 95% prediction coverage.',
        'Shared source observations do not establish exchangeability or independent population replications.',
        'N1 was not refitted; no candidate retraining, generator refitting or future classifier reevaluation occurred in calibration.',
        'N3 / global policy / N4 / outer-test evaluation and real integration remain pending.',
        '',f'Conformal snapshot metadata SHA256: `{metadata_hash}`',
        f'N1 metadata SHA256: `{n1["run_metadata_sha256"]}`',
        '', 'All-blocked groups: '+', '.join(group_summary.loc[group_summary.rows.eq(0),'group_id']), ''])
    (output/'N2_RESEARCH_EVIDENCE.md').write_bytes(note.encode());files['N2_RESEARCH_EVIDENCE.md']=sha(note.encode())
    require(verify_n1(root)==n1,'N1 changed during calibration.')
    checked_bytes(directory/'metadata.json',metadata_hash)
    for relative,expected in source_hashes.items():checked_bytes(C4_ROOT/relative,expected)
    digest=save_json(output/'metadata.json',{'purpose':'verified_n2_calibration_artifact_with_frozen_fallback',
        'created_at_utc':utc_now(),**git_context(),'status':'INSUFFICIENT_NONEMPTY_GROUPS_FALLBACK',
        'conformal_snapshot_metadata_sha256':metadata_hash,'n1_binding':n1,'protocol_freeze_sha256':freeze_hash,
        'assigned_groups':20,'nonempty_groups':17,'all_blocked_groups':3,'scenarios':64,'quantile_rank':18,
        'files':files,'source_sha256':source_hashes,'finite_adjustment_estimated':False,
        'N1_refitted':False,'future_reevaluated':False,'informative_95_percent_intervals':False,
        'outer_test_evaluated':False,'release_authorized':False})
    print('N2 SCORE / GROUP AGGREGATION / FALLBACK / JSON RELOAD CHECKS: PASS')
    print('CALIBRATION:',output/'calibration.json')
    print('OUTPUT:',output)
    print('METADATA SHA256:',digest)
    print('INFORMATIVE 95% INTERVALS: False; AUTO_DEPLOY SUPPORTED: False; RELEASE AUTHORIZED: False')


if __name__=='__main__':main()
