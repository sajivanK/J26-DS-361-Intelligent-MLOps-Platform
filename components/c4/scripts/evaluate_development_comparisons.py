"""Verify the evaluation engine on saved policy results; not final held-out evaluation."""
import json
import numpy as np
import pandas as pd
from _common import parser,resolve_data_root,name,require,save_json,checked_json,checked_bytes,sha,utc_now,git_context
from _evaluation_engine_inputs import VERSION,load_inputs
from c4.evaluation.comparison_decisions import METHOD_IDS,decide_comparison
from c4.evaluation.comparison_calibration import apply_comparison_calibration
from c4.evaluation.pilot_evaluation_metrics import (governance_report,group_counts,forecast_report,interval_report,
    comparison_diagnostic_scores,diagnostic_risk_coverage,PRIMARY_METRICS)
from c4.evaluation.pilot_group_statistics import paired_group_bootstrap
from c4.risk.benchmark_policy import GATES


REPORT_VERSION='development-engine-check-v1'


def compute_reports(data):
    features,targets,index,groups=data['features'],data['targets'],data['index'],data['groups']
    runtime=data['runtime'];reports={};counts={};curves={};forecast_reports={};interval_reports={};subgroups=[]
    rows_by_id={r['scenario_id']:r for r in data['manifest']['scenarios']}
    namespace={g['group_id']:g['namespace'] for g in data['manifest']['groups']}
    for source,forecasts in data['forecasts'].items():forecast_reports[source]=forecast_report(targets,forecasts,index,groups)
    for comparison in METHOD_IDS:
        print('EVALUATING DEVELOPMENT METHOD:',comparison,flush=True)
        source=comparison if comparison in {'A1','A2','A3'} else 'ORIGINAL'
        forecasts=data['forecasts'][source]
        saved=data['decisions'].loc[data['decisions'].comparison.eq(comparison)]
        decisions=saved.set_index('scenario_id').decision.to_dict()
        require(set(decisions)==set(targets.index),'Comparison scenario inventory differs.')
        for sid in targets.index:
            actual=decide_comparison(comparison,features.loc[sid].to_dict(),data['gates'][sid],forecasts.loc[sid].to_dict(),
                runtime['calibrations'],runtime['selected_policies'],runtime['unsupported_contexts'])
            require(actual['decision']==decisions[sid],'Saved development decision differs from frozen executable rule.')
        reports[comparison]=governance_report(targets,index,decisions,groups)
        counts[comparison]=group_counts(targets,index,decisions,groups)
        scores=comparison_diagnostic_scores(comparison,features,forecasts,runtime)
        gate_pass=[all(data['gates'][sid].get(g) is True for g in GATES) for sid in targets.index]
        curves[comparison]=diagnostic_risk_coverage(scores,targets,index,groups,gate_pass)
        if comparison not in {'B1','B2','B3','B4','B6','B7'}:
            cid='ORIGINAL' if comparison in {'PROPOSED','B8','A7','A8'} else 'B5' if comparison=='A6' else comparison
            calibration=runtime['calibrations'][cid]
            adjusted=apply_comparison_calibration(calibration,forecasts)
            interval_reports[comparison]=interval_report(targets,adjusted,index,groups,calibration['active_targets'],
                native_alpha=.05/3 if cid=='B5' else None)
            interval_reports[comparison]['fallback']=calibration['fallback']
            interval_reports[comparison]['rank']=calibration['rank']
        categories={'namespace':[namespace[g] for g in index.group_id],
            'mechanism':[rows_by_id[sid]['mechanism'] for sid in targets.index],
            'nominal_magnitude':[rows_by_id[sid]['magnitude'] for sid in targets.index],
            'diagnosis_condition':[rows_by_id[sid]['diagnosis_condition'] for sid in targets.index],
            'repair_condition':[rows_by_id[sid]['repair_condition'] for sid in targets.index]}
        for field,values in categories.items():
            for value in sorted(set(values),key=str):
                mask=np.array([v==value for v in values]);subtargets=targets.iloc[np.flatnonzero(mask)]
                subindex=index.iloc[np.flatnonzero(mask)]
                subgroup=governance_report(subtargets,subindex,{sid:decisions[sid] for sid in subtargets.index},groups)
                subgroups.append({'comparison':comparison,'category':field,'value':str(value),'rows':len(subtargets),
                    'nonempty_groups':subgroup['nonempty_groups'],**{m:subgroup[m] for m in PRIMARY_METRICS},
                    'human_review_burden':subgroup['human_review_burden']})
    print('RUNNING 2000 SHARED PAIRED ASSIGNED-GROUP BOOTSTRAP DRAWS',flush=True)
    statistics=paired_group_bootstrap(counts,groups)
    return {'evaluation':'policy_calibration_development_engine_check_not_independent_test',
        'governance':reports,'forecast':forecast_reports,'interval':interval_reports,'diagnostic_risk_coverage':curves,
        'subgroups':subgroups,'statistics':statistics,'outer_test_evaluated':False,'production_release_authorized':False}


def main():
    p=parser(__doc__);p.add_argument('--save',action='store_true');p.add_argument('--report-version',type=name,default=REPORT_VERSION)
    args=p.parse_args();root=resolve_data_root(args.data_root)
    data,binding=load_inputs(root);reports=compute_reports(data)
    for comparison,report in reports['governance'].items():
        print('METHOD:',comparison,'UNSAFE AUTO:',report['counts']['unsafe_auto'],'AUTOMATION:',report['equal_group_automation'],
              'CONDITIONAL UNSAFE:',report['equal_group_conditional_unsafe_rate'],'COST:',report['expected_deployment_cost'])
    print('DECISION REPLAY / GOVERNANCE / FORECAST / INTERVAL / DIAGNOSTIC CURVE / PAIRED BOOTSTRAP: PASS')
    print('DEVELOPMENT DATA ONLY; NO HELD-OUT PERFORMANCE OR FORMAL RISK GUARANTEE CLAIM')
    if not args.save:
        print('PLAN ONLY. Use --save to persist the development engine verification.');return
    git=git_context();require(not git['working_tree_dirty'],'Commit evaluation implementation before saving.')
    output=root/'c4_artifacts/evaluations/adult'/VERSION/args.report_version
    require(not output.exists(),'Evaluation report exists; preserve it.')
    require(load_inputs(root)[1]==binding,'Evaluation inputs changed during computation.')
    output.mkdir(parents=True,exist_ok=False);files={}
    def document(filename,payload):files[filename]=save_json(output/filename,payload)
    def csv(filename,rows):
        content=pd.DataFrame(rows).to_csv(index=False,lineterminator='\n').encode();(output/filename).write_bytes(content);files[filename]=sha(content)
    for field in ('governance','forecast','interval','diagnostic_risk_coverage','statistics'):
        document(field+'.json',{'evaluation':reports['evaluation'],'results':reports[field]})
    csv('governance_summary.csv',[{'comparison':m,**{k:v for k,v in r.items() if not isinstance(v,(dict,list))},
        **{'count_'+k:v for k,v in r['counts'].items()}} for m,r in reports['governance'].items()])
    csv('group_counts.csv',[{'comparison':m,**row} for m,r in reports['governance'].items() for row in r['per_group']])
    csv('subgroup_summary.csv',reports['subgroups'])
    csv('diagnostic_risk_coverage.csv',[{'comparison':m,**row} for m,r in reports['diagnostic_risk_coverage'].items() for row in r['curve']])
    csv('bootstrap_intervals.csv',[{'comparison':m,'metric':k,'point_estimate':v['point_estimate'],
        'ci_lower':v['percentile_95_ci'][0] if v['percentile_95_ci'] else None,
        'ci_upper':v['percentile_95_ci'][1] if v['percentile_95_ci'] else None,
        'valid_replicates':v['valid_replicates'],'undefined_replicates':v['undefined_replicates']}
        for m,r in reports['statistics']['methods'].items() for k,v in r.items()])
    note=['# C4 development evaluation engine verification','',
        'These are policy-fold development diagnostics and selection-data results, not untouched outer-test results.',
        '44 scenarios, 11 nonempty of 12 assigned groups; empty assigned groups retained in bootstrap sampling.',
        'One shared set of 2,000 group-resampling draws; seed 20261016. Duplicated draws retain multiplicity.',
        'Undefined conditional risks are null; unsupported bootstrap intervals remain null, never zero-filled.',
        'Risk-coverage curves are forced score-prefix diagnostics, not operational policy thresholds.',
        'B8 and A8 are equivalent; synthetic acceptance fixtures are excluded from research results.','',
        '| Method | Unsafe AUTO count | Group automation | Group conditional unsafe | Group illustrative cost |',
        '|---|---:|---:|---:|---:|']
    for m,r in reports['governance'].items():note.append(f"| {m} | {r['counts']['unsafe_auto']} | {r['equal_group_automation']} | {r['equal_group_conditional_unsafe_rate']} | {r['expected_deployment_cost']} |")
    note+=['','Forecast, interval, bootstrap, subgroup and score-curve details are in the accompanying JSON/CSV files.',
           'No autonomous-deployment benefit or adaptive-policy superiority follows from a review-only primary policy.',
           'Outer pipeline implementation/freezing and held-out-scenario evaluation are still pending.','']
    content='\n'.join(note).encode();(output/'C4_DEVELOPMENT_EVALUATION_EVIDENCE.md').write_bytes(content);files['C4_DEVELOPMENT_EVALUATION_EVIDENCE.md']=sha(content)
    digest=save_json(output/'metadata.json',{'purpose':'verified_development_evaluation_engine',
        'created_at_utc':utc_now(),**git,'binding':binding,'rows':44,'assigned_groups':12,'nonempty_groups':11,
        'admission_cases':data['admission_cases'],'blocked_cases':data['blocked_cases'],
        'bootstrap_replicates':2000,'bootstrap_seed':20261016,'files':files,'independent_evaluation':False,
        'outer_test_evaluated':False,'release_authorized':False})
    for filename,expected in files.items():checked_bytes(output/filename,expected)
    require(checked_json(output,'governance')[0]['results']==reports['governance'],'Metric JSON reload differs.')
    print('OUTPUT:',output);print('METADATA SHA256:',digest)
    print('NEXT: freeze comparison/evaluation readiness, then prepare the outer execution harness.')


if __name__=='__main__':main()
