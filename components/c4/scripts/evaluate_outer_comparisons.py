"""Evaluate the completed outer fold using the unchanged frozen PP1 metric definitions."""
import json
import numpy as np
import pandas as pd
from _common import parser,resolve_data_root,name,require,save_json,checked_json,checked_bytes,sha,utc_now,git_context
from _outer_evaluation_inputs import VERSION,load_inputs,sources
from c4.evaluation.comparison_decisions import METHOD_IDS,decide_comparison
from c4.evaluation.comparison_calibration import apply_comparison_calibration
from c4.evaluation.pilot_evaluation_metrics import (governance_report,group_counts,forecast_report,interval_report,
    comparison_diagnostic_scores,diagnostic_risk_coverage,PRIMARY_METRICS)
from c4.evaluation.pilot_group_statistics import paired_group_bootstrap
from c4.risk.benchmark_policy import GATES


REPORT_VERSION='outer-comparison-evaluation-v1'


def compute_reports(data):
    features,targets,index,groups=data['features'],data['targets'],data['index'],data['groups']
    runtime=data['runtime'];reports={};counts={};curves={};forecast_reports={};interval_reports={};subgroups=[]
    rows_by_id={r['scenario_id']:r for r in data['manifest']['scenarios']}
    namespace={g['group_id']:g['namespace'] for g in data['manifest']['groups']}
    for source,forecasts in data['forecasts'].items():forecast_reports[source]=forecast_report(targets,forecasts,index,groups)
    for comparison in METHOD_IDS:
        print('EVALUATING OUTER METHOD:',comparison,flush=True)
        source=comparison if comparison in {'A1','A2','A3'} else 'ORIGINAL'
        forecasts=data['forecasts'][source]
        saved=data['decisions'].loc[data['decisions'].comparison.eq(comparison)]
        decisions=saved.set_index('scenario_id').decision.to_dict()
        require(set(decisions)==set(targets.index),'Comparison scenario inventory differs.')
        for sid in targets.index:
            actual=decide_comparison(comparison,features.loc[sid].to_dict(),data['gates'][sid],forecasts.loc[sid].to_dict(),
                runtime['calibrations'],runtime['selected_policies'],runtime['unsupported_contexts'])
            require(actual['decision']==decisions[sid],'Saved outer decision differs from frozen executable rule.')
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
    return {'evaluation':'held_out_outer_replication_groups_within_controlled_adult_pilot',
        'governance':reports,'forecast':forecast_reports,'interval':interval_reports,'diagnostic_risk_coverage':curves,
        'subgroups':subgroups,'statistics':statistics,'outer_test_evaluated':True,'production_release_authorized':False}


def display(value):
    return 'undefined' if value is None else f'{value:.6g}' if isinstance(value,float) else str(value)


def evidence_note(reports,binding):
    lines=['# C4 final outer-test comparison evidence','',
        'Scope: held-out replication groups in the Adult accuracy-margin simulation pilot.',
        'This is not a second independent dataset, production validation, or a formal conditional-risk guarantee.',
        '16 assigned groups; 14 nonempty groups; 56 admitted cases; 536 blocked of 592 admissions.',
        'All decisions were frozen for all admitted cases before any outer future evaluator ran.',
        'The recovered snapshot preserves the original failed save and corrects only Markdown newline serialization.',
        'No models, calibration adjustments, selected policies or thresholds were changed for these results.',
        'Primary outcomes: global, true-injected-slice and true-slice-complement candidate-minus-champion accuracy margins.',
        'The diagnosed slice remains a decision-time evidence population and is distinct from the true injected slice.','',
        '| Method | AUTO | REVIEW | ESCALATE | Unsafe AUTO | Group automation | Conditional unsafe | Illustrative cost |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for method,r in reports['governance'].items():
        c=r['counts'];lines.append('| '+' | '.join(display(v) for v in
            (method,c['auto'],c['review'],c['escalate'],c['unsafe_auto'],r['equal_group_automation'],
             r['equal_group_conditional_unsafe_rate'],r['expected_deployment_cost']))+' |')
    lines+=['','## Interpretation limits','',
        'Governance and forecast metrics describe the 56 admitted cases; blocked cases have no invented future labels.',
        'All 16 assigned groups, including two empty groups, are retained in the 2,000 shared paired group bootstrap draws.',
        'Bootstrap seed: 20261016. Intervals are exploratory conditional-simulation percentile intervals.',
        'Undefined conditional risk and unsupported bootstrap intervals are null, never zero-filled.',
        'Paired differences are PROPOSED minus comparator; there are no multiplicity-controlled significance claims.',
        'Costs are the frozen investigator-defined illustrative costs, not measured business costs.',
        'Risk-coverage curves force score-prefix deployment for diagnostics; they do not change the operational policy.',
        'A4 omits the slice interval: its two-target coverage must not be presented as three-target joint coverage.',
        'B5/A6 retain the frozen full-bound Bonferroni fallback. A8 reuses B8 by definition.',
        'The primary measured policy remains review-only. Zero unsafe deployments with zero automation does not establish safe autonomous deployment.',
        'Equivalent/abstaining decision results cannot demonstrate an adaptive-policy or feature-ablation deployment benefit.',
        'Synthetic acceptance fixtures are excluded. Real upstream-producer and live deployment validation remain pending.','',
        '## Checkpoint bindings','']
    lines += [f'- {key}: `{value}`' for key,value in binding.items() if isinstance(value,str)]
    lines+=['','Detailed forecast errors, pinball losses, interval widths/coverage, group counts, subgroup support,',
            'cost sensitivity and paired bootstrap intervals are saved alongside this document.','']
    return '\n'.join(lines).encode('utf-8')


def save_reports(output,reports,data,binding,git):
    require(not output.exists(),'Outer evaluation report exists; preserve it.')
    output.mkdir(parents=True,exist_ok=False);files={}
    def document(filename,payload):files[filename]=save_json(output/filename,payload)
    def csv(filename,rows):
        contents=pd.DataFrame(rows).to_csv(index=False,lineterminator='\n').encode('utf-8')
        with (output/filename).open('xb') as stream:stream.write(contents)
        files[filename]=sha(contents)
    for field in ('governance','forecast','interval','diagnostic_risk_coverage','statistics'):
        document(field+'.json',{'evaluation':reports['evaluation'],'results':reports[field]})
    csv('governance_summary.csv',[{'comparison':m,**{k:v for k,v in r.items() if not isinstance(v,(dict,list))},
        **{'count_'+k:v for k,v in r['counts'].items()}} for m,r in reports['governance'].items()])
    csv('group_counts.csv',[{'comparison':m,**row} for m,r in reports['governance'].items() for row in r['per_group']])
    csv('admission_group_summary.csv',data['group_summary'].to_dict('records'))
    csv('subgroup_summary.csv',reports['subgroups'])
    csv('diagnostic_risk_coverage.csv',[{'comparison':m,**row} for m,r in reports['diagnostic_risk_coverage'].items() for row in r['curve']])
    csv('forecast_summary.csv',[{'forecaster':m,'target':t,'weighting':w,**{k:v for k,v in values.items() if not isinstance(v,dict)}}
        for m,result in reports['forecast'].items() for t,weighted in result.items()
        for w,values in weighted.items() if isinstance(values,dict)])
    csv('pinball_summary.csv',[{'forecaster':m,'target':t,'weighting':w,'quantile':q,'pinball_loss':loss,'kind':kind}
        for m,result in reports['forecast'].items() for t,weighted in result.items()
        for w,values in weighted.items() if isinstance(values,dict)
        for kind in ('pinball','raw_pinball') for q,loss in values.get(kind,{}).items()])
    csv('interval_summary.csv',[{'comparison':m,'target':t,'weighting':w,'fallback':r['fallback'],
        'three_target_comparable':r['three_target_comparable'],**v}
        for m,r in reports['interval'].items() for t,weighted in r['targets'].items() for w,v in weighted.items()])
    csv('bootstrap_intervals.csv',[{'comparison':m,'metric':k,'point_estimate':v['point_estimate'],
        'ci_lower':v['percentile_95_ci'][0] if v['percentile_95_ci'] else None,
        'ci_upper':v['percentile_95_ci'][1] if v['percentile_95_ci'] else None,
        'valid_replicates':v['valid_replicates'],'undefined_replicates':v['undefined_replicates']}
        for m,r in reports['statistics']['methods'].items() for k,v in r.items()])
    csv('paired_differences.csv',[{'comparator':m,'metric':k,'proposed_minus_comparator':v['point_estimate'],
        'ci_lower':v['percentile_95_ci'][0] if v['percentile_95_ci'] else None,
        'ci_upper':v['percentile_95_ci'][1] if v['percentile_95_ci'] else None,
        'valid_replicates':v['valid_replicates'],'undefined_replicates':v['undefined_replicates']}
        for m,r in reports['statistics']['paired_proposed_minus_comparator'].items() for k,v in r.items()])
    csv('cost_sensitivity.csv',[{'comparison':m,'unsafe_auto_cost':cost,'expected_deployment_cost':value}
        for m,r in reports['governance'].items() for cost,value in r['cost_sensitivity'].items()])
    note=evidence_note(reports,binding)
    with (output/'C4_OUTER_EVALUATION_EVIDENCE.md').open('xb') as stream:stream.write(note)
    files['C4_OUTER_EVALUATION_EVIDENCE.md']=sha(note)
    digest=save_json(output/'metadata.json',{'purpose':'verified_final_outer_evaluation','created_at_utc':utc_now(),**git,
        'binding':binding,'rows':56,'assigned_groups':16,'nonempty_groups':14,'all_blocked_groups':2,
        'admission_cases':592,'blocked_cases':536,'bootstrap_replicates':2000,'bootstrap_seed':20261016,
        'files':files,'outer_test_evaluated':True,'independent_dataset_evaluated':False,
        'models_refitted':False,'policies_reselected':False,'production_release_authorized':False})
    for filename,expected in files.items():checked_bytes(output/filename,expected)
    require(checked_json(output,'governance')[0]['results']==reports['governance'],'Governance JSON reload differs.')
    return digest


def main():
    p=parser(__doc__);p.add_argument('--save',action='store_true')
    args=p.parse_args();root=resolve_data_root(args.data_root)
    data,binding=load_inputs(root)
    reports=compute_reports(data)
    require(reports['governance']['A8']==reports['governance']['B8'],'A8/B8 declared equivalence differs.')
    for method,r in reports['governance'].items():
        c=r['counts']
        print('METHOD:',method,'AUTO:',c['auto'],'REVIEW:',c['review'],'ESCALATE:',c['escalate'],
              'UNSAFE AUTO:',c['unsafe_auto'],'GROUP AUTOMATION:',r['equal_group_automation'],
              'CONDITIONAL UNSAFE:',r['equal_group_conditional_unsafe_rate'],'COST:',r['expected_deployment_cost'])
    print('FROZEN DECISION REPLAY / FORECAST / COVERAGE / COST / PAIRED GROUP STATISTICS: PASS')
    print('HELD-OUT OUTER GROUPS; ADULT SIMULATION PILOT; NO FORMAL CONDITIONAL-RISK GUARANTEE')
    if not args.save:
        print('PLAN ONLY. Use --save to persist final outer reports; no training or future prediction.');return
    git=git_context();require(not git['working_tree_dirty'],'Commit reporting implementation before --save.')
    require(load_inputs(root)[1]==binding,'Frozen outer inputs changed during evaluation.')
    output=root/'c4_artifacts/evaluations/adult'/VERSION/REPORT_VERSION
    digest=save_reports(output,reports,data,binding,git)
    print('OUTPUT:',output);print('METADATA SHA256:',digest)
    print('NEXT: export_outer_result_plots.py --save, then preserve final PP1 research evidence.')


if __name__=='__main__':main()
