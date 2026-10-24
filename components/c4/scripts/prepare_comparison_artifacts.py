"""Prepare frozen B5/A1-A6 intervals and policies; execute all B1-B8/A1-A8 development rules."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from _common import parser,resolve_data_root,require,name,checked_json,checked_bytes,save_json,sha,utc_now,git_context
from _comparison_inputs import VERSION,SPEC_VERSION,load_inputs,fixed_predictions
from c4.evaluation.comparison_calibration import CALIBRATION_IDS,fit_comparison_calibration,apply_comparison_calibration,interval_diagnostics
from c4.evaluation.comparison_decisions import METHOD_IDS,POLICY_VARIANTS,comparison_bounds,policy_for_active_targets,decide_comparison
from c4.risk.benchmark_policy import candidate_policies
from c4.risk.benchmark_policy_selection import select_policy,policy_statistics


def main():
    p=parser(__doc__);mode=p.add_mutually_exclusive_group()
    mode.add_argument('--freeze-spec',action='store_true');mode.add_argument('--prepare',action='store_true')
    p.add_argument('--run-version',type=name,default='prepared-v1');args=p.parse_args()
    root=resolve_data_root(args.data_root)
    models,tables,gates,binding=load_inputs(root)
    print('EVALUATION / A1-A3 MODELS / ORIGINAL CONTROLLER / DEVELOPMENT SNAPSHOTS: PASS')
    print('CONFORMAL: 150 SCENARIOS / 37 NONEMPTY GROUPS; POLICY: 44 SCENARIOS / 11 NONEMPTY GROUPS')
    print('INTERVALS: A1 / A2 / A3 / A4 / A5 / B5; A6 REUSES B5 INTERVALS')
    print('B5/A6: BONFERRONI RANK 38 WITH 37 GROUPS -> FULL-BOUND FALLBACK')
    print('POLICIES: CHANGED VARIANTS SELECT THEIR OWN FROZEN-RULE POLICY; PRIMARY CONTROLLER UNCHANGED')
    print('NO N1 FIT / TVAE / CANDIDATE RETRAINING / NEW FUTURE EVALUATION / OUTER ACCESS')
    base=root/'c4_artifacts/comparison_calibrations/adult'/VERSION/SPEC_VERSION
    if args.freeze_spec:
        require(not base.exists(),'Comparison specification exists; preserve it.')
        git=git_context();require(not git['working_tree_dirty'],'Commit comparison code before freezing.')
        digest=save_json(base/'specification.json',{'purpose':'frozen_comparison_preparation_bindings',
            'created_at_utc':utc_now(),**git,'binding':binding,'N2_computed':False,'policies_selected':False,
            'development_outcomes_already_available':True,'outer_outcomes_read':False,'release_authorized':False})
        print('SPECIFICATION:',base/'specification.json');print('SPECIFICATION SHA256:',digest)
        print('Next use --prepare in a separate invocation.');return
    if not args.prepare:
        print('PLAN ONLY. Commit/push, then --freeze-spec, then --prepare.');return
    frozen,spec_hash=checked_json(base,'specification')
    require(frozen['binding']==binding,'Comparison inputs or code changed after specification freeze.')
    output=base/args.run_version;require(not output.exists(),'Comparison run exists; preserve artifacts and inspect before a new attempt.')
    output.mkdir(parents=True,exist_ok=False)
    save_json(output/'claim.json',{'created_at_utc':utc_now(),**git_context(),'specification_sha256':spec_hash,
        'outer_test_accessed':False,'original_artifacts_overwritten':False,'automatic_retry':False})
    files={}
    def csv(filename,frame,index=True):
        data=frame.to_csv(index=index,lineterminator='\n').encode();(output/filename).write_bytes(data);files[filename]=sha(data)
    def document(filename,payload):files[filename]=save_json(output/filename,payload)
    try:
        predictions={}
        for role,table in tables.items():
            print('PREDICTING FIXED MODELS ON SAVED',role.upper(),'FEATURES',flush=True)
            predictions[role]=fixed_predictions(models,table['features'])
            observed=table['original_forecasts'];original=predictions[role]['ORIGINAL']
            require(original.index.equals(observed.index) and set(original.columns)==set(observed.columns)
                    and np.allclose(original.to_numpy(dtype=float),observed.loc[:,original.columns].to_numpy(dtype=float),rtol=0,atol=1e-12),
                    'Original model predictions differ from frozen pre-release forecasts.')
            for variant,frame in predictions[role].items():csv(role+'_'+variant+'_forecasts.csv',frame)
        calibrations={'ORIGINAL':binding['original_calibration']};diagnostics={}
        for comparison in CALIBRATION_IDS:
            source=comparison if comparison in {'A1','A2','A3'} else 'ORIGINAL'
            calibration,scores,group_scores=fit_comparison_calibration(tables['calibration']['targets'],
                predictions['calibration'][source],tables['calibration']['index'],comparison)
            require(calibration['nonempty_groups']==37,'Comparison calibration group inventory differs.')
            calibrations[comparison]=calibration
            csv(comparison+'_scenario_scores.csv',scores);csv(comparison+'_group_scores.csv',group_scores)
            for role in ('calibration','policy'):
                adjusted=apply_comparison_calibration(calibration,predictions[role][source]);csv(role+'_'+comparison+'_intervals.csv',adjusted)
                if role=='calibration':diagnostics[comparison]=interval_diagnostics(tables[role]['targets'],adjusted,tables[role]['index'],calibration['active_targets'])
            print('CALIBRATED:',comparison,'MODE:',calibration['mode'],'RANK:',calibration['rank'],
                  'FALLBACK:',calibration['fallback'],'ADJUSTMENTS:',calibration['adjustments'],flush=True)
        document('calibrations.json',{'purpose':'comparison_conformal_calibrations','calibrations':calibrations,
            'interval_aliases':{'A6':'B5'},'original_N2_changed':False,'independent_coverage_validated':False})
        document('calibration_diagnostics.json',diagnostics)
        index=tables['policy']['index'];features=tables['policy']['features'];targets=tables['policy']['targets'].to_dict('index')
        candidates=candidate_policies();policies=dict(binding['original_selected_policies'])
        selections={};candidate_rows=[];all_candidate_metrics=[]
        for comparison in POLICY_VARIANTS:
            source=comparison if comparison in {'A1','A2','A3'} else 'ORIGINAL'
            family='global' if comparison=='B5' else 'adaptive'
            decisions={p['policy_id']:{} for p in candidates[family]}
            for sid in features.index:
                raw=predictions['policy'][source].loc[[sid]]
                bounds,supported=comparison_bounds(comparison,raw,calibrations)
                for policy in candidates[family]:
                    result=policy_for_active_targets(bounds,features.loc[sid].to_dict(),gates[sid],supported=supported,
                        policy=policy,unsupported_contexts=binding['unsupported_contexts'])
                    decisions[policy['policy_id']][sid]=result['decision']
                    candidate_rows.append({'comparison':comparison,'scenario_id':sid,'policy_id':policy['policy_id'],
                        'decision':result['decision'],'reason':result['reason']})
            selected=select_policy(candidates[family],decisions,targets,index.to_dict('records'),binding['policy_selection_rule'])
            selections[comparison]=selected;policies[comparison]=selected['selected_policy']
            for entry in selected['candidates']:
                metric=entry['metrics'];all_candidate_metrics.append({'comparison':comparison,'policy_id':entry['policy']['policy_id'],
                    'eligible':entry['eligible'],'auto_deploy_groups':metric['auto_deploy_groups'],
                    'equal_group_automation':metric['equal_group_automation'],
                    'equal_group_conditional_unsafe_rate':metric['equal_group_conditional_unsafe_rate'],**metric['decision_counts']})
            print('POLICY:',comparison,selected['status'],selected['selected_policy'],flush=True)
        csv('policy_candidate_signals.csv',pd.DataFrame(candidate_rows),False)
        csv('policy_candidate_metrics.csv',pd.DataFrame(all_candidate_metrics),False)
        document('policy_selections.json',{'purpose':'separately_selected_comparison_policies','selections':selections,
            'selected_policies':policies,'unsupported_contexts':binding['unsupported_contexts'],
            'evaluation':'policy_calibration_selection_not_outer_test','formal_conditional_risk_guarantee_established':False})
        development_rows=[];development_metrics={}
        for comparison in METHOD_IDS:
            source=comparison if comparison in {'A1','A2','A3'} else 'ORIGINAL';decisions={}
            for sid in features.index:
                result=decide_comparison(comparison,features.loc[sid].to_dict(),gates[sid],
                    predictions['policy'][source].loc[sid].to_dict(),calibrations,policies,binding['unsupported_contexts'])
                decisions[sid]=result['decision']
                development_rows.append({'comparison':comparison,'scenario_id':sid,'group_id':index.loc[index.scenario_id.eq(sid),'group_id'].iloc[0],
                    'split_role':'policy_calibration','decision':result['decision'],'reason':result['reason']})
            development_metrics[comparison]=policy_statistics(decisions,targets,index.to_dict('records'))
            print('DEVELOPMENT DECISIONS:',comparison,development_metrics[comparison]['decision_counts'])
        df=pd.DataFrame(development_rows)
        require(df.loc[df.comparison.eq('B8')].decision.tolist()==df.loc[df.comparison.eq('A8')].decision.tolist(),'B8/A8 equivalence failed.')
        require(development_metrics['PROPOSED']['decision_counts']=={'AUTO_DEPLOY':0,'HUMAN_REVIEW':44,'ESCALATE':0}
                and development_metrics['B8']['decision_counts']=={'AUTO_DEPLOY':0,'HUMAN_REVIEW':44,'ESCALATE':0},'Primary frozen controller decisions changed.')
        csv('development_decisions.csv',df,False)
        document('development_metrics.json',{'evaluation':'policy_fold_diagnostics_and_selection_not_independent_test','methods':development_metrics})
        document('comparison_runtime.json',{'purpose':'prepared_frozen_comparison_runtime','method_ids':list(METHOD_IDS),
            'forecaster_source':{c:c if c in {'A1','A2','A3'} else 'ORIGINAL' for c in METHOD_IDS},
            'calibrations':calibrations,'selected_policies':policies,'unsupported_contexts':binding['unsupported_contexts'],
            'equivalent_comparisons':{'A8':'B8'},'primary_controller_changed':False,'production_release_authorized':False})
        require(load_inputs(root)[-1]==binding,'Comparison inputs changed during preparation.')
        note='\n'.join(['# C4 comparison calibration and development checks','',
            'Saved A1-A3 forecasters were used without refitting. Original primary controller unchanged.',
            '150 conformal scenarios / 37 nonempty groups calibrate A1-A5 and B5; A6 reuses B5 intervals.',
            '44 policy scenarios / 11 nonempty groups select each changed policy under the frozen rule.',
            'B5/A6 lack finite Bonferroni rank support and retain full-range bounds.',
            'A4 removes the slice decision target; its unsafe outcome oracle still checks all three original targets.',
            'A1-A3 retain original hard gates and pre-release contexts; they are feature ablations only.',
            'All B1-B8/A1-A8 decision rules executed on development rows; A8 exactly reuses B8.',
            'Development/calibration diagnostics are not independent outer-test results.',
            'Comparators may make deployment signals; this does not change or validate the primary review-only policy.',
            'No TVAE fits, candidate retraining, future reevaluation or outer access occurred.',''])
        (output/'C4_COMPARISON_PREPARATION_EVIDENCE.md').write_bytes(note.encode());files['C4_COMPARISON_PREPARATION_EVIDENCE.md']=sha(note.encode())
        digest=save_json(output/'metadata.json',{'purpose':'prepared_comparison_calibrations_and_policies',
            'created_at_utc':utc_now(),**git_context(),'specification_sha256':spec_hash,
            'evaluation_specification_sha256':binding['evaluation_specification_sha256'],
            'ablation_run_metadata_sha256':binding['ablation_run_metadata_sha256'],
            'controller_freeze_sha256':binding['controller_freeze_sha256'],'calibration_rows':150,'calibration_groups':37,
            'policy_rows':44,'policy_groups':11,'method_ids':list(METHOD_IDS),'files':files,
            'source_sha256':binding['source_sha256'],'N1_refitted':False,'original_N2_modified':False,
            'outer_test_evaluated':False,'evaluation_engine_readiness_frozen':False,'release_authorized':False})
        for filename,expected in files.items():checked_bytes(output/filename,expected)
        require(checked_json(output,'comparison_runtime')[0]['calibrations']==calibrations,'Comparison JSON reload differs.')
        print('COMPARISON CALIBRATIONS / POLICIES / ALL DEVELOPMENT DECISION PATHS / CHECKSUMS: PASS')
        print('OUTPUT:',output);print('METADATA SHA256:',digest)
        print('NEXT: evaluation/statistics engine and readiness freeze; outer groups remain untouched.')
    except Exception as error:
        save_json(output/'failure.json',{'created_at_utc':utc_now(),'exception_type':type(error).__name__,
            'message':str(error),'preserve_partial_artifacts':True,'automatic_retry':False})
        raise


if __name__=='__main__':main()
