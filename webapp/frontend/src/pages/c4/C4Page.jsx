import {useState} from "react";
import {c4Api} from "../../api.js";
import {useApi,useAction} from "../../hooks.js";
import {useSession} from "../../session.jsx";
import {PageHeader,ErrorBox,Empty,KV} from "../../ui/parts.jsx";
import {SafetyChart} from "./C4Charts.jsx";
import C4HybridResearch from "./C4HybridResearch.jsx";
import "./C4Page.css";

const labels={future_global_margin:"Global performance",future_slice_margin:"True injected slice",future_nonaffected_margin:"True slice complement"};
const methodLabels={hybrid:"Hybrid · Ridge / LightGBM / LightGBM",lightgbm:"Quantile LightGBM",linear:"Linear quantile regression",constant:"Constant quantiles"};
const gateLabels={repair_approved:"Repair approved",privacy_check:"Privacy screen",diagnosis_stability:"Diagnosis stability",schema_compatible:"Schema compatibility",rollback_ready:"Rollback readiness"};
const number=v=>typeof v==="number"&&Number.isFinite(v);
const value=v=>v==null?"Unavailable":typeof v==="boolean"?(v?"Yes":"No"):Array.isArray(v)?(v.length?v.join(", "):"None listed"):typeof v==="object"?JSON.stringify(v):String(v);
const percent=v=>number(v)?`${(100*v).toFixed(2)}%`:"Unavailable";
const pp=v=>number(v)?`${v>0?"+":""}${(100*v).toFixed(3)} pp`:"Unavailable";
const short=v=>number(v)?v.toFixed(3):"Unavailable";
function Details({title,children}){return <details className="c4-details"><summary>{title}</summary><div>{children}</div></details>;}
function Fields({rows}){return <dl className="c4-evidence-fields">{rows.map(([k,v])=><div key={k}><dt>{k}</dt><dd>{value(v)}</dd></div>)}</dl>;}
function Raw({data}){return <pre className="c4-technical-json">{JSON.stringify(data,null,2)}</pre>;}
function reason(r){
 if(r.failed_gates?.length)return "A mandatory safety check failed or is unknown. Release requires investigation.";
 if(r.n3?.release_signal==="ESCALATE")return "A complete calibrated interval lies below its required safety boundary.";
 if(r.n3?.release_signal==="HUMAN_REVIEW"){
  const n=Object.values(r.n3?.dimensions||{}).filter(d=>d.status==="UNCERTAIN").length;
  return `Future safety is uncertain in ${n} of 3 populations.${r.selected_policy_kind==="review_only"?" The selected policy also requires human review.":""}`;
 }
 if(r.selected_policy_kind==="review_only")return "The selected policy requires human review even when interval safety is established.";
 return r.decision==="AUTO_DEPLOY"?"The evidence meets the configured release eligibility requirements.":value(r.reason);
}

function c4ErrorMessage(error){
 if(!error)return null;
 const message=String(error?.message??error);
 if(/cannot reach|failed to fetch|networkerror|network request failed/i.test(message)){
  return "Unable to reach the C4 backend. Check that the server is running on port 8000, then reload this page and retry.";
 }
 if(/^503\b/.test(message)){
  return "C4 model or evidence artifacts are unavailable. Check the backend configuration and artifact paths. No new decision was produced.";
 }
 if(/^5\d\d\b/.test(message)){
  return "The C4 request could not be completed. Check that the backend is running and inspect its logs. No new decision was produced.";
 }
 return message;
}

export default function C4Page(){
 const info=useApi(c4Api.info,[]);
 const [caseId,setCaseId]=useSession("c4:selected-case-v3","revised-outer-case");
 const [method,setMethod]=useSession("c4:selected-method-v3","hybrid"),[family,setFamily]=useSession("c4:selected-family-v3","adaptive");
 const [saved,setSaved]=useSession("c4:case-assessment-v3",null);
 const revised=caseId==="revised-outer-case";
 const [evaluate,state]=useAction(revised?c4Api.revisedEvaluate:c4Api.evaluate);
 function changeCase(next){setCaseId(next);setMethod(next==="revised-outer-case"?"hybrid":"lightgbm");setSaved(null);state.setError(null);}
 async function run(){const result=await evaluate({case_id:caseId,method,family});if(result)setSaved(result);}
 const selectionChanged=saved&&(saved.result.method!==method||saved.result.family!==family);
 return <div className="c4-page">
  <PageHeader eyebrow="Component 4 · Kesigan M" title="C4 Release Control" subtitle="Assess whether the repaired candidate is ready for release."/>
  <section className="c4-card">
   <div className="c4-runbar"><label>Case<select aria-label="Recorded case" value={caseId} disabled={state.busy} onChange={e=>changeCase(e.target.value)}><option value="revised-outer-case">Revised Adult case · interaction / low / label-corrupted repair</option><option value="real-outer-case">Original PP1 Adult case · interaction / medium</option></select></label>
    <button className="btn primary" type="button" disabled={state.busy||!info.data} onClick={run}>{state.busy?"Evaluating…":"Evaluate recorded case"}</button></div>
   <p className="small"><span className="c4-source">{revised?"Revised recorded replay · simulated upstream":"Original PP1 recorded case · simulated upstream"}</span></p>
   <Details title="Advanced settings"><div className="c4-runbar">
    <label>Forecaster<select aria-label="Case forecaster" value={method} disabled={state.busy} onChange={e=>setMethod(e.target.value)}>{(revised?["hybrid","lightgbm","linear","constant"]:["lightgbm","linear"]).map(m=><option value={m} key={m}>{methodLabels[m]}</option>)}</select></label>
    <label>Policy family<select aria-label="Case policy family" value={family} disabled={state.busy} onChange={e=>setFamily(e.target.value)}><option value="adaptive">Evidence-adaptive</option><option value="global">Global</option></select></label>
   </div><p className="small muted">Changes apply on the next evaluation. {revised?"Replays saved pre-future predictions, calibrated intervals and policy decisions; no new model inference.":"Uses the original frozen controller with recorded case evidence."} No retraining or deployment runs.</p></Details>
  </section>
  <ErrorBox error={c4ErrorMessage(state.error||info.error)}/>
  {saved&&(state.busy||state.error||info.error)&&<p className="small muted" role="status">Previous successful result shown below. {state.busy?"A new evaluation is in progress.":"The latest request failed; this is not a new decision."}</p>}
  {selectionChanged&&<p className="small muted" role="status">Showing the previous evaluation settings. Evaluate again to apply the selected forecaster and policy family.</p>}
  {saved?<Assessment payload={saved}/>:<Empty>Evaluate the recorded case to see candidate performance, future-safety intervals and the release decision.</Empty>}
  <SupportingResearch/>
  <ControllerVerification/>
 </div>;
}
function Assessment({payload:p}){
 const replay=p.mode==="verified_revised_recorded_replay";
 const r=p.result,d=p.case_details||{},up=d.upstream_evidence||{},c1=up.c1_evidence||{},c2=up.c2_diagnosis||{},c3=up.c3_repair_package||{};
 const dims=r.n3?.dimensions||{},safe=Object.values(dims).filter(v=>v.status==="SAFE").length;
 const passed=Object.values(p.hard_gates||{}).filter(v=>v===true).length;
 const reviewOnly=r.selected_policy_kind==="review_only";
 const n3Uncertain=r.n3?.release_signal==="HUMAN_REVIEW";
 const block=r.failed_gates?.length?"Mandatory check failed":n3Uncertain?(reviewOnly?"Safety uncertain; review-only policy":"Safety uncertain"):reviewOnly?"Review-only policy":r.n3?.release_signal==="ESCALATE"?"Unsafe evidence":"None";
 const comparisons=p.candidate_evaluation||{};
 const populationRows=[["global","Global"],["affected_slice","Diagnosed slice"],["non_affected","Diagnosed-slice complement"]];
 const title={HUMAN_REVIEW:"HUMAN REVIEW REQUIRED",ESCALATE:"ESCALATION REQUIRED",AUTO_DEPLOY:"AUTO-DEPLOY ELIGIBLE"}[r.decision]||r.decision;
 const observedMissing=!Object.keys(comparisons).length;
 return <>
  {replay&&<p className="small muted" role="status">Verified revised recorded replay · saved predictions and decisions from before future evaluation. No new inference, training or deployment.</p>}
  <section className={`c4-verdict c4-${r.decision.toLowerCase()}`} role="status" aria-live="polite"><h2>{title}</h2><p>{reason(r)}</p><p className="c4-next-action">Next action: {r.decision==="ESCALATE"?"Investigate the failed checks or unsafe evidence before proceeding.":r.decision==="AUTO_DEPLOY"?"Eligibility signal only. Production deployment is not connected.":"Review the candidate before release."}</p></section>
  <section className="c4-kpis">{[["Hard gates",`${passed} / ${Object.keys(p.hard_gates||{}).length} passed`],["Future safety",(r.N2_executed||r.N2_replayed)?`${safe} / 3 established`:"Not assessed"],["Release policy",reviewOnly?"Review only":value(r.selected_policy_kind)]].map(([k,v])=><div className="c4-card c4-kpi" key={k}><span className="c4-label">{k}</span><strong>{v}</strong></div>)}</section>
  <section className="c4-card"><h2>Case information</h2><div className="c4-case-meta">
   {[["Case ID",r.evidence_id],["Dataset","UCI Adult"],["Assessment source",replay?"Revised pre-future recorded replay":"Original PP1 frozen controller"],["Champion",d.champion_model_id],["Candidate",d.candidate_model_id],["Forecaster used",methodLabels[r.method]||r.method],["Policy family used",r.family]].map(([k,v])=><div key={k}><span className="c4-label">{k}</span><strong>{value(v)}</strong></div>)}</div></section>
  <section className="c4-card"><h2>Observed pre-release performance</h2><p className="small muted">Measured before the release decision on the diagnosed populations.</p>
   {observedMissing?<Empty>Detailed recorded evidence is unavailable. Re-evaluate after the backend update.</Empty>:<>
    <div className="c4-scroll"><table className="c4-table"><thead><tr><th>Population</th><th>Evaluation rows</th><th>Champion accuracy</th><th>Candidate accuracy</th><th>Change</th></tr></thead><tbody>{populationRows.map(([key,name])=>{const row=comparisons[key]||{};return <tr key={key}><td>{name}</td><td>{value(row.rows)}</td><td>{percent(row.champion?.accuracy)}</td><td>{percent(row.candidate?.accuracy)}</td><td className="mono">{pp(row.candidate_minus_champion?.accuracy)}</td></tr>;})}</tbody></table></div>
    <Details title="Additional metrics"><div className="c4-scroll"><table className="c4-table"><thead><tr><th>Population</th><th>Champion macro-F1</th><th>Candidate macro-F1</th><th>Change</th><th>Champion AUC</th><th>Candidate AUC</th></tr></thead><tbody>{populationRows.map(([key,name])=>{const row=comparisons[key]||{};return <tr key={key}><td>{name}</td><td>{short(row.champion?.macro_f1)}</td><td>{short(row.candidate?.macro_f1)}</td><td>{pp(row.candidate_minus_champion?.macro_f1)}</td><td>{short(row.champion?.auc)}</td><td>{short(row.candidate?.auc)}</td></tr>;})}</tbody></table></div></Details>
   </>}
  </section>
  <section className="c4-card"><h2>Future-safety assessment</h2><p className="small muted">Forecast candidate-minus-champion accuracy differences. pp = percentage points.</p>
   {Object.entries(dims).map(([key,dimension])=><SafetyChart key={key} name={labels[key]||key} dimension={dimension} median={r.n1_forecasts?.[key+"__median"]}/>)}
   {!Object.keys(dims).length&&<Empty>Future safety was not assessed because a mandatory gate stopped the controller.</Empty>}
   <p className="small muted">Benchmark scope: future slice forecasts use the true injected slice and its complement. The C2 diagnosed slice supplies pre-release evidence and is a different population.</p>
  </section>
  <section><h2>Upstream and repair evidence</h2><div className="c4-grid3">
   <article className="c4-card"><h3>C1 · Drift evidence</h3><Fields rows={[["Predicted harm (raw recorded value)",c1.predicted_harm],["Harm metric",c1.harm_metric],["Harm severity",c1.harm_severity],["Affected features",c1.affected_features]]}/></article>
   <article className="c4-card"><h3>C2 · Diagnosis</h3><Fields rows={[["Diagnosed slice rule (processed features)",c2.affected_slice_rule],["Slice support",percent(c2.slice_support)],["Diagnosis stability",short(c2.diagnosis_stability)]]}/></article>
   <article className="c4-card"><h3>C3 · Repair</h3><Fields rows={[["Recorded package status",c3.repair_package_status],["Synthetic rows",c3.n_synthetic_rows],["Fidelity",short(c3.fidelity_score)],["Simulation privacy screen",c3.privacy_status]]}/></article>
  </div><p className="small muted">Inputs shown are from the recorded simulation. Package approval and privacy-screen PASS are simulation evidence, not live production approval or privacy certification.</p>
  <Details title="View input details"><Raw data={up}/></Details></section>
  <div className="c4-columns"><section className="c4-card"><h2>Safety checks</h2>{Object.entries(p.hard_gates||{}).map(([key,v])=><div className="c4-gate-label" key={key}><span>{gateLabels[key]||key}</span><strong className={v===true?"c4-pass":"c4-fail"}>{v===true?"PASS":v===false?"FAIL":"UNKNOWN"}</strong></div>)}<p className="small muted">Rollback readiness refers to verified offline artifact restoration.</p></section>
   <section className="c4-card"><h2>Release policy</h2><Fields rows={[["Evidence context",r.evidence_context?.context],["Evidence quality",short(r.evidence_context?.quality)],["Auto-release eligible",r.decision==="AUTO_DEPLOY"],["Selected policy",reviewOnly?"Review only":r.selected_policy_id],["Blocking reason",block]]}/><p className="small muted">Quality is a heuristic score, not a calibrated safety probability.</p></section></div>
  <Details title="How C4 reached this decision"><Fields rows={[["N1 · Forecasting",`${r.N1_replayed?"Saved predictions replayed":r.N1_executed?"Executed":"Not executed"} · ${methodLabels[r.method]||r.method}`],["N2 · Joint uncertainty",(r.N2_executed||r.N2_replayed)?`${r.N2_replayed?"Saved intervals replayed · ":""}Adjustment ${pp(p.n2_calibration?.adjustment)} · N2-v2`:"Not executed"],["N3 · Population safety",Object.values(dims).map(x=>x.status).join(" / ")||"Not assessed"],["N4 · Release policy",reviewOnly?"Review-only fallback":r.selected_policy_id]]}/><p className="small muted">N4 cannot override a failed gate or uncertain safety assessment. Production deployment is not connected.</p></Details>
  <Details title="Audit and technical details"><Fields rows={[["Case ID",r.evidence_id],["Recorded pre-release freeze",d.pre_release_recorded_at],[p.controller_freeze_label||"Controller freeze SHA256",p.controller_freeze_sha256],["Input file SHA256",p.input_file_sha256],["Pre-release evidence SHA256",d.pre_release_evidence_sha256],["Upstream record SHA256",d.upstream_simulation_sha256],["Evidence reconstruction verified",d.displayed_evidence_reconstructed],["New model inference executed",p.inference_executed],["Future targets used",r.future_targets_used],["Production release authorized",r.production_release_authorized]]}/>{p.provenance&&<Details title="Revised calibration and policy provenance"><Fields rows={Object.entries(p.provenance)}/></Details>}<Details title="Raw response JSON"><Raw data={p}/></Details></Details>
 </>;
}
function SupportingResearch(){
 const [open,setOpen]=useState(false),[version,setVersion]=useState("hybrid");
 return <details className="c4-details" onToggle={e=>setOpen(e.currentTarget.open)}><summary>Research findings · supporting evidence</summary>{open&&<div>
  <div className="c4-tabs" role="group" aria-label="Research evidence version"><button type="button" aria-pressed={version==="hybrid"} onClick={()=>setVersion("hybrid")}>Revised N1 comparison</button><button type="button" aria-pressed={version==="original"} onClick={()=>setVersion("original")}>Original PP1 benchmark</button></div>
  {version==="hybrid"?<C4HybridResearch/>:<ResearchSummary/>}
 </div>}</details>;
}
function ResearchSummary(){
 const result=useApi(c4Api.results,[]),d=result.data;
 if(!d)return <><ErrorBox error={c4ErrorMessage(result.error)}/><p className="small muted">{result.loading?"Loading…":"Research evidence unavailable."}</p></>;
 const primary=d.governance.find(x=>x.comparison==="PROPOSED"),coverage=d.intervals.PROPOSED;
 return <><p className="small muted">Held-out Adult simulation results, separate from the selected case.</p><KV rows={[["Assigned outer groups",d.counts.assigned_groups],["Admitted / blocked scenarios",`${d.counts.admitted} / ${d.counts.blocked}`],["AUTO_DEPLOY / HUMAN_REVIEW / ESCALATE",`${primary.count_auto} / ${primary.count_review} / ${primary.count_escalate}`],["Scenario joint coverage",percent(coverage.scenario_simultaneous_coverage)],["Whole-group joint coverage",percent(coverage.group_all_variant_simultaneous_coverage)]]}/><p className="small">Nominal calibration target: 95%. Observed whole-group coverage was 85.71%; the pilot does not establish a 95% held-out group-level guarantee. With zero deployments, conditional deployment risk is undefined.</p></>;
}
function ControllerVerification(){
 const [verify,state]=useAction(c4Api.fixtures),[results,setResults]=useState(null);
 async function run(){const out=await verify();if(out)setResults(out);}
 return <Details title="Controller verification · synthetic technical checks"><p className="small muted">Not research performance evidence. These fixtures run the actual controller class with synthetic forecasters and explicit test policies.</p><button className="btn" type="button" disabled={state.busy} onClick={run}>{state.busy?"Checking…":"Run acceptance checks"}</button><ErrorBox error={c4ErrorMessage(state.error)}/>{results?.fixtures.map(f=><div className="c4-fixture c4-between" key={f.fixture_id}><span>{f.fixture_id}</span><strong className={f.passed?"c4-pass":"c4-fail"}>{f.passed?"PASS":"FAIL"} · {f.actual_decision}</strong></div>)}</Details>;
}
