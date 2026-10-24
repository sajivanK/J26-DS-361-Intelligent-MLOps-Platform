import C4LiveAssessment from "./C4LiveAssessment.jsx";
import {useState,useRef} from "react";
import {Link} from "react-router-dom";
import {api,c4Api} from "../../api.js";
import {useApi,useAction} from "../../hooks.js";
import {PageHeader,ErrorBox,Empty} from "../../ui/parts.jsx";
import {SafetyChart,PerformanceChart} from "./C4Charts.jsx";
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
  return "Unable to reach the backend. Check port 8000 and the C4 worker on port 8004. No new decision was produced.";
 }
 if(/^503\b/.test(message)){
  return "C4 model or evidence artifacts are unavailable. Check the backend configuration and artifact paths. No new decision was produced.";
 }
 if(/^5\d\d\b/.test(message)){
  return "The C4 request could not be completed. Check that the backend is running and inspect its logs. No new decision was produced.";
 }
 return message;
}

async function workspaceRequest(path,body){
 const response=await fetch(`/api/c4/workspace${path}`,body?{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)}:{});
 const out=await response.json().catch(()=>null);
 if(!response.ok)throw new Error(`${response.status}: ${typeof out?.detail==="string"?out.detail:"C4 workspace request failed"}`);
 return out;
}
const historyList=()=>workspaceRequest("/history?limit=100");
const recordedAssessment=()=>workspaceRequest("/evaluate",{request_id:crypto.randomUUID()});

export default function C4Page(){
 const info=useApi(c4Api.info,[]),health=useApi(api.health,[]);
 const [mode,setMode]=useState("research"),[saved,setSaved]=useState(null);
 const [received,setReceived]=useState(null);
 const history=useApi(historyList,[]);
 const [selectedHistory,setSelectedHistory]=useState(null);
 const historyRequest=useRef(0);
 const [evaluate,state]=useAction(recordedAssessment);
 const [openEntry,openState]=useAction(id=>workspaceRequest(`/history/${encodeURIComponent(id)}`));
 function closeHistory(){historyRequest.current+=1;setSelectedHistory(null);openState.setError(null);}
 async function openHistory(id){
  const request=++historyRequest.current;
  setSelectedHistory(null);
  const entry=await openEntry(id);
  if(entry&&request===historyRequest.current){setSelectedHistory(entry);}
 }
 async function run(){
  const response=await evaluate();
  if(!response)return;
  const out=response.assessment;
  if(!out){state.setError("Assessment response is missing.");return;}
  const r=out.result;
  if(out.mode!=="verified_revised_recorded_replay"||r?.method!=="hybrid"||r?.family!=="adaptive"||
     !["HUMAN_REVIEW","ESCALATE","AUTO_DEPLOY"].includes(r?.decision)||
     r.production_release_authorized!==false||r.future_targets_used!==false){
   state.setError("Unexpected assessment response. No new decision was accepted.");return;
  }
  setSaved(out);setReceived(response.entry.recorded_at);history.reload();
 }
 return <div className="c4-page">
  <PageHeader eyebrow="Component 4 · Kesigan M" title="C4 Release Control" subtitle="Assess the repaired candidate, inspect population safety, and determine the next release action." chip="Adult · hybrid controller"/>
  <section className="c4-card c4-mode-card">
   <div className="c4-tabs" role="group" aria-label="Evaluation mode">
    <button type="button" aria-pressed={mode==="research"} onClick={()=>{closeHistory();setMode("research");}}>Research simulation</button>
    <button type="button" aria-pressed={mode==="integrated"} onClick={()=>{closeHistory();setMode("integrated");}}>Integrated evaluation</button>
    <button type="button" aria-pressed={mode==="history"} onClick={()=>{closeHistory();setMode("history");}}>C4 assessment history</button>
   </div>
   <p className="small muted">{mode==="history"?"Saved C4 assessment events and evidence snapshots.":mode==="research"?"Inspect a frozen benchmark assessment with simulated upstream evidence.":""}</p>
  </section>
  {mode==="integrated"?<Integration health={health}/>:mode==="history"?<><History history={history} onOpen={openHistory} onClose={closeHistory} busy={openState.busy}/><ErrorBox error={c4ErrorMessage(openState.error)}/>{selectedHistory&&<><div className="c4-note c4-between"><span>Saved assessment · {new Date(selectedHistory.entry.recorded_at).toLocaleString()} · recorded simulation replay</span><button className="btn" type="button" onClick={closeHistory}>Close assessment</button></div><Assessment key={selectedHistory.entry.id} payload={selectedHistory.assessment}/></>}</>:<>
   <section className="c4-card c4-run-card">
    <div className="c4-runbar"><div><h2>Recorded Adult assessment</h2><p>Interaction drift · low severity · broadened slice · label-corrupted repair</p><span className="c4-source">Recorded replay · simulated upstream</span></div>
     <button className="btn primary" type="button" disabled={state.busy||info.loading||!info.data} onClick={run}>{state.busy?"Loading assessment…":"Evaluate recorded case"}</button>
    </div>
    <p className="small muted">Hybrid: Ridge / LightGBM / LightGBM · Evidence-adaptive policy. Saved evidence only; no new training, inference or deployment.</p>
    {info.error&&<button className="btn" type="button" disabled={info.loading||state.busy} onClick={info.reload}>Retry connection</button>}
   </section>
   <ErrorBox error={c4ErrorMessage(state.error||info.error)}/>
   {saved&&(state.busy||state.error||info.error)&&<div className="c4-note" role="status">Previous successful result below. {state.busy?"A refresh is in progress.":"The latest request or connection failed; this is not a new decision."}</div>}
   {!saved&&<ControllerVerification/>}
   {saved?<><p className="small muted">Response received in this browser: {new Date(received).toLocaleString()}. The underlying assessment is recorded.</p><Assessment payload={saved}/></>:<Empty>Load the recorded assessment to inspect the decision, population intervals, safety gates and evidence trail.</Empty>}
  </>}
 </div>;
}
function History({history,onOpen,onClose,busy}){
 const [filter,setFilter]=useState("ALL");
 const entries=history.data?.entries||[];
 const rows=entries.filter(row=>filter==="ALL"||row.decision===filter);
 return <section className="c4-card">
  <div className="c4-between"><div><h2>C4 assessment history</h2><p className="muted">Server-saved recorded assessments. Replaying a case creates an assessment event, not a new repair or independent validation case.</p></div><button className="btn" disabled={history.loading} onClick={()=>{onClose();history.reload();}}>Refresh history</button></div>
  <ErrorBox error={c4ErrorMessage(history.error)}/>
  <div className="c4-history-summary"><span><b>{history.data?.total??"—"}</b> saved events</span><span><b>{history.data?.unique_cases??"—"}</b> distinct cases</span><span className="c4-source">Recorded simulations</span></div>
  <label className="c4-filter">Decision <select value={filter} onChange={e=>{onClose();setFilter(e.target.value);}}><option value="ALL">All decisions</option><option value="HUMAN_REVIEW">Human review</option><option value="ESCALATE">Escalate</option><option value="AUTO_DEPLOY">Release eligible</option></select></label>
  <p className="small muted">Showing the latest 100 events. Evidence snapshots are rechecked when opened. Times use your browser timezone.</p>
  {history.loading?<p role="status">Loading history…</p>:rows.length?<div className="c4-scroll"><table className="c4-table"><thead><tr><th>Recorded at</th><th>Case</th><th>Decision</th><th>Source</th><th>Evidence</th></tr></thead><tbody>{rows.map(row=><tr key={row.id}><td>{new Date(row.recorded_at).toLocaleString()}</td><td className="c4-history-case">{row.case_id}</td><td><span className={`c4-status ${row.decision==="HUMAN_REVIEW"?"c4-uncertain":row.decision==="ESCALATE"?"c4-unsafe":"c4-safe"}`}>{row.decision.replaceAll("_"," ")}</span></td><td>Recorded replay</td><td><button className="btn" disabled={busy} onClick={()=>onOpen(row.id)}>Open assessment</button></td></tr>)}</tbody></table></div>:!history.error&&<Empty>No saved assessments match this decision filter.</Empty>}
 </section>;
}
function Integration({health}){
 return <C4LiveAssessment health={health}/>;
}
function Assessment({payload:p}){
 const [section,setSection]=useState("overview");
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
  {replay&&<p className="small muted" role="status">Verified recorded replay · saved predictions and decisions from before future evaluation. No new inference, training or deployment.</p>}
  <section className={`c4-verdict c4-${r.decision.toLowerCase()}`} role="status" aria-live="polite"><h2>{title}</h2><p>{reason(r)}</p><p className="c4-next-action">Next action: {r.decision==="ESCALATE"?"Investigate the failed checks or unsafe evidence before proceeding.":r.decision==="AUTO_DEPLOY"?"Eligibility signal only. Production deployment is not connected.":"Review the candidate before release."}</p></section>
  <section className="c4-kpis">{[["Hard gates",`${passed} / ${Object.keys(p.hard_gates||{}).length} passed`],["Future safety",(r.N2_executed||r.N2_replayed)?`${safe} / 3 meet boundary`:"Not assessed"],["Release policy",reviewOnly?"Review only":value(r.selected_policy_kind)]].map(([k,v])=><div className="c4-card c4-kpi" key={k}><span className="c4-label">{k}</span><strong>{v}</strong></div>)}</section>
  <nav className="c4-section-nav" aria-label="Assessment sections">
   {[["overview","Overview"],["performance","Performance"],["evidence","Evidence & policy"],["audit","Audit & verification"]].map(([id,name])=><button key={id} type="button" aria-pressed={section===id} onClick={()=>setSection(id)}>{name}</button>)}
  </nav>
  <div className="c4-assessment-panel" hidden={section!=="overview"}>
  <section className="c4-card c4-case-card"><div className="c4-between"><h2>Case information</h2><span className="c4-source">Verified recorded simulation</span></div><p>Interaction drift · low severity · broadened slice · label-corrupted repair</p><div className="c4-case-meta">
   {[["Case ID",r.evidence_id],["Dataset","UCI Adult"],["Assessment source",replay?"Recorded pre-future assessment":"Frozen controller"],["Champion",d.champion_model_id],["Candidate",d.candidate_model_id],["Forecaster used",methodLabels[r.method]||r.method],["Policy family used",r.family]].map(([k,v])=><div key={k}><span className="c4-label">{k}</span><strong>{value(v)}</strong></div>)}</div></section>
  <section className="c4-card"><h2>Future-safety assessment</h2><p className="small muted">Forecast candidate-minus-champion accuracy differences. pp = percentage points.</p>
   <div className="c4-safety-grid">{Object.entries(labels).map(([key,label])=>{const dimension=dims[key];return dimension&&[dimension.lower,dimension.upper,dimension.boundary].every(number)&&dimension.lower<=dimension.upper&&["SAFE","UNSAFE","UNCERTAIN"].includes(dimension.status)?<SafetyChart key={key} name={label} dimension={dimension} median={r.n1_forecasts?.[key+"__median"]}/>:<p key={key}>{label}: valid interval unavailable.</p>;})}
   </div>{!Object.keys(dims).length&&<Empty>Future safety was not assessed because a mandatory gate stopped the controller.</Empty>}
   <p className="small muted">Benchmark scope: future slice forecasts use the true injected slice and its complement. The C2 diagnosed slice supplies pre-release evidence and is a different population.</p>
  </section>
  <section className="c4-card"><h2>How C4 reached this decision</h2><div className="c4-stage-grid">
   <article className="c4-stage-n1"><h3><span className="c4-stage-code">N1</span>Forecasting</h3><p className="small muted">Saved forecast change versus champion</p><Fields rows={Object.entries(labels).map(([target,label])=>{const v=r.n1_forecasts?.[target+"__median"];return [label,!number(v)?"Unavailable":v===0?"No predicted change":`${(Math.abs(v)*100).toFixed(3)} pp ${v<0?"lower":"higher"} than champion`];})}/><p className="small muted">These are predicted differences, not absolute accuracy.</p></article>
   <article className="c4-stage-n2"><h3><span className="c4-stage-code">N2</span>Joint uncertainty</h3><p className="small muted">Calibrated uncertainty for all three populations</p><Fields rows={[["Saved adjustment",pp(p.n2_calibration?.adjustment)],["Population intervals",`${Object.keys(dims).length} saved intervals`],["Evidence source",r.N2_replayed?"Verified recorded intervals":"Replay not established"]]}/></article>
   <article className="c4-stage-n3"><h3><span className="c4-stage-code">N3</span>Population safety</h3><p className="small muted">{safe} of 3 populations meet their boundaries</p><Fields rows={Object.entries(labels).map(([target,label])=>[label,dims[target]?.status||"Unavailable"])}/></article>
   <article className="c4-stage-n4"><h3><span className="c4-stage-code">N4</span>Release policy</h3><p className="small muted">{reviewOnly?"Human review required by saved policy":value(r.selected_policy_id)}</p><Fields rows={[["Evidence context",r.evidence_context?.context],["Hard gates",`${passed} / ${Object.keys(p.hard_gates||{}).length} passed`],["Final decision",r.decision],["Reason",block]]}/></article>
  </div><p className="small muted">N4 cannot override failed gates or uncertain safety. Evidence is from the saved assessment. Production deployment is not connected.</p></section>
  </div>
  <div className="c4-assessment-panel" hidden={section!=="performance"}>
  <section className="c4-card"><h2>Observed pre-release performance</h2><p className="small muted">Measured classification performance on the diagnosed populations before the release decision; not forecasting accuracy.</p>
   {observedMissing?<Empty>Detailed recorded evidence is unavailable. Re-evaluate after the backend update.</Empty>:<>
    <PerformanceChart rows={populationRows.map(([key,name])=>({name,...comparisons[key]}))}/><div className="c4-scroll"><table className="c4-table"><thead><tr><th>Population</th><th>Evaluation rows</th><th>Champion accuracy</th><th>Candidate accuracy</th><th>Change</th></tr></thead><tbody>{populationRows.map(([key,name])=>{const row=comparisons[key]||{};return <tr key={key}><td>{name}</td><td>{value(row.rows)}</td><td>{percent(row.champion?.accuracy)}</td><td>{percent(row.candidate?.accuracy)}</td><td className="mono">{pp(row.candidate_minus_champion?.accuracy)}</td></tr>;})}</tbody></table></div>
    <Details title="Additional metrics"><div className="c4-scroll"><table className="c4-table"><thead><tr><th>Population</th><th>Champion macro-F1</th><th>Candidate macro-F1</th><th>Change</th><th>Champion AUC</th><th>Candidate AUC</th></tr></thead><tbody>{populationRows.map(([key,name])=>{const row=comparisons[key]||{};return <tr key={key}><td>{name}</td><td>{short(row.champion?.macro_f1)}</td><td>{short(row.candidate?.macro_f1)}</td><td>{pp(row.candidate_minus_champion?.macro_f1)}</td><td>{short(row.champion?.auc)}</td><td>{short(row.candidate?.auc)}</td></tr>;})}</tbody></table></div></Details>
   </>}
  </section>
  </div>
  <div className="c4-assessment-panel" hidden={section!=="evidence"}>
  <section><h2>Upstream and repair evidence</h2><div className="c4-grid3">
   <article className="c4-card c4-tone-blue"><h3>C1 · Drift evidence</h3><Fields rows={[["Predicted harm (raw recorded value)",c1.predicted_harm],["Harm metric",c1.harm_metric],["Harm severity",c1.harm_severity],["Affected features",c1.affected_features]]}/></article>
   <article className="c4-card c4-tone-purple"><h3>C2 · Diagnosis</h3><Fields rows={[["Diagnosed slice rule (processed features)",c2.affected_slice_rule],["Slice support",percent(c2.slice_support)],["Diagnosis stability",short(c2.diagnosis_stability)]]}/></article>
   <article className="c4-card c4-tone-teal"><h3>C3 · Repair</h3><Fields rows={[["Recorded package status",c3.repair_package_status],["Synthetic rows",c3.n_synthetic_rows],["Fidelity",short(c3.fidelity_score)],["Simulation privacy screen",c3.privacy_status]]}/></article>
  </div><p className="small muted">Inputs shown are from the recorded simulation. Package approval and privacy-screen PASS are simulation evidence, not live production approval or privacy certification.</p>
  <Details title="View input details"><Raw data={up}/></Details></section>
  <div className="c4-columns"><section className="c4-card"><h2>Safety checks</h2>{Object.entries(p.hard_gates||{}).map(([key,v])=><div className="c4-gate-label" key={key}><span>{gateLabels[key]||key}</span><strong className={v===true?"c4-pass":"c4-fail"}>{v===true?"PASS":v===false?"FAIL":"UNKNOWN"}</strong></div>)}<p className="small muted">Rollback readiness refers to verified offline artifact restoration.</p></section>
   <section className="c4-card"><h2>Release policy</h2><Fields rows={[["Evidence context",r.evidence_context?.context],["Auto-release eligible",r.decision==="AUTO_DEPLOY"],["Selected policy",reviewOnly?"Review only":r.selected_policy_id],["Blocking reason",block]]}/><Details title="Evidence-quality details"><Fields rows={[["Heuristic score",short(r.evidence_context?.quality)]]}/><p className="small muted">Weights: stability 0.35; harm-interval quality 0.25; fidelity 0.25; utility quality 0.15. This is not accuracy or a calibrated safety probability.</p></Details></section></div>
  </div>
  <div className="c4-assessment-panel" hidden={section!=="audit"}>
  <ModelsValidation/>
  <Details title="Audit and technical details"><Fields rows={[["Case ID",r.evidence_id],["Recorded pre-release freeze",d.pre_release_recorded_at],[p.controller_freeze_label||"Controller freeze SHA256",p.controller_freeze_sha256],["Input file SHA256",p.input_file_sha256],["Pre-release evidence SHA256",d.pre_release_evidence_sha256],["Upstream record SHA256",d.upstream_simulation_sha256],["Evidence reconstruction verified",d.displayed_evidence_reconstructed],["New model inference executed",p.inference_executed],["Future targets used",r.future_targets_used],["Production release authorized",r.production_release_authorized]]}/>{p.provenance&&<Details title="Calibration and policy provenance"><Fields rows={Object.entries(p.provenance)}/></Details>}<Details title="Raw response JSON"><Raw data={p}/></Details></Details>
  <ControllerVerification/>
  </div>
 </>;
}
function ModelsValidation(){
 const [evidence,setEvidence]=useState(null);
 const [load,state]=useAction(c4Api.hybridResults);
 const inventory=[
  ["hybrid_ridge_lightgbm_lightgbm","Active hybrid","Ridge + residual bands","Quantile LightGBM","Quantile LightGBM"],
  ["same_external_training_lightgbm","LightGBM baseline","Quantile LightGBM","Quantile LightGBM","Quantile LightGBM"],
  ["same_external_training_linear","Linear quantile baseline","Linear quantile regression","Linear quantile regression","Linear quantile regression"],
  ["same_external_training_constant_quantiles","Constant baseline","Constant quantiles","Constant quantiles","Constant quantiles"],
 ];
 async function verify(){
  setEvidence(null);
  const out=await load();
  if(!out)return;
  if(out.evidence_verified!==true||out.version!=="adult-hybrid-fresh-validation-v1"||
     out.production_release_authorized!==false||!Array.isArray(out.methods)||
     out.methods.length!==inventory.length||new Set(out.methods.map(m=>m.id)).size!==inventory.length||
     !inventory.every(([id])=>out.methods.some(m=>m.id===id))){
   state.setError("Comparison evidence does not match the expected four-method study.");return;
  }
  setEvidence(out);
 }
 return <Details title="Models and validation">
  <p><strong>Active N1: a target-specific hybrid.</strong> Ridge predicts the global margin; separate quantile LightGBM models predict slice and complement margins. The two LightGBM entries represent different targets.</p>
  <div className="c4-scroll"><table className="c4-table"><thead><tr><th>Forecaster</th><th>Global margin</th><th>Slice margin</th><th>Complement margin</th></tr></thead><tbody>{inventory.map(([id,name,global,slice,complement])=><tr key={id} className={id.startsWith("hybrid_")?"c4-primary-row":undefined}><td>{name}</td><td>{global}</td><td>{slice}</td><td>{complement}</td></tr>)}</tbody></table></div>
  <p className="small muted">Ridge is regularized linear regression, not linear quantile regression. Its raw global bands use training-residual quantiles. The linear quantile baseline fits separate conditional quantile regressions. N2 calibrates each forecaster's intervals separately.</p>
  <Fields rows={[["Development comparison","Group-separated model comparisons with training-only preprocessing and group weighting."],["Fresh validation design","Separate conformal-calibration, policy-selection and outer-test scenario groups."],["Validation scope","UCI Adult scenario groups evaluated on a shared external Adult cohort; not independent cross-dataset validation."],["Baseline role","Experimental comparisons, not additional models combined into the current hybrid decision."]]}/>
  <div className="c4-actions"><button className="btn" type="button" disabled={state.busy} onClick={verify}>{state.busy?"Verifying saved evidence…":"Verify saved comparison evidence"}</button></div>
  <ErrorBox error={c4ErrorMessage(state.error)}/>
  {evidence&&<div className="c4-note" role="status"><strong>Four-method comparison evidence verified</strong><p>Study: {evidence.version}. This is a read-only lookup of saved research results; no models were fitted or selected.</p><a href="/api/c4/research-results/hybrid" target="_blank" rel="noreferrer">Open full saved comparison JSON</a><p className="small muted">Model names and successful evidence verification do not establish model superiority or production-release authorization. Performance claims require the corresponding measured comparison.</p></div>}
 </Details>;
}

function ControllerVerification(){
 const [verify,state]=useAction(c4Api.fixtures),[results,setResults]=useState(null);
 async function run(){const out=await verify();if(out)setResults(out);}
 return <Details title="Controller verification · synthetic technical checks"><p className="small muted">Not research performance evidence. These fixtures run the actual controller class with synthetic forecasters and explicit test policies.</p><button className="btn" type="button" disabled={state.busy} onClick={run}>{state.busy?"Checking…":"Run acceptance checks"}</button><ErrorBox error={c4ErrorMessage(state.error)}/>{results&&(state.busy||state.error)&&<p role="status">Previous verification result shown; no new successful check yet.</p>}{results?.fixtures?.map(f=><div className="c4-fixture c4-between" key={f.fixture_id}><span>{f.fixture_id}</span><strong className={f.passed?"c4-pass":"c4-fail"}>{f.passed?"PASS":"FAIL"} · {f.actual_decision}</strong></div>)}</Details>;
}
