import {useState} from "react";
import {c4Api} from "../../api.js";
import {useApi} from "../../hooks.js";
import {ErrorBox} from "../../ui/parts.jsx";

const targets=[
 ["future_global_margin","Global performance"],
 ["future_slice_margin","True injected slice"],
 ["future_nonaffected_margin","True slice complement"],
];
const decimal=(v,digits=4)=>typeof v==="number"&&Number.isFinite(v)?v.toFixed(digits):"Undefined";
const percent=v=>typeof v==="number"&&Number.isFinite(v)?`${(v*100).toFixed(2)}%`:"Undefined";

export default function C4HybridResearch(){
 const result=useApi(c4Api.hybridResults,[]);
 const [target,setTarget]=useState("future_global_margin");
 const [family,setFamily]=useState("adaptive");
 const data=result.data;
 if(!data)return <div className="c4-research-content">
  <p className="small muted">Revised N1 results are separate from the recorded case and original PP1 benchmark.</p>
  <ErrorBox error={result.error?"Revised research results could not be loaded. Check the C4 worker and its saved research artifacts. The recorded-case assessment is separate.":null}/>
  {result.loading?<p role="status" className="small muted">Loading verified research results…</p>:<button type="button" className="btn" onClick={()=>result.reload()}>Retry research results</button>}
 </div>;
 const counts=data.counts;
 const methods=data.methods.map(method=>({...method,metric:method.metrics.find(row=>row.target===target)}));
 return <div className="c4-research-content">
  <div className="c4-between"><div><h3>Revised N1 · Adult scenario-group evaluation</h3><p className="small muted">Saved experimental results · separate from the recorded case above</p></div><span className="c4-source">Verified saved evidence</span></div>
  <div className="c4-research-stats">
   <div><span className="c4-label">Candidates evaluated</span><strong>{counts.admitted_cases}</strong><span className="small muted">{counts.blocked_cases} admission-blocked / {counts.planned_cases} planned</span></div>
   <div><span className="c4-label">Groups with evaluated candidates</span><strong>{counts.nonempty_groups} / {counts.assigned_groups}</strong><span className="small muted">{counts.all_blocked_groups} groups fully blocked</span></div>
   <div><span className="c4-label">Selected release policy</span><strong>Review only</strong><span className="small muted">No empirical autonomous deployments</span></div>
  </div>
  <div className="c4-note">The hybrid is the predeclared experimental primary. Its additional global forecasting gain over same-training LightGBM is modest; slice and complement heads are shared.</div>
  <div className="c4-runbar c4-research-controls"><label>Forecast population<select aria-label="Research target population" value={target} onChange={event=>setTarget(event.target.value)}>{targets.map(([key,label])=><option key={key} value={key}>{label}</option>)}</select></label></div>
  <h3>Forecast performance</h3>
  <p className="small muted">Lower MAE, RMSE and pinball loss are better; higher R² is better. Errors are in accuracy percentage points (pp), not classification accuracy percentages.</p>
  <div className="c4-scroll"><table className="c4-table"><caption className="c4-sr-only">Revised forecast errors for {targets.find(([key])=>key===target)?.[1]}</caption><thead><tr><th scope="col">Forecaster</th><th scope="col">MAE ↓ (pp)</th><th scope="col">RMSE ↓ (pp)</th><th scope="col">R² ↑</th><th scope="col">Mean pinball ↓ (pp)</th></tr></thead><tbody>{methods.map(method=><tr key={method.id} className={method.is_primary?"c4-primary-row":undefined}><th scope="row">{method.label}{method.is_primary&&<span className="c4-primary-label">Primary</span>}</th><td>{decimal(method.metric.mae_pp)}</td><td>{decimal(method.metric.rmse_pp)}</td><td>{decimal(method.metric.r2)}</td><td>{decimal(method.metric.mean_pinball_pp)}</td></tr>)}</tbody></table></div>
  <h3>Joint coverage and interval width</h3>
  <p className="small muted">Joint coverage checks all three forecast populations together. Width and interval score below refer to the selected population and use equal group weights.</p>
  <div className="c4-scroll"><table className="c4-table"><thead><tr><th scope="col">Forecaster</th><th scope="col">Scenario joint coverage</th><th scope="col">Whole-group joint coverage</th><th scope="col">Full interval width (pp)</th><th scope="col">Interval score ↓ (pp)</th></tr></thead><tbody>{methods.map(method=><tr key={method.id} className={method.is_primary?"c4-primary-row":undefined}><th scope="row">{method.label}</th><td>{percent(method.coverage.scenario_joint)}</td><td>{percent(method.coverage.whole_group_joint)}</td><td>{decimal(method.metric.width_pp)}</td><td>{decimal(method.metric.interval_score_pp)}</td></tr>)}</tbody></table></div>
  <div className="c4-research-limit"><strong>Interpretation limits</strong><p className="small">The {counts.cohort_rows.toLocaleString()}-row external Adult cohort was also used during development. This evaluates held-out scenario groups, not an independent dataset. Observed coverage does not establish a release guarantee; the nominal calibration target is {percent(data.nominal_coverage)}. Complement forecasting remains weak.</p></div>
  <div className="c4-runbar c4-research-controls"><label>Saved policy family<select aria-label="Research policy family" value={family} onChange={event=>setFamily(event.target.value)}><option value="adaptive">Evidence-adaptive</option><option value="global">Global</option></select></label></div>
  <h3>Frozen-policy decisions</h3>
  <p className="small muted">Decisions on the {counts.admitted_cases} admitted cases. The {counts.blocked_cases} admission-blocked cases are counted separately; they have no future safety labels.</p>
  <div className="c4-scroll"><table className="c4-table"><thead><tr><th scope="col">Forecaster</th><th scope="col">AUTO_DEPLOY</th><th scope="col">HUMAN_REVIEW</th><th scope="col">ESCALATE</th><th scope="col">Automation</th><th scope="col">Conditional unsafe deployment rate</th></tr></thead><tbody>{methods.map(method=>{const policy=method.policies[family];return <tr key={method.id} className={method.is_primary?"c4-primary-row":undefined}><th scope="row">{method.label}</th><td>{policy.decision_counts.AUTO_DEPLOY}</td><td>{policy.decision_counts.HUMAN_REVIEW}</td><td>{policy.decision_counts.ESCALATE}</td><td>{percent(policy.automation)}</td><td>{policy.decision_counts.AUTO_DEPLOY===0?"Undefined · no deployments":percent(policy.conditional_unsafe_rate)}</td></tr>;})}</tbody></table></div>
  <p className="small muted">Policy calibration did not support LOW or HIGH contexts. Evidence quality is a heuristic, not a failure probability. Production release remains unauthorized.</p>
  <details className="c4-details"><summary>Revised research provenance</summary><dl className="c4-evidence-fields"><div><dt>Experiment version</dt><dd>{data.version}</dd></div>{Object.entries(data.provenance).map(([key,value])=><div key={key}><dt>{key.replaceAll("_"," ")}</dt><dd className="mono">{value}</dd></div>)}</dl></details>
 </div>;
}
