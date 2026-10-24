import {useState} from "react";
const pp = v => `${(v * 100).toFixed(3)} pp`;
export function SafetyChart({name,dimension,median}){
 const [active,setActive]=useState(null);
 const {lower:lo,upper:hi,boundary,status}=dimension;
 const min=Math.min(-.03,lo-.004,boundary-.004),max=Math.max(.015,hi+.004,boundary+.004);
 const pos=v=>100*(v-min)/(max-min);
 const color={SAFE:"#0f766e",UNCERTAIN:"#b45309",UNSAFE:"#be123c"}[status]||"#64748b";
 const hasMedian=Number.isFinite(median);
 function interactions(id,label,reading,position){
  const item={id,label,reading,position};
  return {onPointerEnter:event=>{if(event.pointerType!=="touch")setActive(item);},onPointerLeave:event=>{if(event.pointerType!=="touch")setActive(null);},onFocus:()=>setActive(item),onBlur:()=>setActive(null),onClick:()=>setActive(item)};
 }
 const points=[["lower","Lower bound",lo],["upper","Upper bound",hi],...(hasMedian?[["median","Forecast median",median]]:[]),["boundary","Safety boundary",boundary]];
 return <article className="c4-interval c4-interactive-interval" style={{"--c4-band-color":color}}>
  <div className="c4-between"><h3>{name}</h3><span className={`c4-status c4-${status.toLowerCase()}`}>{status}</span></div>
  <div className="c4-chart-numbers"><span>Forecast change <b>{hasMedian?pp(median):"Unavailable"}</b></span><span>Interval <b>{pp(lo)} to {pp(hi)}</b></span><span>Safety boundary <b>{pp(boundary)}</b></span></div>
  <div className="c4-margin-plot c4-point-plot" onKeyDown={e=>{if(e.key==="Escape")setActive(null);}}>
   <div className="c4-plot-axis"/>
   <button type="button" className="c4-quantile-band c4-band-target" style={{left:`${pos(lo)}%`,width:`${pos(hi)-pos(lo)}%`}} aria-label={`${name} calibrated interval: ${pp(lo)} to ${pp(hi)}`} {...interactions("interval","Calibrated interval",`${pp(lo)} to ${pp(hi)}`,(pos(lo)+pos(hi))/2)}/>
   {points.map(([id,label,v])=><button key={id} type="button" className={`c4-chart-point c4-point-${id}`} style={{left:`${pos(v)}%`}} aria-label={`${name} ${label}: ${pp(v)}`} {...interactions(id,label,pp(v),pos(v))}>{id==="boundary"&&<span>Boundary</span>}</button>)}
   <div className="c4-axis-labels"><span>{pp(min)}</span><span>{pp(max)}</span></div>
   {active&&<div className="c4-point-tooltip" role="status" style={{left:`${Math.max(28,Math.min(72,active.position))}%`}}><strong>{active.label}</strong><span>{active.reading}</span><button type="button" aria-label="Close chart value" onClick={()=>setActive(null)}>×</button></div>}
  </div>
  <p className="small muted">{status==="SAFE"?"The interval remains above the required safety boundary.":status==="UNSAFE"?"The interval remains below the required safety boundary.":"The interval crosses the safety boundary; review is required."}</p>
 </article>;
}

export function DecisionBars({rows}) {
  return <div className="c4-bars">{rows.map(r=><div key={r.comparison} className="c4-bar-row">
    <span className="mono">{r.comparison}</span>
    <div className="c4-bar" role="img" aria-label={`${r.comparison}: ${r.count_auto} deploy, ${r.count_review} review, ${r.count_escalate} escalate`}>
      {[["auto",r.count_auto],["review",r.count_review],["escalate",r.count_escalate]].map(([k,n])=>Number(n)>0&&<span key={k} className={`c4-bar-${k}`} style={{width:`${100*Number(n)/Number(r.count_n)}%`}}>{n}</span>)}
    </div>
  </div>)}</div>;
}

export function PerformanceChart({rows}){
 return <div className="c4-performance-chart"><div className="c4-chart-legend"><span><i className="c4-champion-dot"/>Champion</span><span><i className="c4-candidate-dot"/>Candidate</span><span>Accuracy · fixed 0–100% scale</span></div>{rows.map(row=>{
 const valid=v=>typeof v==="number"&&Number.isFinite(v)&&v>=0&&v<=1;
 return <div className="c4-performance-row" key={row.name}><strong>{row.name}</strong>{[["Champion",row.champion?.accuracy,"champion"],["Candidate",row.candidate?.accuracy,"candidate"]].map(([label,v,kind])=><div className="c4-performance-series" key={kind}><span>{label}</span><div className="c4-performance-track" role="img" aria-label={`${row.name} ${label}: ${valid(v)?(v*100).toFixed(2)+"%":"unavailable"}`}><div className={`c4-performance-fill c4-${kind}-fill`} style={{width:valid(v)?`${v*100}%`:"0%"}}/></div><b>{valid(v)?`${(v*100).toFixed(2)}%`:"Unavailable"}</b></div>)}</div>;
 })}</div>;
}
