const pp = v => `${(v * 100).toFixed(3)} pp`;
export function SafetyChart({name, dimension, median}) {
  const {lower:lo,upper:hi,boundary,status} = dimension;
  const min=Math.min(-.03,lo-.004,boundary-.004), max=Math.max(.015,hi+.004,boundary+.004);
  const x=v=>30+(v-min)/(max-min)*640;
  const hasMedian=Number.isFinite(median);
  return <article className="c4-interval">
    <div className="c4-between"><h3>{name}</h3><span className={`c4-status c4-${status.toLowerCase()}`}>{status}</span></div>
    <div className="c4-chart-numbers"><span>Median <b>{hasMedian?pp(median):"Unavailable"}</b></span><span>Interval <b>{pp(lo)} to {pp(hi)}</b></span><span>Boundary <b>{pp(boundary)}</b></span></div>
    <svg viewBox="0 0 700 88" role="img" aria-label={`${name}: ${pp(lo)} to ${pp(hi)}; boundary ${pp(boundary)}; ${status}`}>
      <title>{name} calibrated future accuracy margin</title>
      <line x1="30" x2="670" y1="48" y2="48" stroke="var(--line)" strokeWidth="2"/>
      <line x1={x(boundary)} x2={x(boundary)} y1="5" y2="55" stroke="var(--muted)" strokeDasharray="4 4"/>
      <rect x={x(lo)} y="20" width={Math.max(1,x(hi)-x(lo))} height="23" rx="5" fill="var(--tint)" stroke="var(--accent)"/>
      {hasMedian&&<circle cx={x(median)} cy="31.5" r="5" fill="var(--accent)"/>}
      {[min,0,max].map((v,i)=><text key={i} x={x(v)} y="77" textAnchor="middle" fill="var(--muted)">{pp(v)}</text>)}
    </svg>
    <p className="small muted">{status==="SAFE"?"The full interval meets this population's safety requirement.":status==="UNSAFE"?"The full interval lies below this population's safety boundary.":"The interval includes outcomes below the required boundary; safety is uncertain."}</p>
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
