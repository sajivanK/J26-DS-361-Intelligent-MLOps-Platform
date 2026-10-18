import { classicText, isC1Alert, isClassicAlarm } from "./c1format.js";

// The comparison that shows why C1 matters: a classic monitor vs C1's harm-based answer.
export default function ClassicVsC1({ v }) {
  const alert = isC1Alert(v);
  return (
    <div className="grid2">
      <div className={`cmp ${isClassicAlarm(v) ? "alarm" : "calm"}`}>
        <span className="k">A classic drift monitor would say</span><b>{classicText(v)}</b>
      </div>
      <div className={`cmp ${alert ? "alarm" : "calm"}`}>
        <span className="k">C1 says</span>
        <b>{alert ? `Repair needed · severity ${v.sev}` : "No repair needed"} · interaction {Math.round(v.pint * 100)}%</b>
      </div>
    </div>
  );
}
