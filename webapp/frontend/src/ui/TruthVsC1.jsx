import { sg } from "./c1format.js";

// Truth vs C1 (confirmed "Option B"): two bars from zero, C1's 95% interval as a whisker on C1's bar,
// and the 1 pp line where severity starts. truth may be null (uploaded batch, no labels).
export default function TruthVsC1({ truth, harm, lo, hi }) {
  const has = truth != null;
  const W = 640, pl = 110;
  const min = Math.min(lo, has ? truth : 0, 0) - 0.6;
  const max = Math.max(hi, has ? truth : 0, 1) + 0.6;
  const X = (v) => pl + ((v - min) / (max - min)) * (W - pl - 70);
  const label = { stroke: "var(--surface)", strokeWidth: 4, paintOrder: "stroke" };
  const bar = (y, v, color) => (
    <rect x={Math.min(X(0), X(v))} y={y} width={Math.max(2, Math.abs(X(v) - X(0)))} height="20" rx="10" fill={color} opacity=".85" />
  );
  return (
    <div className="stack tight">
      <div className="scroll-x">
        <svg viewBox={`0 0 ${W} 118`} width="100%" style={{ minWidth: 460 }} role="img"
          aria-label={`Truth ${has ? sg(truth) + " pp" : "unknown"}, C1 ${sg(harm)} pp, interval ${sg(lo)} to ${sg(hi)} pp`}>
          <line x1={X(0)} x2={X(0)} y1="6" y2="92" stroke="var(--muted)" />
          <line x1={X(1)} x2={X(1)} y1="6" y2="92" stroke="var(--truth)" strokeDasharray="4 3" />
          <text x={X(1)} y="108" textAnchor="middle" fill="var(--muted)" fontSize="10">1 pp</text>

          <text x={pl - 10} y="28" textAnchor="end" fill="var(--muted)">Truth</text>
          {has ? (<>
            {bar(14, truth, "var(--truth)")}
            <circle cx={X(truth)} cy="24" r="9" fill="var(--truth)" stroke="var(--surface)" strokeWidth="3" />
            <text x={Math.max(X(truth), X(0)) + 14} y="28" fill="#B45309" fontWeight="600" {...label}>{sg(truth)} pp</text>
          </>) : <text x={X(0) + 8} y="28" fill="var(--muted)">unknown: no labels for an uploaded batch</text>}

          <text x={pl - 10} y="70" textAnchor="end" fill="var(--muted)">C1</text>
          {bar(56, harm, "var(--accent-2)")}
          <line x1={X(lo)} x2={X(hi)} y1="66" y2="66" stroke="var(--fg)" strokeWidth="1.5" />
          <line x1={X(lo)} x2={X(lo)} y1="58" y2="74" stroke="var(--fg)" strokeWidth="1.5" />
          <line x1={X(hi)} x2={X(hi)} y1="58" y2="74" stroke="var(--fg)" strokeWidth="1.5" />
          <circle cx={X(harm)} cy="66" r="8" fill="var(--accent-2)" stroke="var(--surface)" strokeWidth="3" />
          <text x={Math.max(X(harm), X(hi), X(0)) + 8} y="70" fill="var(--accent)" fontWeight="600" {...label}>{sg(harm)} pp</text>
        </svg>
      </div>
      <div className="chart-legend">
        <span><i style={{ background: "var(--truth)" }} />truth</span>
        <span><i style={{ background: "var(--accent-2)" }} />C1 predicted</span>
        <span>├─┤ C1 95% interval ({sg(lo)} to {sg(hi)} pp)</span>
        <span style={{ color: "var(--truth)" }}>┆ 1 pp, where severity starts</span>
      </div>
    </div>
  );
}
