import { sg } from "./c1format.js";

// One number line: C1's 95% interval (band), its prediction (dot) and, if known, the true harm (diamond).
export default function IntervalLine({ harm, lo, hi, truth }) {
  const has = truth != null;
  const vals = has ? [lo, hi, truth] : [lo, hi];
  const min = Math.min(-4, Math.floor(Math.min(...vals) - 1));
  const max = Math.max(6, Math.ceil(Math.max(...vals) + 1));
  const W = 600, pad = 24;
  const x = (v) => pad + ((v - min) / (max - min)) * (W - 2 * pad);
  const step = max - min > 30 ? 5 : 2;
  const ticks = [];
  for (let v = Math.ceil(min / step) * step; v < max; v += step) ticks.push(v);
  return (
    <div className="scroll-x">
      <svg viewBox={`0 0 ${W} 94`} width="100%" style={{ minWidth: 420 }} role="img"
        aria-label={`Interval ${sg(lo)} to ${sg(hi)} pp, prediction ${sg(harm)}${has ? `, truth ${sg(truth)}` : ""}`}>
        <line x1={pad} x2={W - pad} y1="68" y2="68" stroke="var(--line)" strokeWidth="2" />
        {ticks.map((v) => (
          <g key={v}>
            <line x1={x(v)} x2={x(v)} y1="66" y2="71" stroke="var(--muted)" />
            <text x={x(v)} y="86" textAnchor="middle" fill="var(--muted)">{v}</text>
          </g>
        ))}
        <line x1={x(0)} x2={x(0)} y1="6" y2="68" stroke="var(--line)" strokeDasharray="3 3" />
        <rect x={x(lo)} y="22" width={Math.max(2, x(hi) - x(lo))} height="26" rx="5" fill="var(--tint)" stroke="var(--accent)" strokeWidth="1.5" />
        <circle cx={x(harm)} cy="35" r="6" fill="var(--accent)" />
        {has && (
          <rect x={x(truth) - 6} y="29" width="12" height="12" transform={`rotate(45 ${x(truth)} 35)`}
            fill="var(--truth)" stroke="var(--fg)" strokeWidth="1.2" />
        )}
        <text x={x(harm)} y="15" textAnchor="middle" fill="var(--accent)">C1 {sg(harm)}</text>
        <text x={W - pad} y="86" textAnchor="end" fill="var(--muted)">pp</text>
      </svg>
    </div>
  );
}
