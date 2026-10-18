// Charts for the C1 page, drawn as plain SVG (no chart library).
import { isC1Alert, isClassicAlarm } from "../../ui/c1format.js";

// Session history: predicted harm (dot) and 95% interval (bar) per batch, with classic vs C1 lanes.
export function HistoryChart({ runs, selected, onSelect }) {
  const B = runs.slice(-8), off = runs.length - B.length, n = Math.max(B.length, 5);
  const W = 880, H = 250, pl = 60, pr = 12, pt = 14, ph = 150;
  const lo = Math.min(-7, ...B.map((b) => Math.floor(b.v.lo)));
  const hi = Math.max(8, ...B.map((b) => Math.ceil(b.v.hi)));
  const x = (i) => pl + ((i + 0.5) * (W - pl - pr)) / n;
  const y = (v) => pt + ph * (1 - (v - lo) / (hi - lo));
  const step = hi - lo > 40 ? 10 : 4;
  const grid = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi; v += step) grid.push(v);
  const ya = pt + ph + 48, yc = ya + 22;
  const pill = (cx, cy, label) => (<g><rect x={cx - 22} y={cy - 7} width="44" height="14" rx="7" fill="var(--warn-line)" /><text x={cx} y={cy + 4} textAnchor="middle" fill="#3B2504" fontSize="9.5">{label}</text></g>);
  return (
    <div className="scroll-x">
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ minWidth: 600 }} role="img" aria-label="Predicted harm per batch with 95% intervals">
        {grid.map((v) => (<g key={v}><line x1={pl} x2={W - pr} y1={y(v)} y2={y(v)} stroke="var(--line)" /><text x={pl - 6} y={y(v) + 4} textAnchor="end" fill="var(--muted)">{v}</text></g>))}
        <line x1={pl} x2={W - pr} y1={y(1)} y2={y(1)} stroke="var(--truth)" strokeDasharray="5 3" strokeWidth="1.5" />
        {B.map((b, i) => {
          const idx = off + i, on = idx === selected;
          const pick = () => onSelect(idx);
          return (
            <g key={idx} className="pt" tabIndex="0" role="button" aria-label={`Batch ${idx + 1}`} onClick={pick}
              onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pick(); } }}>
              <rect x={x(i) - 40} y={pt} width="80" height={ph + 30} fill={on ? "var(--tint)" : "transparent"} rx="6" />
              <line x1={x(i)} x2={x(i)} y1={y(b.v.lo)} y2={y(b.v.hi)} stroke="var(--accent-2)" strokeWidth="3" opacity=".45" strokeLinecap="round" />
              <circle cx={x(i)} cy={y(b.v.harm)} r={on ? 7 : 5.5} fill="var(--accent)" stroke="var(--surface)" strokeWidth="2" />
              <text x={x(i)} y={pt + ph + 16} textAnchor="middle" fill={on ? "var(--fg)" : "var(--muted)"}>#{idx + 1}</text>
              <text x={x(i)} y={pt + ph + 28} textAnchor="middle" fill="var(--muted)" fontSize="9.5">{b.short}</text>
            </g>);
        })}
        <text x={pl - 6} y={ya + 4} textAnchor="end" fill="var(--muted)" fontSize="10">classic</text>
        <text x={pl - 6} y={yc + 4} textAnchor="end" fill="var(--muted)" fontSize="10">C1</text>
        {B.map((b, i) => (
          <g key={"l" + i}>
            {isClassicAlarm(b.v) ? pill(x(i), ya, "ALARM") : <circle cx={x(i)} cy={ya} r="3" fill="var(--line)" />}
            {isC1Alert(b.v) ? pill(x(i), yc, "ALERT") : <circle cx={x(i)} cy={yc} r="3" fill="var(--line)" />}
          </g>))}
      </svg>
    </div>
  );
}

// Novelty 1: harm error by method (horizontal bars, C1 highlighted).
export function ErrorBars({ rows }) {
  const max = Math.max(...rows.map((r) => r[1])) * 1.05;
  const W = 420, rowH = 25, pad = 92, H = rows.length * rowH + 8;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ maxWidth: 560 }} role="img" aria-label="Harm prediction error by method">
      {rows.map(([l, v], i) => {
        const yy = 4 + i * rowH, w = ((W - pad - 48) * v) / max, hiL = l.startsWith("C1");
        return (<g key={l}>
          <text x={pad - 8} y={yy + 14} textAnchor="end" fill="var(--muted)">{l}</text>
          <rect x={pad} y={yy + 3} width={w} height="15" rx="3" fill={hiL ? "var(--accent-2)" : "var(--muted)"} opacity={hiL ? 1 : 0.5} />
          <text x={pad + w + 6} y={yy + 14} fill="var(--fg)">{v.toFixed(2)}</text>
        </g>);
      })}
    </svg>
  );
}

// Novelty 2: coverage by drift type, one shared margin (grey) vs per-extent margins (teal).
export function CoverageBars({ cats, a, b }) {
  const lo = 0.85, hi = 1, W = 440, H = 200, pl = 34, pb = 40, ptp = 8, cw = (W - pl - 6) / cats.length;
  const y = (v) => ptp + (H - ptp - pb) * (1 - (v - lo) / (hi - lo));
  const ticks = [0.85, 0.88, 0.91, 0.94, 0.97, 1.0];
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ maxWidth: 600 }} role="img" aria-label="Interval coverage by drift type">
      {ticks.map((v) => (<g key={v}><line x1={pl} x2={W - 6} y1={y(v)} y2={y(v)} stroke="var(--line)" /><text x={pl - 4} y={y(v) + 4} textAnchor="end" fill="var(--muted)">{v.toFixed(2)}</text></g>))}
      {cats.map((c, i) => {
        const x0 = pl + i * cw + cw * 0.18, bw = cw * 0.3, cx = pl + i * cw + cw / 2;
        return (<g key={c}>
          <rect x={x0} y={y(a[i])} width={bw} height={y(lo) - y(a[i])} fill="var(--muted)" opacity=".5" />
          <rect x={x0 + bw + 2} y={y(b[i])} width={bw} height={y(lo) - y(b[i])} fill="var(--accent-2)" />
          <text x={cx} y={H - 20} textAnchor="middle" fill="var(--muted)">{c}</text>
        </g>);
      })}
      <line x1={pl} x2={W - 6} y1={y(0.95)} y2={y(0.95)} stroke="var(--truth)" strokeWidth="1.5" strokeDasharray="5 3" />
    </svg>
  );
}

// Novelty 3: drift extent confusion matrix.
export function ConfusionMatrix({ labels, matrix }) {
  return (
    <svg viewBox="0 0 300 150" width="100%" style={{ maxWidth: 340 }} role="img" aria-label="Extent confusion matrix">
      {labels.map((l, j) => <text key={"c" + l} x={138 + j * 62} y="12" textAnchor="middle" fill="var(--muted)">{l}</text>)}
      {matrix.map((r, i) => (
        <g key={i}>
          <text x="96" y={41 + i * 40} textAnchor="end" fill="var(--muted)">{labels[i]}</text>
          {r.map((v, j) => (<g key={j}>
            <rect x={110 + j * 62} y={18 + i * 40} width="58" height="36" rx="4" fill="var(--accent-2)" opacity={0.08 + v * 0.85} />
            <text x={139 + j * 62} y={41 + i * 40} textAnchor="middle" fill={v > 0.5 ? "var(--surface)" : "var(--fg)"}>{v.toFixed(3)}</text>
          </g>))}
        </g>))}
    </svg>
  );
}
