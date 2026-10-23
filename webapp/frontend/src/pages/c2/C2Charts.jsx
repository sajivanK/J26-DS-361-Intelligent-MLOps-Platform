// Charts for the C2 page and report, drawn as plain SVG (no chart library), colours from tokens.css.
// Hover any mark for its exact value (SVG <title>). Series colours follow the feature, not its rank.
import { clock, pct, STATE } from "./c2format.js";

const SERIES = ["var(--chart-1)", "var(--chart-2)", "var(--chart-3)", "var(--chart-4)", "var(--chart-5)"];

// ── What the model relies on: mean |SHAP| per feature, normal vs this run (dumbbell) ──────────────
export function RelianceChart({ rec }) {
  const ts = rec.temporal_shap, ref = ts.mean_abs_shap_ref, now = ts.mean_abs_shap_now;
  const root = rec.state === "clean" ? null : rec.to_c3.root_cause_feature;
  const feats = Object.keys(ref).sort((a, b) => Math.max(now[b], ref[b]) - Math.max(now[a], ref[a])).slice(0, 8);
  const max = Math.max(...feats.map((f) => Math.max(now[f], ref[f]))) * 1.1;
  const W = 520, pl = 118, pr = 54, rh = 28, H = feats.length * rh + 30;
  const x = (v) => pl + ((W - pl - pr) * v) / max;
  const step = max > 1.5 ? 0.5 : max > 0.6 ? 0.2 : 0.1;
  const ticks = [];
  for (let v = 0; v <= max + 1e-9; v += step) ticks.push(v);
  return (
    <div className="scroll-x">
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ minWidth: 420 }} role="img" aria-label="Mean absolute SHAP per feature, normal versus this run">
        {ticks.map((v) => (<g key={v}><line x1={x(v)} x2={x(v)} y1={4} y2={H - 22} stroke="var(--line)" />
          <text x={x(v)} y={H - 6} textAnchor="middle" fill="var(--muted)">{v.toFixed(step < 0.2 ? 1 : 2)}</text></g>))}
        {feats.map((f, i) => {
          const y = 16 + i * rh, a = ref[f], b = now[f], up = b > a * 1.1, down = b < a * 0.9;
          const col = up ? "var(--chart-2)" : down ? "var(--chart-1)" : "var(--muted)";
          return (
            <g key={f}>
              <title>{`${f}: normal ${a.toFixed(3)}, this run ${b.toFixed(3)} (×${(b / a).toFixed(2)})`}</title>
              <rect x={0} y={y - 12} width={W} height={24} fill="transparent" />
              <text x={pl - 10} y={y + 4} textAnchor="end" fill={f === root ? "var(--fg)" : "var(--muted)"} fontWeight={f === root ? 600 : 400}>{f}</text>
              <line x1={x(a)} x2={x(b)} y1={y} y2={y} stroke={col} strokeWidth="3" opacity=".5" strokeLinecap="round" />
              <circle cx={x(a)} cy={y} r="5" fill="var(--muted)" opacity=".45" />
              <circle cx={x(b)} cy={y} r="6.5" fill={col} stroke="var(--surface)" strokeWidth="2" />
              {(up || down) && <text x={Math.max(x(a), x(b)) + 10} y={y + 4} fill="var(--fg)" fontSize="10.5">×{(b / a).toFixed(1)}</text>}
            </g>);
        })}
      </svg>
    </div>
  );
}

// ── Why this feature: Drift Impact Score split into its weighted evidence (stacked bars) ──────────
export const SIGNALS = [["E", "explanation change", "var(--chart-1)"], ["D", "data drift", "var(--chart-2)"],
  ["DI", "decision change", "var(--chart-3)"], ["PS", "prediction shift", "var(--chart-5)"]];

export function EvidenceChart({ rec }) {
  const dis = rec.evidence.drift_impact_scores, sig = rec.evidence.signals, w = rec.evidence.weights;
  const feats = Object.keys(dis).slice(0, 6);
  const W = 520, pl = 118, pr = 48, rh = 30, H = feats.length * rh + 28;
  const x = (v) => (W - pl - pr) * v;
  return (
    <div className="scroll-x">
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ minWidth: 420 }} role="img" aria-label="Drift Impact Score per feature, split by evidence">
        {[0, 0.25, 0.5, 0.75, 1].map((v) => (<g key={v}><line x1={pl + x(v)} x2={pl + x(v)} y1={4} y2={H - 22} stroke="var(--line)" />
          <text x={pl + x(v)} y={H - 6} textAnchor="middle" fill="var(--muted)">{v}</text></g>))}
        {feats.map((f, i) => {
          const y = 8 + i * rh;
          let cur = pl;
          return (
            <g key={f}>
              <text x={pl - 10} y={y + 13} textAnchor="end" fill={i === 0 ? "var(--fg)" : "var(--muted)"} fontWeight={i === 0 ? 600 : 400}>{f}</text>
              {SIGNALS.map(([k, name, col], j) => {
                const v = (w[j] || 0) * (sig[f]?.[k] || 0), width = x(v), x0 = cur;
                cur += width;
                return width > 0.5 ? (
                  <rect key={k} x={x0} y={y} width={Math.max(width - 2, 0.5)} height="18" rx="3" fill={col}>
                    <title>{`${f} · ${name}: ${(sig[f][k] || 0).toFixed(2)} × weight ${w[j]} = ${v.toFixed(3)}`}</title>
                  </rect>) : null;
              })}
              <text x={cur + 6} y={y + 13} fill="var(--fg)" fontSize="10.5">{dis[f].toFixed(2)}</text>
            </g>);
        })}
      </svg>
    </div>
  );
}

// ── Temporal SHAP across stored runs: z per feature (log scale), alarm line ────────────────────────
export function temporalWindow(matrix, last = 12) {
  const n = matrix.runs.length, s = Math.max(0, n - last);
  const z = Object.fromEntries(Object.entries(matrix.z).map(([f, v]) => [f, v.slice(s)]));
  return { runs: matrix.runs.slice(s), times: matrix.times.slice(s), z, thresh: matrix.z_thresh };
}

export function topFeatures(win, k = 5) {
  // fixed identity for the window: strongest recent evidence (last 3 runs) first
  const score = (v) => v.slice(-3).reduce((a, b) => a + Math.max(b || 0, 0), 0);
  return Object.entries(win.z).sort((a, b) => score(b[1]) - score(a[1])).slice(0, k).map((x) => x[0]);
}

export function TemporalChart({ win, selected, onSelect }) {
  const top = topFeatures(win), n = Math.max(win.runs.length, 6), feats = Object.keys(win.z);
  const W = 780, H = 250, pl = 46, pr = 128, pt = 14, pb = 40;
  const lo = Math.log10(0.5), hi = Math.log10(Math.max(200, ...feats.flatMap((f) => win.z[f])) * 1.2);
  const x = (i) => pl + ((i + 0.5) * (W - pl - pr)) / n;
  const y = (v) => pt + (H - pt - pb) * (1 - (Math.log10(Math.max(v || 0, 0.5)) - lo) / (hi - lo));
  const last = win.runs.length - 1;
  const labels = top.map((f, k) => ({ t: f, y: y(win.z[f][last]), c: SERIES[k] }))
    .concat([{ t: `alarm ${win.thresh.toFixed(2)}`, y: y(win.thresh), c: null }]).sort((a, b) => a.y - b.y);
  for (let i = 1; i < labels.length; i++) labels[i].y = Math.max(labels[i].y, labels[i - 1].y + 13);
  return (
    <div className="scroll-x">
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ minWidth: 560 }} role="img" aria-label="Explanation change z per feature across stored runs">
        {[1, 10, 100, 1000].filter((v) => Math.log10(v) <= hi).map((v) => (<g key={v}>
          <line x1={pl} x2={W - pr} y1={y(v)} y2={y(v)} stroke="var(--line)" />
          <text x={pl - 6} y={y(v) + 4} textAnchor="end" fill="var(--muted)">{v}</text></g>))}
        {win.runs.map((r, i) => (
          <g key={r} className="c2-hit" onClick={() => onSelect(r)}>
            <title>{`Show run #${r}`}</title>
            <rect x={x(i) - 22} y={pt} width="44" height={H - pt - pb + 30} rx="6" fill={r === selected ? "var(--tint)" : "transparent"} />
            <text x={x(i)} y={H - 22} textAnchor="middle" fill={r === selected ? "var(--fg)" : "var(--muted)"} fontWeight={r === selected ? 600 : 400}>#{r}</text>
            <text x={x(i)} y={H - 8} textAnchor="middle" fill="var(--muted)" fontSize="9.5">{clock(win.times[i])}</text>
          </g>))}
        <line x1={pl} x2={W - pr} y1={y(win.thresh)} y2={y(win.thresh)} stroke="var(--warn-line)" strokeWidth="1.5" strokeDasharray="5 3" />
        {feats.filter((f) => !top.includes(f)).map((f) => (
          <polyline key={f} points={win.z[f].map((v, i) => `${x(i)},${y(v)}`).join(" ")} fill="none" stroke="var(--muted)" strokeWidth="1" opacity=".25" />))}
        {[...top].reverse().map((f) => {
          const col = SERIES[top.indexOf(f)];
          return (<g key={f}>
            <polyline points={win.z[f].map((v, i) => `${x(i)},${y(v)}`).join(" ")} fill="none" stroke={col} strokeWidth="2" strokeLinejoin="round" />
            {win.z[f].map((v, i) => (<circle key={i} cx={x(i)} cy={y(v)} r="4.5" fill={col} stroke="var(--surface)" strokeWidth="2">
              <title>{`run #${win.runs[i]} · ${f}: z ${v.toFixed(2)}`}</title></circle>))}
          </g>);
        })}
        {labels.map((l) => (<g key={l.t}>
          {l.c && <circle cx={W - pr + 6} cy={l.y} r="3.5" fill={l.c} />}
          <text x={W - pr + 14} y={l.y + 4} fill={l.c ? "var(--fg)" : "var(--warn-fg)"} fontSize="10.5">{l.t}</text></g>))}
      </svg>
    </div>
  );
}

export function TemporalLegend({ win }) {
  const top = topFeatures(win);
  return (
    <div className="chart-legend">
      {top.map((f, k) => <span key={f}><i style={{ background: SERIES[k] }} />{f}</span>)}
      <span><i style={{ background: "var(--muted)", opacity: 0.3 }} />other features</span>
      <span><i style={{ background: "var(--warn-line)" }} />alarm line</span>
    </div>
  );
}

const shade = (v) => (v < 1 ? 0 : v < 2 ? 1 : v < 4 ? 2 : v < 10 ? 3 : v < 50 ? 4 : 5);

export function ZHeatmap({ win, order, selected, onSelect }) {
  return (
    <div className="scroll-x">
      <table className="tbl c2-heat">
        <thead><tr><th>feature</th>{win.runs.map((r) => (
          <th key={r} className={r === selected ? "on" : ""}><button type="button" className="c2-colbtn" onClick={() => onSelect(r)}>#{r}</button></th>))}</tr></thead>
        <tbody>
          {order.map((f) => (
            <tr key={f}><td className="f">{f}</td>
              {win.z[f].map((v, i) => {
                const k = shade(v);
                return (<td key={i} className={`cell ${v > win.thresh ? "hot" : ""} ${win.runs[i] === selected ? "sel" : ""}`}
                  style={{ background: `var(--seq-${k})`, color: k >= 4 ? "var(--surface)" : "var(--fg)" }}
                  title={`run #${win.runs[i]} · ${f}: z ${v.toFixed(2)}`}>{v < 0 ? "−" : ""}{Math.abs(v).toFixed(1)}</td>);
              })}
            </tr>))}
        </tbody>
      </table>
    </div>
  );
}

export function HeatKey({ thresh }) {
  return (
    <div className="c2-heatkey">z per feature and run:
      {["under 1", "1 to 2", "2 to 4", "4 to 10", "10 to 50", "50 and more"].map((t, k) => (
        <span key={t}><span className="sw" style={{ background: `var(--seq-${k})` }} />{t}</span>))}
      <span>· amber outline: above the alarm line ({thresh.toFixed(2)})</span>
    </div>
  );
}

// ── Who is affected: share of the group in training data vs this batch ─────────────────────────────
export function ShareBars({ training, batch }) {
  const W = 520, H = 92, pl = 118, pr = 60, m = Math.max(batch, training || 0, 0.05) * 1.05;
  const sc = (v) => ((W - pl - pr) * v) / m;
  const rows = [["training data", training, "var(--muted)", 0.45], ["this batch", batch, "var(--chart-2)", 1]];
  return (
    <div className="scroll-x">
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ minWidth: 380, maxWidth: 640 }} role="img" aria-label="Share of the affected group in training data and in this batch">
        {rows.map(([l, v, c, o], i) => {
          const y = 10 + i * 38, w = Math.max(sc(v || 0), 2);
          return (<g key={l}>
            <text x={pl - 10} y={y + 15} textAnchor="end" fill="var(--muted)">{l}</text>
            <rect x={pl} y={y} width={w} height="22" rx="4" fill={c} opacity={o}><title>{`${l}: ${pct(v, 1)}`}</title></rect>
            <text x={pl + w + 8} y={y + 15} fill="var(--fg)">{v == null ? "–" : pct(v, 1)}</text>
          </g>);
        })}
      </svg>
    </div>
  );
}

// ── Recorded results (Why trust it) ────────────────────────────────────────────────────────────────
export function ProofBars({ chart }) {
  const max = Math.max(...chart.rows.map((r) => r[1]), chart.ref || 0, 1) * 1.08;
  const W = 460, rowH = 26, pad = 150, H = chart.rows.length * rowH + 10;
  const sc = (v) => ((W - pad - 46) * v) / max;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ maxWidth: 560 }} role="img" aria-label={chart.label}>
      {chart.rows.map(([l, v], i) => {
        const y = 4 + i * rowH, me = l.startsWith("C2");
        return (<g key={l}>
          <text x={pad - 8} y={y + 14} textAnchor="end" fill={me ? "var(--fg)" : "var(--muted)"}>{l}</text>
          <rect x={pad} y={y + 3} width={Math.max(sc(v), 2)} height="16" rx="4" fill={me ? "var(--accent-2)" : "var(--chart-1)"} opacity={me ? 1 : 0.35} />
          <text x={pad + Math.max(sc(v), 2) + 6} y={y + 15} fill="var(--fg)">{v.toFixed(1)}</text>
        </g>);
      })}
      {chart.ref != null && <line x1={pad + sc(chart.ref)} x2={pad + sc(chart.ref)} y1="0" y2={H} stroke="var(--truth)" strokeWidth="1.5" strokeDasharray="5 3" />}
    </svg>
  );
}

// ── Report: share of decisions changed per run (bars), C1's predicted drop (diamond) ───────────────
const STATE_COL = { clean: "var(--chart-3)", harmless: "var(--chart-3)", suspected: "var(--chart-4)", confirmed: "var(--chart-2)" };

export function TrendBars({ trend, current }) {
  const W = 640, H = 210, pl = 40, pr = 12, pt = 16, pb = 36, n = Math.max(trend.length, 6);
  const vals = trend.flatMap((t) => [t.decisions_changed_percent, t.detection_step_predicted_points]).filter((v) => v != null);
  const max = Math.max(4, ...vals) * 1.18;
  const x = (i) => pl + ((i + 0.5) * (W - pl - pr)) / n, bw = Math.min(46, (W - pl - pr) / n - 16);
  const y = (v) => pt + (H - pt - pb) * (1 - Math.max(v, 0) / max);
  const step = max > 20 ? 5 : max > 10 ? 4 : 2, grid = [];
  for (let v = 0; v <= max; v += step) grid.push(v);
  return (
    <div className="scroll-x">
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ minWidth: 440, maxWidth: 760 }} role="img" aria-label="Share of decisions changed per run">
        {grid.map((v) => (<g key={v}><line x1={pl} x2={W - pr} y1={y(v)} y2={y(v)} stroke="var(--line)" />
          <text x={pl - 6} y={y(v) + 4} textAnchor="end" fill="var(--muted)">{v}</text></g>))}
        <line x1={pl} x2={W - pr} y1={y(2)} y2={y(2)} stroke="var(--truth)" strokeWidth="1.5" strokeDasharray="5 3" />
        <text x={pl + 4} y={y(2) - 5} fill="var(--warn-fg)" fontSize="10">harmful above 2%</text>
        {trend.map((t, i) => {
          const v = t.decisions_changed_percent || 0, c1 = t.detection_step_predicted_points, on = t.run === current;
          return (<g key={t.run}>
            {on && <rect x={x(i) - bw / 2 - 6} y={pt - 6} width={bw + 12} height={H - pt - pb + 34} rx="6" fill="var(--tint)" />}
            <rect x={x(i) - bw / 2} y={y(v)} width={bw} height={Math.max(y(0) - y(v), 2)} rx="4" fill={STATE_COL[t.state]}>
              <title>{`run #${t.run}: ${v.toFixed(1)}% of decisions changed · ${STATE[t.state].label}`}</title></rect>
            <text x={x(i)} y={y(v) - 6} textAnchor="middle" fill="var(--fg)" fontSize="10.5">{v.toFixed(1)}</text>
            {c1 != null && (<path d={`M${x(i) + bw / 2 + 6},${y(c1) - 5} l5,5 l-5,5 l-5,-5 z`} fill="var(--fg)" opacity=".75">
              <title>{`run #${t.run}: C1 predicted ${c1.toFixed(1)} pp`}</title></path>)}
            <text x={x(i)} y={H - 16} textAnchor="middle" fill={on ? "var(--fg)" : "var(--muted)"} fontWeight={on ? 600 : 400}>#{t.run}</text>
          </g>);
        })}
      </svg>
    </div>
  );
}

export function FactorBars({ factors, root }) {
  const rows = factors.filter((f) => f.decisions_changed_percent > 0.05);
  if (!rows.length) return <p className="small muted">No single factor changes a measurable share of decisions.</p>;
  const max = Math.max(...rows.map((f) => f.decisions_changed_percent), 1) * 1.1, W = 620, pl = 230, pr = 64, rh = 30, H = rows.length * rh + 8;
  return (
    <div className="scroll-x">
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ minWidth: 420, maxWidth: 760 }} role="img" aria-label="Decisions each factor changes">
        {rows.map((f, i) => {
          const y = 4 + i * rh, w = Math.max(((W - pl - pr) * f.decisions_changed_percent) / max, 2);
          const me = f.factor === root;        // orange marks the root cause only, never just the biggest bar
          return (<g key={f.factor}>
            <text x={pl - 10} y={y + 15} textAnchor="end" fill={me ? "var(--fg)" : "var(--muted)"} fontWeight={me ? 600 : 400}>{f.factor}{me ? " (root cause)" : ""}</text>
            <rect x={pl} y={y + 3} width={w} height="18" rx="4" fill={me ? "var(--chart-2)" : "var(--chart-1)"} opacity={me ? 1 : 0.4} />
            <text x={pl + w + 8} y={y + 16} fill="var(--fg)">{f.decisions_changed_percent.toFixed(1)}%</text>
          </g>);
        })}
      </svg>
    </div>
  );
}

