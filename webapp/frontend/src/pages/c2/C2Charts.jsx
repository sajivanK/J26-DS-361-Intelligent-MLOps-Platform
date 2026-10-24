// Charts for the C2 page and report, v2 design (Tharmika N). Colours from tokens.css (--c2-*).
// Every chart has a text alternative (aria-label) and its numbers written next to the marks.
import { STATE, sg } from "./c2format.js";

const STEP = (max) => (max <= 1.2 ? 0.25 : max <= 3 ? 0.5 : max <= 6 ? 1 : 2);
const gridBg = (p) => `repeating-linear-gradient(90deg, transparent 0, transparent calc(${p}% - 1px), var(--c2-line-soft) calc(${p}% - 1px), var(--c2-line-soft) ${p}%)`;
const ticks = (max) => { const s = STEP(max), out = []; for (let v = 0; v <= max + 1e-9; v += s) out.push(+v.toFixed(2)); return out; };

// ── What the model relies on: normal (grey) vs this run (orange more / blue less) ─────────────────
export function RelianceChart({ rec }) {
  const ts = rec.temporal_shap, ref = ts.mean_abs_shap_ref, now = ts.mean_abs_shap_now;
  const feats = Object.keys(now).sort((a, b) => Math.max(now[b], ref[b]) - Math.max(now[a], ref[a])).slice(0, 8);
  const top = Math.max(...feats.map((f) => Math.max(now[f], ref[f])), 0.1);
  const tk = ticks(top * 1.1), max = tk[tk.length - 1] < top ? top * 1.1 : tk[tk.length - 1] + STEP(top) * 0.4;
  const P = (v) => (v / max) * 100;
  const root = rec.to_c3.root_cause_feature;
  return (
    <div className="c2-chartpanel" role="img" aria-label={`Mean absolute SHAP per feature, normal versus this run. ${feats.map((f) => `${f}: ${ref[f].toFixed(2)} normal, ${now[f].toFixed(2)} now`).join("; ")}`}>
      {feats.map((f) => {
        const a = ref[f], b = now[f], r = a ? b / a : 1, up = r > 1.05, down = r < 0.95;
        const c = up ? "var(--c2-orange)" : down ? "var(--c2-blue)" : "var(--c2-ink-3)";
        const lo = Math.min(P(a), P(b)), hi = Math.max(P(a), P(b));
        return (
          <div key={f} className="c2-crow" style={{ height: 44 }}>
            <span className="l" style={f === root ? { fontWeight: 700, color: "var(--fg)" } : null} title={f}>{f}</span>
            <div className="p" style={{ height: 44, background: gridBg((STEP(top) / max) * 100) }}>
              <span style={{ position: "absolute", top: "50%", height: 3, marginTop: -1.5, borderRadius: 2, left: `${lo}%`, width: `${hi - lo}%`, background: c, opacity: 0.45 }} />
              <span style={{ position: "absolute", top: "50%", left: `${P(a)}%`, width: 13, height: 13, margin: "-6.5px 0 0 -6.5px", borderRadius: "50%", background: "var(--c2-dot)" }} />
              <span style={{ position: "absolute", top: "50%", left: `${P(b)}%`, width: 20, height: 20, margin: "-10px 0 0 -10px", borderRadius: "50%", background: c }} />
              <span style={{ position: "absolute", top: "50%", transform: "translateY(-50%)", ...(hi > 62 ? { right: `calc(${100 - lo}% + 18px)` } : { left: `calc(${hi}% + 18px)` }), fontWeight: 600, fontSize: ".93rem", color: c, whiteSpace: "nowrap" }}>
                {up || down ? `×${r.toFixed(1)}` : "no change"}</span>
            </div>
          </div>);
      })}
      <div className="c2-crow" style={{ paddingTop: 8 }}><span />
        <div className="c2-axis">{tk.map((v) => <span key={v} style={{ left: `${P(v)}%` }}>{v.toFixed(STEP(top) < 0.5 ? 2 : 1)}</span>)}</div></div>
      <div className="c2-axisname">Mean |SHAP| value</div>
    </div>
  );
}

// ── Why this feature: Drift Impact Score split into its weighted evidence ──────────────────────────
export const SIGNALS = [["E", "Explanation change", "How the model explains the feature", "var(--c2-blue)", "var(--c2-blue-bg)"],
  ["D", "Data drift", "Distribution change in the data", "var(--c2-orange)", "var(--c2-orange-bg)"],
  ["DI", "Decision change", "How decisions shifted", "var(--c2-green)", "var(--c2-green-bg)"],
  ["PS", "Prediction shift", "Change in predicted probabilities", "var(--c2-pink)", "var(--c2-red-bg)"]];

export function EvidenceChart({ rec }) {
  const dis = rec.evidence.drift_impact_scores, sig = rec.evidence.signals, w = rec.evidence.weights;
  const feats = Object.keys(dis).slice(0, 7), root = rec.to_c3.root_cause_feature;
  return (
    <div role="img" aria-label={`Drift Impact Score per feature: ${feats.map((f) => `${f} ${dis[f].toFixed(2)}`).join(", ")}`} style={{ padding: "4px 18px 0 0" }}>
      {feats.map((f) => {
        const parts = SIGNALS.map(([k, , , col], j) => ({ k, col, v: (w[j] || 0) * (sig[f]?.[k] || 0) })).filter((p) => p.v > 0.002);
        const tot = dis[f];
        return (
          <div key={f} className="c2-crow" style={{ height: 50 }}>
            <span className="l" style={f === root ? { fontWeight: 700, color: "var(--fg)" } : null} title={f}>{f}</span>
            <div className="p c2-grid-x4" style={{ height: 50, display: "flex", alignItems: "center" }}>
              <span style={{ display: "flex", flex: "none", height: 34, gap: 2, width: `${Math.min(tot, 1) * 100}%` }}>
                {parts.map((p, i) => (
                  <span key={p.k} title={`${f} · ${SIGNALS.find((s) => s[0] === p.k)[1]}: ${p.v.toFixed(3)}`}
                    style={{ height: 34, flex: `${Math.round(p.v * 1000)} 1 0`, background: p.col, borderRadius: i === 0 ? (parts.length === 1 ? 3 : "3px 0 0 3px") : i === parts.length - 1 ? "0 3px 3px 0" : 0 }} />))}
              </span>
              <b style={{ marginLeft: 12 }}>{tot.toFixed(2)}</b>
            </div>
          </div>);
      })}
      <div className="c2-crow" style={{ paddingTop: 8 }}><span />
        <div className="c2-axis">{[0, 0.25, 0.5, 0.75, 1].map((v) => <span key={v} style={{ left: `${v * 100}%`, transform: v === 1 ? "translateX(-70%)" : undefined }}>{v === 1 ? "1.0" : v}</span>)}</div></div>
      <div className="c2-axisname">Drift Impact Score</div>
    </div>
  );
}

// ── Temporal SHAP: z per feature across stored runs (log or linear scale) ──────────────────────────
export const LINE_COLS = ["var(--c2-blue)", "var(--c2-orange)", "var(--c2-green)", "var(--c2-amber)", "var(--c2-pink)"];

export function temporalWindow(matrix, last = 12) {
  const n = matrix.runs.length, from = Math.max(0, n - last);
  const z = Object.fromEntries(Object.entries(matrix.z).map(([f, v]) => [f, v.slice(from)]));
  return { runs: matrix.runs.slice(from), times: (matrix.times || []).slice(from), z, thresh: matrix.z_thresh };
}

export function topFeatures(win, k = 5) {
  return Object.entries(win.z).map(([f, v]) => [f, Math.max(...v.map((x) => x ?? 0))]).sort((a, b) => b[1] - a[1]).slice(0, k).map((x) => x[0]);
}

export function TemporalChart({ win, feats, selected, onSelect, scale = "log", hidden = {} }) {
  const W = 900, H = 360, x0 = 110, x1 = 860, y0 = 34, y1 = 296;
  const n = win.runs.length;
  const X = (i) => (n === 1 ? (x0 + x1) / 2 : x0 + 40 + (i * (x1 - x0 - 80)) / (n - 1));
  const all = feats.flatMap((f) => win.z[f]).filter((v) => v != null);
  const zmax = Math.max(win.thresh * 1.5, ...all, 1);
  const lo = 0.1, hiL = Math.pow(10, Math.ceil(Math.log10(zmax)));
  const Y = scale === "log"
    ? (v) => y1 - ((Math.log10(Math.max(v, lo)) - Math.log10(lo)) / (Math.log10(hiL) - Math.log10(lo))) * (y1 - y0)
    : (v) => y1 - (Math.max(v, 0) / (zmax * 1.08)) * (y1 - y0);
  const grid = scale === "log" ? [0.1, 1, 10, 100, 1000].filter((v) => v <= hiL) : [0, 0.25, 0.5, 0.75, 1].map((t) => +(t * zmax).toFixed(0));
  const si = win.runs.indexOf(selected);
  const tipF = si >= 0 ? feats.filter((f) => !hidden[f]).reduce((best, f) => ((win.z[f][si] ?? -1) > (win.z[best]?.[si] ?? -1) ? f : best), feats[0]) : null;
  const tv = tipF ? win.z[tipF][si] : null;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="c2-svgchart" role="img"
      aria-label={`z per feature across runs ${win.runs.join(", ")}, ${scale} scale, alarm line ${win.thresh?.toFixed(2)}`}>
      {si >= 0 && <rect x={X(si) - 30} y={y0} width={60} height={y1 - y0} fill="var(--c2-blue-bg)" />}
      {win.runs.map((r, i) => <line key={r} x1={X(i)} x2={X(i)} y1={y0} y2={y1} stroke="var(--c2-line-soft)" />)}
      {grid.map((v) => (<g key={v}><line x1={x0} x2={x1 + 10} y1={Y(v)} y2={Y(v)} stroke="var(--c2-line)" />
        <text x={x0 - 12} y={Y(v) + 5} textAnchor="end" fontSize="15" fill="var(--c2-ink-3)">{v}</text></g>))}
      {si >= 0 && <line x1={X(si)} x2={X(si)} y1={y0} y2={y1} stroke="var(--c2-blue)" strokeOpacity=".5" strokeDasharray="4 4" />}
      <line x1={x0} x2={x1 + 10} y1={Y(win.thresh)} y2={Y(win.thresh)} stroke="var(--c2-alarm-line)" strokeWidth="2" strokeDasharray="7 5" />
      {feats.map((f, j) => hidden[f] ? null : (
        <g key={f}>
          <polyline fill="none" stroke={LINE_COLS[j]} strokeWidth="2.5" strokeLinejoin="round"
            points={win.z[f].map((v, i) => `${X(i)},${Y(v ?? 0)}`).join(" ")} />
          {win.z[f].map((v, i) => (<circle key={i} cx={X(i)} cy={Y(v ?? 0)} r={f === tipF && i === si ? 7 : 5} fill={LINE_COLS[j]}
            stroke={f === tipF && i === si ? "var(--c2-blue-tile)" : "var(--surface)"} strokeWidth={f === tipF && i === si ? 4 : 2}>
            <title>{`${f} · run #${win.runs[i]} · z ${sg(v)}`}</title></circle>))}
        </g>))}
      <g>
        <rect x={x1 - 100} y={Y(win.thresh) + 8} width="100" height="26" rx="5" fill="var(--c2-amber-bg)" stroke="var(--c2-alarm-line)" />
        <text x={x1 - 50} y={Y(win.thresh) + 26} textAnchor="middle" fontSize="14" fill="var(--c2-amber-fg)">Alarm: {win.thresh?.toFixed(2)}</text>
      </g>
      {tipF && tv != null && (() => {
        const tx = X(si) > x0 + 200 ? X(si) - 166 : X(si) + 20, ty = Math.min(Math.max(Y(tv) - 26, y0), y1 - 52);
        return (<g>
          <rect x={tx} y={ty} width="146" height="52" rx="8" fill="var(--surface)" stroke="var(--c2-blue-line)" />
          <text x={tx + 14} y={ty + 21} fontSize="14" fontWeight="600" fill="var(--c2-blue)">{tipF}</text>
          <text x={tx + 14} y={ty + 42} fontSize="14" fill="var(--fg)"><tspan fontWeight="700">{sg(tv)}</tspan> (z = {sg(tv)})</text>
        </g>);
      })()}
      {win.runs.map((r, i) => (<g key={r} onClick={() => onSelect?.(r)} style={{ cursor: onSelect ? "pointer" : "default" }}>
        <rect x={X(i) - 30} y={y1 + 4} width="60" height="30" fill="transparent" />
        <text x={X(i)} y={y1 + 24} textAnchor="middle" fontSize="15" fontWeight={r === selected ? 700 : 400} fill={r === selected ? "var(--fg)" : "var(--c2-ink-2)"}>#{r}</text></g>))}
      <text x={(x0 + x1) / 2} y={H - 6} textAnchor="middle" fontSize="14" fill="var(--c2-ink-3)">Run (chronological order)</text>
      <text x="30" y={(y0 + y1) / 2} textAnchor="middle" fontSize="14" fill="var(--c2-ink-3)" transform={`rotate(-90 30 ${(y0 + y1) / 2})`}>Explanation change (z)</text>
    </svg>
  );
}

const heatClass = (v) => (v == null ? "h0" : v < 1 ? "h0" : v < 2 ? "h1" : v < 4 ? "h2" : v < 10 ? "h3" : v < 50 ? "h4" : "h5");

export function ZHeatmap({ win, order, selected, onSelect, root }) {
  return (
    <div className="scroll-x">
      <table className="c2-heat" style={{ minWidth: 560 }}>
        <thead><tr><th className="f">Feature</th>{win.runs.map((r) => (
          <th key={r} scope="col" style={r === selected ? { color: "var(--fg)" } : null}>
            {onSelect ? <button type="button" className="c2-link" onClick={() => onSelect(r)} style={{ color: "inherit", fontWeight: r === selected ? 700 : 600 }}>#{r}</button> : `#${r}`}</th>))}</tr></thead>
        <tbody>{order.map((f) => (
          <tr key={f} className={f === root ? "sel" : ""}><td className="f">{f}</td>
            {win.z[f].map((v, i) => (<td key={i} className={`v ${heatClass(v)} ${v != null && v > win.thresh ? "on" : ""}`}
              title={`${f} · run #${win.runs[i]} · z ${sg(v)}`}>{sg(v)}</td>))}</tr>))}
        </tbody>
      </table>
    </div>
  );
}

export function HeatKey({ thresh }) {
  return (
    <div className="c2-heatkey"><span>z per feature and run:</span>
      {[["h0", "under 1"], ["h1", "1 to 2"], ["h2", "2 to 4"], ["h3", "4 to 10"], ["h4", "10 to 50"], ["h5", "50 and more"]].map(([c, l]) => (
        <span key={c}><i className={c} />{l}</span>))}
      <span><i className="on" />Above alarm line ({thresh?.toFixed(2)})</span>
    </div>
  );
}

// ── Range that caused it: where the drifted range sits on the feature's training values ─────────────
export function RangeBar({ lo, hi, refShare }) {
  // Quantile scale: the share of training data inside the range fixes where the boundary sits.
  const s = refShare == null ? 0.5 : Math.min(Math.max(refShare, 0.02), 0.98);
  const from = lo != null && hi == null ? 1 - s : 0, to = hi != null && lo == null ? s : 1;
  const mark = lo != null ? from : to;
  const val = lo != null ? lo : hi;
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
      <div className="c2-rangebar" role="img" aria-label={`Drifted range: ${lo != null ? `above ${val}` : `up to ${val}`}, about ${Math.round(s * 100)}% of training values`}>
        <span className="fill" style={{ left: `${from * 100}%`, right: `${(1 - to) * 100}%` }} />
        <span className="mark" style={{ left: `${mark * 100}%` }} />
      </div>
      <div style={{ position: "relative", height: 18, fontSize: ".8rem", color: "var(--c2-ink-3)" }}>
        <span style={{ position: "absolute", left: 0 }}>lowest</span>
        <b style={{ position: "absolute", left: `${mark * 100}%`, transform: "translateX(-50%)", color: "var(--fg)" }}>{typeof val === "number" ? +val.toFixed(4) : val}</b>
        <span style={{ position: "absolute", right: 0 }}>highest</span>
      </div>
    </div>
  );
}

export function Donut({ share, label = "of batch" }) {
  const r = 48, c = 2 * Math.PI * r, on = Math.max(0, Math.min(1, share)) * c;
  return (
    <svg viewBox="0 0 120 120" width="112" height="112" role="img" aria-label={`${Math.round(share * 100)}% ${label}`} style={{ flex: "none" }}>
      <circle cx="60" cy="60" r={r} fill="none" stroke="var(--track)" strokeWidth="14" />
      <circle cx="60" cy="60" r={r} fill="none" stroke="var(--c2-green)" strokeWidth="14" strokeLinecap="round" strokeDasharray={`${on} ${c}`} transform="rotate(-90 60 60)" />
      <text x="60" y="62" textAnchor="middle" fontSize="22" fontWeight="700" fill="var(--fg)">{Math.round(share * 100)}%</text>
      <text x="60" y="80" textAnchor="middle" fontSize="11" fill="var(--c2-ink-3)">{label}</text>
    </svg>
  );
}

export function Sparkline({ values, line, color }) {
  const v = values.map((x) => Math.max(x ?? 0, 0.1)), mx = Math.max(20, ...v);
  const Y = (z) => 22 - ((Math.log10(z) + 1) / (Math.log10(mx) + 1)) * 20;
  const pts = v.map((z, i) => `${(i * 100) / Math.max(v.length - 1, 1)},${Y(z).toFixed(1)}`).join(" ");
  return (
    <svg viewBox="0 0 100 24" width="100" height="24" aria-hidden="true">
      <line x1="0" x2="100" y1={Y(line)} y2={Y(line)} stroke="var(--c2-alarm-line)" strokeDasharray="3 2" strokeOpacity=".7" />
      <polyline points={pts} fill="none" stroke={color} strokeWidth="2" />
    </svg>
  );
}

// ── Recorded benchmark bars (Why trust it) ─────────────────────────────────────────────────────────
export function ProofBars({ chart }) {
  const max = Math.max(chart.ref || 0, ...chart.rows.map((r) => r[1])) * 1.1 || 1;
  return (
    <div className="c2-pbars">
      {chart.rows.map(([name, v]) => {
        const me = name.startsWith("C2");
        return (<div key={name} style={{ display: "contents" }}>
          <span className={`l ${me ? "me" : ""}`}>{name}</span>
          <span className="b"><span style={{ width: `${Math.max((v / max) * 100, 0.6)}%`, background: me ? "var(--accent)" : "var(--c2-blue-tile)" }} />
            <span className="c2-mono" style={{ fontSize: ".8rem" }}>{v.toFixed(1)}</span></span>
        </div>);
      })}
      {chart.ref != null && <><span className="l c2-muted">best possible</span><span className="b c2-muted" style={{ fontSize: ".8rem" }}>oracle {chart.ref.toFixed(1)}</span></>}
    </div>
  );
}

// ── Report: share of decisions changed per run, C1's predicted drop as a diamond ───────────────────
export function RunBars({ trend, current, mode = "pct" }) {
  const rows = mode === "rows" && trend.every((t) => t.decisions_changed_rows != null);
  const val = (t) => (rows ? t.decisions_changed_rows : t.decisions_changed_percent);
  const W = 640, H = 300, x0 = 70, x1 = 630, y0 = 20, y1 = 260;
  const vals = trend.flatMap((t) => (rows ? [val(t)] : [val(t), t.detection_step_predicted_points])).filter((v) => v != null);
  const top = Math.max(5, ...vals) * 1.15, raw = top / 5, mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 5, 10].map((m) => m * mag).find((s) => s >= raw) || raw;
  const Y = (v) => y1 - (Math.max(v, 0) / top) * (y1 - y0);
  const n = Math.max(trend.length, 5), slot = (x1 - x0) / n, bw = Math.min(62, slot * 0.55);
  const X = (i) => x0 + slot * (i + 0.5) - 10;
  const COL = { clean: "var(--c2-green)", harmless: "var(--c2-green)", suspected: "var(--c2-yellow)", confirmed: "var(--c2-orange)" };
  const grid = []; for (let v = 0; v <= top; v += step) grid.push(v);
  return (
    <svg viewBox={`0 0 ${W} ${H + 24}`} className="c2-svgchart" role="img"
      aria-label={`Decisions changed per run: ${trend.map((t) => `run ${t.run} ${rows ? val(t) : `${sg(val(t))}%`}`).join(", ")}`}>
      {trend.map((t, i) => t.run === current && <rect key="sel" x={X(i) - bw / 2 - 12} y={y0 - 12} width={bw + 46} height={y1 - y0 + 50} rx="8" fill="var(--c2-blue-bg)" />)}
      {grid.map((v) => (<g key={v}><line x1={x0} x2={x1} y1={Y(v)} y2={Y(v)} stroke="var(--c2-line-soft)" />
        <text x={x0 - 10} y={Y(v) + 5} textAnchor="end" fontSize="14" fill="var(--c2-ink-3)">{+v.toFixed(1)}</text></g>))}
      <line x1={x0} x2={x0} y1={y0} y2={y1} stroke="var(--c2-line)" />
      <text x="22" y={(y0 + y1) / 2} textAnchor="middle" fontSize="13" fill="var(--c2-ink-3)" transform={`rotate(-90 22 ${(y0 + y1) / 2})`}>{rows ? "Decisions changed (rows)" : "Decisions changed (%)"}</text>
      {!rows && <><line x1={x0} x2={x1} y1={Y(2)} y2={Y(2)} stroke="var(--c2-alarm-line)" strokeWidth="1.8" strokeDasharray="6 4" />
      <text x={x1} y={Y(2) - 7} textAnchor="end" fontSize="12.5" fill="var(--c2-amber-fg)">Harmful above 2%</text></>}
      {trend.map((t, i) => {
        const v = val(t) || 0, c1 = rows ? null : t.detection_step_predicted_points, h = Math.max(y1 - Y(v), 2);
        return (<g key={t.run}>
          <rect x={X(i) - bw / 2} y={y1 - h} width={bw} height={h} rx="5" fill={COL[t.state]}><title>{`run #${t.run}: ${rows ? `${v} decisions` : `${sg(v)}% of decisions`} · ${STATE[t.state].label}`}</title></rect>
          <text x={X(i)} y={y1 - h - 8} textAnchor="middle" fontSize="14" fontWeight="700" fill="var(--fg)">{rows ? v.toLocaleString() : sg(v)}</text>
          {c1 != null && <path d={`M${X(i) + bw / 2 + 14},${Y(c1) - 6} l6,6 l-6,6 l-6,-6 z`} fill="var(--c2-diamond)"><title>{`run #${t.run}: C1 predicted ${sg(c1)} pp accuracy drop`}</title></path>}
          <text x={X(i)} y={y1 + 24} textAnchor="middle" fontSize="14" fontWeight={t.run === current ? 700 : 400} fill="var(--c2-ink-2)">#{t.run}</text>
        </g>);
      })}
    </svg>
  );
}
