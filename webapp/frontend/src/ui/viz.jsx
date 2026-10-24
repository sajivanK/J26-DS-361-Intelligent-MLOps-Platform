// Small charts for tiles and cards (v0.4). They always show REAL data: the recent saved runs
// (useRecent below) or this session's runs. Each chart has a short caption saying what it shows.
import { useApi } from "../hooks.js";
import { api } from "../api.js";
import { Icon } from "../layout/icons.jsx";

// Sparkline: values oldest -> newest.
export function Spark({ values, color = "var(--accent-2)", w = 110, h = 30, caption }) {
  const v = (values || []).filter((x) => x != null && !Number.isNaN(x));
  if (v.length < 2) return null;
  const lo = Math.min(...v), hi = Math.max(...v), r = hi - lo || 1;
  const pts = v.map((x, i) => [2 + (i * (w - 4)) / (v.length - 1), h - 4 - ((x - lo) / r) * (h - 8)]);
  const d = "M" + pts.map(([x, y]) => `${x.toFixed(1)} ${y.toFixed(1)}`).join(" L");
  const [lx, ly] = pts[pts.length - 1];
  return (
    <span className="spark">
      <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} aria-hidden="true">
        <path d={`${d} L${lx.toFixed(1)} ${h} L2 ${h} Z`} fill={color} opacity=".12" />
        <path d={d} fill="none" stroke={color} strokeWidth="2" strokeLinejoin="round" strokeLinecap="round" />
        <circle cx={lx} cy={ly} r="2.8" fill={color} />
      </svg>
      {caption && <small>{caption}</small>}
    </span>
  );
}

// Mini bars: items [{ v: 0..1, c: colour }] oldest -> newest.
export function MiniBars({ items, w = 110, h = 26, caption }) {
  if (!items || !items.length) return null;
  const n = items.length, gap = 3, bw = (w - (n - 1) * gap) / n;
  return (
    <span className="spark">
      <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} aria-hidden="true">
        {items.map((it, i) => {
          const bh = Math.max(3, it.v * (h - 2));
          return <rect key={i} x={i * (bw + gap)} y={h - bh} width={bw} height={bh} rx="1.5" fill={it.c} />;
        })}
      </svg>
      {caption && <small>{caption}</small>}
    </span>
  );
}

// Four-step dots (C4 while it has no real decisions).
export function StepDots({ w = 150, caption }) {
  const xs = [8, w / 3, (2 * w) / 3, w - 8];
  return (
    <span className="spark">
      <svg width={w} height="20" viewBox={`0 0 ${w} 20`} aria-hidden="true">
        <path d={`M8 10H${w - 8}`} stroke="#CFC5F7" strokeWidth="2" />
        {xs.map((x, i) => <circle key={i} cx={x} cy="10" r={i ? 5 : 6} fill="#FFFFFF" stroke={i ? "#B9A8F2" : "#6D4FD8"} strokeWidth={i ? 2 : 2.5} />)}
      </svg>
      {caption && <small>{caption}</small>}
    </span>
  );
}

export function IconChip({ name, tone = "teal", size = "", iconSize = 20 }) {
  return <span className={`ic ${size} tone-${tone}`}><Icon name={name} size={iconSize} className="" /></span>;
}

// ── real series from the saved runs (newest first from the API) ──────────────────────
const C3_BAR = { APPROVED_FOR_REVIEW: [1, "#16A34A"], REJECTED: [0.7, "#98A2B3"], FAILED: [0.7, "#F04438"],
  BLOCKED: [0.35, "#CBD3DC"], ABSTAINED_NO_REPAIR: [0.35, "#CBD3DC"] };

export function seriesFrom(runs, n = 12) {
  const all = (runs || []).slice().reverse();                       // oldest -> newest
  const withTruth = all.filter((r) => r.truth_pp != null && r.c1_pp != null);
  const last = (a, k) => a.slice(-k);
  const pipe = all.filter((r) => r.mode === "pipeline" && r.pipeline_runtime != null);
  return {
    truth: last(withTruth, n).map((r) => r.truth_pp),
    c1: last(all.filter((r) => r.c1_pp != null), n).map((r) => r.c1_pp),
    inside: last(withTruth.filter((r) => r.inside != null), 20).map((r) => ({ v: r.inside ? 1 : 0.25, c: r.inside ? "#14B8A6" : "#F04438" })),
    c2: last(all.filter((r) => r.c2_ok === true || r.c2_ok === false), n).map((r) => ({ v: r.c2_ok ? 1 : 0.3, c: r.c2_ok ? "#22A05B" : "#B7DFC6" })),
    c3: last(all.filter((r) => r.c3_status), 8).map((r) => { const [v, c] = C3_BAR[r.c3_status] || [0.5, "#CBD3DC"]; return { v, c }; }),
    runtimes: last(pipe, 9).map((r) => r.pipeline_runtime),
    outcomes: all.reduce((m, r) => { m[r.outcome] = (m[r.outcome] || 0) + 1; return m; }, {}),
    total: all.length,
    insideN: withTruth.filter((r) => r.inside === true).length,
    insideOf: withTruth.filter((r) => r.inside != null).length,
  };
}

// recent saved runs; `dep` re-fetches (e.g. the number of Drift Lab runs this session)
export function useRecent(dep) {
  const runs = useApi(api.runs, [dep]);
  return { runs: runs.data || [], s: seriesFrom(runs.data || []), loading: runs.loading };
}
