// C1 page, "Research evidence" tab (owner: Sajivan K).
// Every number comes from GET /api/c1/research, which reads the result JSONs committed in c1/results/.
import { useState } from "react";
import { api } from "../../api.js";
import { useApi } from "../../hooks.js";
import { ErrorBox } from "../../ui/parts.jsx";
import "./C1Research.css";

const pct0 = (v) => (v == null ? "–" : `${Math.round(v * 100)}%`);
const pct1 = (v) => (v == null ? "–" : `${(v * 100).toFixed(1)}%`);
const f2 = (v) => (v == null ? "–" : v.toFixed(2));
const COL = { baseline: "#C7CDD6", reference: "#E2E6EB", ablation: "#9ED9D1", n1: "#0F766E" };

export default function C1Research({ ds, setDs }) {
  const res = useApi(() => api.c1Research(ds), [ds]);
  const [target, setTarget] = useState("fixed");
  if (res.loading) return <div className="empty">Loading research evidence…</div>;
  if (res.error) return <ErrorBox error={res.error} />;
  const r = res.data;
  const n1 = r.n1[target], n2 = r.n2[target], n3 = r.n3, fr = r.fresh;
  const pp = `pp of ${r.metric}`;

  return (
    <div className="c1r stack" style={{ gap: 18 }}>
      <div className="c1r-top">
        <p className="muted">How we know it works: {r.protocol.conditions.toLocaleString()} injected drift conditions,
          group-aware cross-validation ({r.protocol.repeats} repeats × {r.protocol.folds} folds), and {fr.n_batches} fresh unseen batches.</p>
        <div className="c1r-switches">
        <div className="c1r-seg" role="group" aria-label="Dataset">
          <button type="button" aria-pressed={ds === "uci"} onClick={() => setDs("uci")}>UCI Adult</button>
          <button type="button" aria-pressed={ds === "ieee"} onClick={() => setDs("ieee")}>IEEE-CIS Fraud</button>
        </div>
        <div className="c1r-seg" role="group" aria-label="Harm target">
          <button type="button" aria-pressed={target === "fixed"} onClick={() => setTarget("fixed")}
            title="Batch result against the overall test result: includes natural sampling noise, as in production">Fixed target</button>
          <button type="button" aria-pressed={target === "paired"} onClick={() => setTarget("paired")}
            title="Same rows before and after the drift: noise cancels, only drift harm remains">Paired target</button>
        </div>
        </div>
      </div>

      <section className="c1r-kpis">
        <Kpi tag="N1 · harm" value={`${f2(n1.n1_mae)} pp`} text={`mean error of the predicted ${r.metric} drop (${n1.atc_name}: ${f2(n1.atc_mae)} pp)`} />
        <Kpi tag="N1 · vs baseline" value={pct0(n1.atc_improvement_pct / 100)} text={`lower error than ${n1.atc_name}, the standard label-free estimator`} />
        <Kpi tag="N2 · interval" value={pct1(fr.overall.coverage)} text={`of unseen batches had the true harm inside the 95% interval (${pct1(n2.coverage_mondrian)} in CV)`} />
        <Kpi tag="N3 · extent" value={pct1(n3.accuracy)} text="drift extent (none / single / multi) predicted correctly" />
      </section>

      <section className="card">
        <Head badge="N1" title="Harm prediction without labels" />
        <p className="c1r-q">How far off is each method when predicting the {r.metric} drop? Lower is better. {target === "fixed" ? "Fixed" : "Paired"} target, all {r.protocol.conditions.toLocaleString()} conditions.</p>
        <div className="c1r-split">
          <div>
            <HBars bars={n1.bars} unit={`MAE, ${pp}`} />
            <div className="c1r-legend"><Lg c={COL.baseline} t="published baselines" /><Lg c={COL.ablation} t="one signal family only" /><Lg c={COL.n1} t="N1" /></div>
          </div>
          <div className="c1r-facts">
            <Fact b={pct0(n1.atc_improvement_pct / 100)} t={<>lower error than {n1.atc_name}. {sigText(n1.tests)}</>} />
            {n1.detector_gain.mae_diff != null && (
              <Fact b={`${n1.detector_gain.mae_diff < 0 ? "−" : "+"}${Math.abs(n1.detector_gain.mae_diff).toFixed(2)} pp`}
                t={`change in error when the 6 detectors are added on top of all prediction statistics (better in ${n1.detector_gain.repeats} of 10 repeats).`} />)}
            {n1.model_comparison && (
              <Fact b="Model" t={<>meta-learner comparison (same folds, untuned): {n1.model_comparison.map((m) => `${m.model} ${f2(m.mae)}`).join(" · ")}</>} />)}
            <Fact warn b="Limit" t="drift on a high-harm feature never seen in calibration is under-predicted." />
          </div>
        </div>
        <Src files={r.sources.filter((s) => s.startsWith("novelty1") || s.startsWith("baselines"))} />
      </section>

      <section className="card">
        <Head badge="N2" title="95% interval: each fix answers a measured failure" />
        <p className="c1r-q">Share of batches whose true harm fell inside the interval (mean over {n2.splits} group-aware splits).</p>
        <div className="c1r-two">
          <div>
            <h3>1 · Batch size: plain conformal under-covers small batches</h3>
            <CovBars groups={n2.by_batch.map((b) => `${b.batch_size.toLocaleString()} rows`)}
              series={[n2.by_batch.map((b) => b.plain), n2.by_batch.map((b) => b.normalised)]} colors={["#C7CDD6", "#0D9488"]} />
            <div className="c1r-legend"><Lg c="#C7CDD6" t="Plain conformal" /><Lg c="#0D9488" t="Batch-size normalised" /><Lg line t="95% target" /></div>
          </div>
          <div>
            <h3>2 · Drift extent: one shared q under-covers multi-feature drift</h3>
            <CovBars groups={n2.by_type.map((t) => t.type)}
              series={[n2.by_type.map((t) => t.normalised), n2.by_type.map((t) => t.mondrian)]} colors={["#9ED9D1", "#0F766E"]} />
            <div className="c1r-legend"><Lg c="#9ED9D1" t="Normalised (one q)" /><Lg c="#0F766E" t="Mondrian by N3 extent (engine)" /><Lg line t="95% target" /></div>
          </div>
        </div>
        <div className="c1r-facts row3">
          <Fact b={`${f2(n2.multi_before)} → ${f2(n2.multi_after)}`} t="multi-feature coverage after calibrating per N3-predicted extent." />
          <Fact b={`${n2.width_before.toFixed(1)} → ${n2.width_after.toFixed(1)}`} t="mean interval width in pp: the cost of honest intervals." />
          <Fact warn b="Limit" t="coverage drops for drift on features never seen in calibration." />
        </div>
        <Src files={r.sources.filter((s) => s.startsWith("novelty2") || s.startsWith("mondrian"))} />
      </section>

      <section className="c1r-two">
        <div className="card">
          <Head badge="N3" title="Drift extent and affected features" />
          <p className="c1r-q">Predicted extent (columns) for each true extent (rows), {r.protocol.repeats} repeats × {r.protocol.folds} folds.</p>
          <Confusion labels={n3.labels} m={n3.matrix} />
          <div className="c1r-facts">
            <Fact b={pct1(n3.accuracy)} t={`extent accuracy · F1 none ${f2(n3.f1.none)} · single ${f2(n3.f1.single)} · multi ${f2(n3.f1.multi)}`} />
            <Fact b={f2(fr.no_drift.mean_affected_listed)} t="features listed on clean batches (fresh test)" />
            <Fact warn b={`AUC ${f2(n3.interaction_auc)}`} t="interaction_suspected: experimental. Interaction drift keeps every feature's distribution, so detectors barely see it." />
          </div>
          <Src files={r.sources.filter((s) => s.startsWith("novelty3"))} />
        </div>
        <div className="card">
          <Head badge="✓" dark title={`Fresh test: ${fr.n_batches} unseen batches`} />
          <p className="c1r-q">New seeds, magnitudes and batch sizes never used in training. Engine {fr.engine_version}. Fixed target.</p>
          <div className="c1r-tablewrap">
            <table className="c1r-table">
              <thead><tr><th>Drift</th><th className="n">n</th><th className="n">true pp</th><th className="n">error</th><th className="n">in 95%</th><th className="n">width</th><th className="n">extent</th><th className="n">top-1</th></tr></thead>
              <tbody>{fr.rows.map((x) => (
                <tr key={x.type}><td>{x.type}</td><td className="n">{x.n}</td><td className="n">{f2(x.true_pp)}</td><td className="n">{f2(x.mae)}</td>
                  <td className="n">{pct0(x.coverage)}</td><td className="n">{x.width.toFixed(1)}</td><td className="n">{pct0(x.extent)}</td><td className="n">{x.top1 == null ? "–" : pct0(x.top1)}</td></tr>))}
              </tbody>
            </table>
          </div>
          <div className="c1r-facts">
            <Fact b={`${f2(fr.overall.mae_fixed)} pp`} t={<>mean error · <b>{pct1(fr.no_drift.false_alarm_rate)}</b> false alarms on clean batches · <b>{fr.overall.mean_runtime_s.toFixed(2)} s</b> per batch</>} />
          </div>
          <Src files={r.sources.filter((s) => s.startsWith("engine_test"))} />
        </div>
      </section>
    </div>
  );
}

function sigText(tests) {
  const yes = tests.filter((t) => t.n1_better).map((t) => t.name);
  const no = tests.filter((t) => !t.n1_better).map((t) => t.name);
  return [yes.length ? `Significantly better than ${yes.join(", ")} (Wilcoxon on condition groups, p < 0.05).` : "",
    no.length ? `Not significantly better than ${no.join(", ")} on this target.` : ""].filter(Boolean).join(" ");
}

function Kpi({ tag, value, text }) {
  return <div className="card c1r-kpi"><span className="c1r-tag">{tag}</span><span className="c1r-v">{value}</span><span className="c1r-k">{text}</span></div>;
}
function Head({ badge, title, dark }) {
  return <div className="c1r-hd"><span className={`c1r-badge ${dark ? "dark" : ""}`}>{badge}</span><h2>{title}</h2></div>;
}
function Fact({ b, t, warn }) {
  return <div className={`c1r-fact ${warn ? "warn" : ""}`}><b>{b}</b><span>{t}</span></div>;
}
function Lg({ c, t, line }) {
  return <span className="c1r-lg"><i className={line ? "line" : ""} style={line ? undefined : { background: c }} />{t}</span>;
}
function Src({ files }) {
  return <div className="c1r-src">source: c1/results/{files.join(" · ")}</div>;
}

function HBars({ bars, unit }) {
  const W = 560, rowh = 30, lab = 170, mx = Math.ceil(Math.max(...bars.map((b) => b.mae)) + 0.2);
  const x = (v) => ((W - lab - 60) * v) / mx;
  const ticks = Array.from({ length: mx + 1 }, (_, i) => i);
  return (
    <svg viewBox={`0 0 ${W} ${bars.length * rowh + 44}`} width="100%" role="img" aria-label={`Error by method, ${unit}`} className="c1r-svg">
      {bars.map((b, i) => {
        const y = i * rowh, n1 = b.kind === "n1";
        return (
          <g key={b.name}>
            <text x={lab - 10} y={y + 19} textAnchor="end" fontSize="12.5" fontWeight={n1 ? 700 : 500} fill="var(--fg)">{b.name}</text>
            <rect x={lab} y={y + 6} width={x(b.mae)} height="18" rx="5" fill={COL[b.kind]} />
            <text x={lab + x(b.mae) + 6} y={y + 19} fontSize="12.5" className="mono" fontWeight={n1 ? 700 : 500} fill={n1 ? "var(--accent)" : "#374151"}>{b.mae.toFixed(2)}</text>
          </g>);
      })}
      {ticks.map((t) => <text key={t} x={lab + x(t)} y={bars.length * rowh + 22} fontSize="11" fill="var(--muted)" textAnchor="middle">{t}</text>)}
      <text x={lab + x(mx) / 2} y={bars.length * rowh + 40} fontSize="11" fill="var(--muted)" textAnchor="middle">{unit}</text>
    </svg>
  );
}

function CovBars({ groups, series, colors, h = 190 }) {
  const W = 420, left = 34, bw = 22, ymin = Math.min(0.8, Math.floor(Math.min(...series.flat()) * 20) / 20);
  const gap = (W - left - 10) / groups.length;
  const Y = (v) => 10 + (1 - (v - ymin) / (1 - ymin)) * h;
  const ticks = []; for (let t = ymin; t <= 1.0001; t += 0.05) ticks.push(Math.round(t * 100) / 100);
  return (
    <svg viewBox={`0 0 ${W} ${h + 40}`} width="100%" role="img" aria-label="Coverage chart" className="c1r-svg">
      {ticks.map((t) => (
        <g key={t}><line x1={left} x2={W - 6} y1={Y(t)} y2={Y(t)} stroke="#EEF1F4" />
          <text x={left - 6} y={Y(t) + 4} fontSize="10.5" fill="var(--muted)" textAnchor="end">{Math.round(t * 100)}%</text></g>))}
      <line x1={left} x2={W - 6} y1={Y(0.95)} y2={Y(0.95)} stroke="var(--truth)" strokeDasharray="5 4" strokeWidth="1.6" />
      {groups.map((g, gi) => {
        const cx = left + gap * gi + gap / 2;
        return (
          <g key={g}>
            {series.map((ser, si) => {
              const v = ser[gi], xx = cx + (si - series.length / 2) * (bw + 3);
              return (<g key={si}><rect x={xx} y={Y(v)} width={bw} height={10 + h - Y(v)} rx="4" fill={colors[si]} />
                <text x={xx + bw / 2} y={Y(v) - 4} fontSize="9.5" textAnchor="middle" fill="#374151" className="mono">{Math.round(v * 100)}</text></g>);
            })}
            <text x={cx} y={h + 28} fontSize="11.5" textAnchor="middle" fill="var(--fg)">{g}</text>
          </g>);
      })}
    </svg>
  );
}

function Confusion({ labels, m }) {
  return (
    <div className="c1r-cm" role="table" aria-label="Drift extent confusion matrix">
      <div />
      {labels.map((l) => <div key={l} className="h top">{l}</div>)}
      {labels.map((row, i) => [
        <div key={`h${row}`} className="h">{row}</div>,
        ...labels.map((col, j) => {
          const v = m[i][j], diag = i === j;
          return <div key={`${row}${col}`} className={`c ${diag ? "diag" : v > 0.03 ? "warm" : ""}`}>{(v * 100).toFixed(1)}%</div>;
        }),
      ])}
    </div>
  );
}
