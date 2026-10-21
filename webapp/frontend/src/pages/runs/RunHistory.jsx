import { useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../../api.js";
import { useApi } from "../../hooks.js";
import { OUTCOME_CLASS, OUTCOME_HELP, sg } from "../../ui/c1format.js";
import { Empty, ErrorBox, PageHeader } from "../../ui/parts.jsx";
import "./Runs.css";

export const SOURCE_LABEL = { drift_lab: "Drift Lab", c1_upload: "C1 upload", pipeline_upload: "Pipeline upload" };
export const TYPE_LABEL = {
  none: "clean", marginal_single: "one feature", marginal_multi: "several features",
  interaction: "hidden interaction", combined: "shift + interaction",
};
export const when = (iso) => new Date(iso).toLocaleString(undefined, { day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit" });

const FILTERS = {
  source: [["all", "All"], ["drift_lab", "Drift Lab"], ["c1_upload", "C1 upload"], ["pipeline_upload", "Pipeline upload"]],
  dataset: [["all", "All"], ["uci", "UCI Adult"], ["ieee", "IEEE-CIS"]],
  outcome: [["all", "All"], ["correct", "Correct"], ["false alert", "False alert"], ["missed harm", "Missed harm"], ["no truth", "No truth"]],
  inside: [["all", "All"], ["yes", "Inside"], ["no", "Outside"]],
};

function keep(r, f) {
  if (f.source !== "all" && r.source !== f.source) return false;
  if (f.dataset !== "all" && r.dataset !== f.dataset) return false;
  if (f.outcome === "correct" && !r.outcome.startsWith("correct")) return false;
  if (f.outcome !== "all" && f.outcome !== "correct" && r.outcome !== f.outcome) return false;
  if (f.inside === "yes" && r.inside !== true) return false;
  if (f.inside === "no" && r.inside !== false) return false;
  return true;
}

export default function RunHistory() {
  const list = useApi(api.runs, []);
  const [f, setF] = useState({ source: "all", dataset: "all", outcome: "all", inside: "all" });
  const all = list.data || [];
  const rows = all.filter((r) => keep(r, f));
  const known = all.filter((r) => r.truth_pp != null);
  const n = known.length;
  const count = (fn) => known.filter(fn).length;
  const mae = n ? known.reduce((a, r) => a + Math.abs(r.truth_pp - r.c1_pp), 0) / n : 0;

  return (
    <>
      <PageHeader eyebrow="Platform · evidence" title="Run history"
        subtitle="Every Drift Lab run and every batch checked on the C1 page, saved as one file per run. Open any run to see the full analysis."
        chip={`${all.length} saved run${all.length === 1 ? "" : "s"}`} />
      <ErrorBox error={list.error} />

      <section className="card stack">
        <div className="head"><h3>All runs with a known truth (Drift Lab)</h3>
          <a className="pbtn runs-export" href={api.runsExportUrl()} download>Export CSV</a></div>
        <div className="runs-summary">
          <div className="s"><b>{n}</b><span>runs with truth</span></div>
          <div className="s"><b>{n ? `${count((r) => r.inside)}/${n}` : "-"}</b><span>truth inside interval</span></div>
          <div className="s"><b>{n ? `${mae.toFixed(2)} pp` : "-"}</b><span>mean error</span></div>
          <div className="s"><b>{n ? count((r) => r.classic_alarm) : "-"}</b><span>classic alarms</span></div>
          <div className="s"><b>{n ? count((r) => r.severity !== "none") : "-"}</b><span>C1 repair alerts</span></div>
          <div className="s"><b>{n ? count((r) => r.outcome === "false alert") : "-"}</b><span>C1 false alerts</span></div>
          <div className="s"><b>{n ? count((r) => r.outcome === "missed harm") : "-"}</b><span>C1 missed harm</span></div>
        </div>
      </section>

      <section className="card stack">
        <div className="runs-filters">
          {Object.entries(FILTERS).map(([k, opts]) => (
            <div className="field" key={k}>
              <label htmlFor={`f-${k}`}>{k === "inside" ? "Interval" : k[0].toUpperCase() + k.slice(1)}</label>
              <select id={`f-${k}`} value={f[k]} onChange={(e) => setF({ ...f, [k]: e.target.value })}>
                {opts.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
              </select>
            </div>
          ))}
          <span className="small muted">{rows.length} shown</span>
        </div>
        {list.loading ? <p className="muted">Loading…</p> : !all.length ? (
          <Empty>No saved runs yet. Run something in the <b>Drift Lab</b> or check a batch on the <b>C1 page</b>.</Empty>
        ) : (
          <div className="scroll-x">
            <table className="tbl">
              <thead><tr><th>When</th><th>Source</th><th>Data</th><th>Drift / file</th><th>Features</th><th className="mono">Strength</th>
                <th className="mono">Rows</th><th className="mono">Truth</th><th className="mono">C1</th><th>Inside?</th><th>Outcome</th><th /></tr></thead>
              <tbody>
                {rows.map((r) => (
                  <tr key={r.run_id}>
                    <td className="mono">{when(r.saved_at)}</td>
                    <td><span className="src">{SOURCE_LABEL[r.source] || r.source}{r.mode === "pipeline" ? " · pipeline" : ""}</span></td>
                    <td>{r.dataset.toUpperCase()}</td>
                    <td>{r.drift_type ? TYPE_LABEL[r.drift_type] : r.file}</td>
                    <td className="mono">{(r.features || []).join(", ") || "-"}</td>
                    <td className="mono">{r.drift_type && r.drift_type !== "none" ? r.magnitude.toFixed(2) : "-"}</td>
                    <td className="mono">{r.batch_size?.toLocaleString() ?? "-"}</td>
                    <td className="mono">{sg(r.truth_pp)}</td>
                    <td className="mono">{sg(r.c1_pp)}</td>
                    <td>{r.inside == null ? "-" : r.inside ? "✓" : "✗"}</td>
                    <td><span className={`outcome ${OUTCOME_CLASS[r.outcome]}`} title={OUTCOME_HELP[r.outcome]}>{r.outcome}</span></td>
                    <td><Link className="linkbtn" to={`/runs/${r.run_id}`}>View</Link></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        <p className="small muted">Outcome compares C1's repair decision with the truth at the 1 pp severity threshold. Hover a tag for its meaning.</p>
      </section>
    </>
  );
}
