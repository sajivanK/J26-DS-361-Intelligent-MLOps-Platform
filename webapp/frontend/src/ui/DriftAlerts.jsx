// Drift alerts waiting for C2 (requested by Tharmika for the C2 page; platform code, owner Sajivan).
// A drift alert is a saved run where C1 found harmful drift and C2 has not diagnosed that batch yet.
// Diagnose runs C2 on the same batch (without labels) and saves the result with the run; Dismiss stops
// counting it. The sidebar badge on the C2 tab shows the same number.
import { Link } from "react-router-dom";
import { api } from "../api.js";
import { useAction, useApi } from "../hooks.js";
import { refreshAlerts } from "../platform.jsx";
import { sg } from "./c1format.js";
import { Icon } from "../layout/icons.jsx";
import "./DriftAlerts.css";

const SOURCE = { drift_lab: "Drift Lab", c1_upload: "C1 page upload", pipeline_upload: "Pipeline upload" };
const SHOW_DONE = 5;

export default function DriftAlerts({ onDiagnosed }) {
  const list = useApi(api.alerts, []);
  const [doDiag, diag] = useAction(api.alertDiagnose);
  const [doDismiss, dis] = useAction(api.alertDismiss);
  const items = list.data?.items || [];
  const waiting = items.filter((i) => i.alert === "waiting");
  const other = items.filter((i) => i.alert !== "waiting").slice(0, SHOW_DONE);
  const busy = diag.busy || dis.busy;

  async function act(fn, id, diagnosed) {
    const out = await fn(id);
    if (!out) return;
    list.reload();
    refreshAlerts();
    if (diagnosed && onDiagnosed) onDiagnosed(out);
  }
  if (list.loading && !list.data) return null;
  if (!items.length) return null;

  const when = (s) => new Date(s.saved_at).toLocaleString([], { day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit" });
  const what = (s) => s.source === "drift_lab"
    ? `${(s.drift_type || "").replace("marginal_", "")} · ${(s.features || []).join(", ") || "clean"}${s.magnitude ? ` · ${s.magnitude}` : ""}`
    : s.file ? String(s.file).split(/[\\/]/).pop() : "uploaded batch";
  const status = (s) => {
    if (s.alert === "waiting") return <span className="da-tag">waiting</span>;
    if (s.alert === "unsupported") return <span className="small muted">C2 supports {(list.data.c2_datasets || ["uci"]).join(", ").toUpperCase()} only</span>;
    if (s.alert === "dismissed") return <span className="small muted">dismissed</span>;
    const grade = s.c2_ok === true ? " ✓" : s.c2_ok === false ? ` ✗ (truth: ${s.true_cause || "-"})` : "";
    return <span className="da-ok">diagnosed · {s.c2_category}: {s.c2_root}{grade}</span>;
  };
  const row = (s) => (
    <tr key={s.run_id} className={s.alert === "waiting" ? "" : "da-done"}>
      <td className="mono"><Link to={`/runs/${s.run_id}`}>{s.run_id}</Link></td>
      <td className="mono">{when(s)}</td>
      <td>{SOURCE[s.source] || s.source}<span className="small muted"> · {s.dataset.toUpperCase()}</span></td>
      <td className="mono">{what(s)}</td>
      <td>{s.severity}</td>
      <td className="mono">{sg(s.c1_pp)} pp</td>
      <td>{status(s)}</td>
      <td className="da-act">{s.alert === "waiting" && (<>
        <button className="btn sm primary" type="button" disabled={busy} onClick={() => act(doDiag, s.run_id, true)}>
          {diag.busy ? "Diagnosing…" : "Diagnose"}</button>
        <button className="btn sm" type="button" disabled={busy} onClick={() => act(doDismiss, s.run_id, false)}>Dismiss</button>
      </>)}</td>
    </tr>
  );

  return (
    <section className="stack" aria-label="Drift alerts">
      {waiting.length > 0 && (
        <div className="da-banner" role="status">
          <Icon name="warn" size={20} className="" />
          <span><b>{waiting.length} drift alert{waiting.length === 1 ? " is" : "s are"} waiting for diagnosis.</b> C1 found harmful drift in these runs; C2 has not diagnosed them yet.</span>
        </div>)}
      <div className="card stack tight">
        <div className="section-title"><h3>Drift alerts</h3>
          <span className="small muted">from Drift Lab "Run C1 only", the C1 page and uploads · click a run to open it</span></div>
        <div className="scroll-x">
          <table className="tbl">
            <thead><tr><th>Run</th><th>When</th><th>Source</th><th>Drift</th><th>C1 severity</th><th className="mono">C1 harm</th><th>Status</th><th /></tr></thead>
            <tbody>{[...waiting, ...other].map(row)}</tbody>
          </table>
        </div>
        {(diag.error || dis.error) && <div className="error">{String(diag.error || dis.error)}</div>}
        <p className="small muted">Diagnose runs C2 on the same batch, without its labels, and saves the result with that run. Dismiss stops counting it without a diagnosis.</p>
      </div>
    </section>
  );
}
