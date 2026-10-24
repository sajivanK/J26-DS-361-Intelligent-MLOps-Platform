// Small shared building blocks. Use these on your page so every page looks the same.
import { Icon } from "../layout/icons.jsx";

export function PageHeader({ eyebrow, title, subtitle, chip, icon, right }) {
  return (
    <div className="head">
      <div className="titles" style={icon ? { flexDirection: "row", gap: 16, alignItems: "center" } : undefined}>
        {icon && <span className="ic tone-teal" style={{ width: 54, height: 54, borderRadius: 16 }}><Icon name={icon} size={28} className="" /></span>}
        <div className="titles">
        {eyebrow && <span className="eyebrow">{eyebrow}</span>}
        <h1>{title}</h1>
        {subtitle && <p className="muted" style={{ maxWidth: "72ch", fontSize: ".95rem" }}>{subtitle}</p>}
        </div>
      </div>
      {(chip || right) && (
        <div className="stack" style={{ alignItems: "flex-end", gap: 10 }}>
          {chip && <span className="chip mono">{chip}</span>}
          {right}
        </div>)}
    </div>
  );
}

export function Verdict({ title, text, tone = "ok", aside, extra }) {
  const warn = tone === "warn";
  return (
    <div className={`verdict ${warn ? "warn" : ""}`} role="status">
      <span className="vic"><Icon name={warn ? "warn" : "checkCircle"} size={22} className=""
        style={{ color: warn ? "var(--t-amber-ic)" : "var(--t-green-ic)" }} /></span>
      <span className="vtx"><b>{title}</b><span>{text}</span></span>
      {aside && <span className="small mono aside">{aside}</span>}
      {extra}
    </div>
  );
}

export function SourceBadge({ source }) {
  return <span className={`badge ${source}`}>{source}</span>;
}

export function ErrorBox({ error }) {
  return error ? <div className="error" role="alert">{error}</div> : null;
}

export function Empty({ children }) {
  return <div className="empty">{children}</div>;
}

export function FeatureTags({ features, emptyText = "none listed", mark }) {
  if (!features || !features.length) return <span className="tagf none">{emptyText}</span>;
  return features.map((f) => <span key={f} className="tagf">{f}{mark && mark(f) ? " ✓" : ""}</span>);
}

export function KV({ rows }) {
  return (
    <div className="kv">
      {rows.map(([k, v]) => [<span key={k + "k"}>{k}</span>, <span key={k + "v"}>{v}</span>])}
    </div>
  );
}

export function JsonBox({ title = "Raw output (JSON)", note, data }) {
  return (
    <details className="more">
      <summary>{title}</summary>
      <div>
        {note && <p className="small muted">{note}</p>}
        <pre>{data ? JSON.stringify(data, null, 2) : "Nothing yet."}</pre>
      </div>
    </details>
  );
}
