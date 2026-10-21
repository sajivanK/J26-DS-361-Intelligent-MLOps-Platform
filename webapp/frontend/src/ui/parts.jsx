// Small shared building blocks. Use these on your page so every page looks the same.

export function PageHeader({ eyebrow, title, subtitle, chip }) {
  return (
    <div className="head">
      <div className="titles">
        {eyebrow && <span className="eyebrow">{eyebrow}</span>}
        <h1>{title}</h1>
        {subtitle && <p className="muted" style={{ maxWidth: "72ch" }}>{subtitle}</p>}
      </div>
      {chip && <span className="chip">{chip}</span>}
    </div>
  );
}

export function Verdict({ title, text, tone = "ok", aside }) {
  return (
    <div className={`verdict ${tone === "warn" ? "warn" : ""}`} role="status">
      <b>{title}</b><span>{text}</span>
      {aside && <span className="small mono aside">{aside}</span>}
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
