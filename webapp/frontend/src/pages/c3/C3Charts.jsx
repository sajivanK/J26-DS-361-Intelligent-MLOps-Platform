// Plain SVG charts for Component 3 Synthetic Data Repair.
// Strict design system compliance: only CSS variables (var(--...)), no hardcoded colors.

export function GeneratorComparisonChart({ generators = [] }) {
  if (!generators.length) return null;
  const W = 580, rowH = 34, pad = 85, H = generators.length * rowH + 24;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ maxWidth: 640 }} role="img" aria-label="Generator Fidelity vs KS Distance">
      <text x={pad} y={12} fill="var(--muted)" fontSize="11" fontFamily="var(--sans)">Generator & Slice Fidelity (% higher is better)</text>
      <text x={W - 8} y={12} textAnchor="end" fill="var(--muted)" fontSize="11" fontFamily="var(--sans)">{"Mean KS (\u2264 0.35 pass)"}</text>
      {generators.map((g, i) => {
        const yy = 22 + i * rowH;
        const barW = Math.max(10, ((W - pad - 110) * g.fidelity));
        const isPrescribed = g.name === "TVAE";
        return (
          <g key={g.name}>
            <text x={pad - 10} y={yy + 18} textAnchor="end" fill="var(--fg)" fontWeight={isPrescribed ? "600" : "400"} fontSize="13" fontFamily="var(--sans)">
              {g.name}
            </text>
            <rect x={pad} y={yy + 4} width={W - pad - 110} height="20" rx="4" fill="var(--track)" />
            <rect
              x={pad}
              y={yy + 4}
              width={barW}
              height="20"
              rx="4"
              fill={isPrescribed ? "var(--accent)" : "var(--accent-2)"}
              opacity={isPrescribed ? "1" : "0.75"}
            />
            <text x={pad + 8} y={yy + 18} fill="var(--surface)" fontSize="11" fontWeight="600" fontFamily="var(--mono)">
              {(g.fidelity * 100).toFixed(1)}%
            </text>
            <text x={W - 8} y={yy + 18} textAnchor="end" fill={g.mean_ks <= 0.35 ? "var(--ok-fg)" : "var(--warn-fg)"} fontSize="12" fontFamily="var(--mono)">
              KS {g.mean_ks.toFixed(3)}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

export function MinimumEffectiveSearchChart({ items = [] }) {
  if (!items.length) return null;
  const W = 580, H = 160, pl = 65, pr = 20, pt = 20, pb = 35;
  const pw = W - pl - pr;
  const ph = H - pt - pb;

  const f1Vals = items.map((d) => parseFloat(d.f1_gain.replace("+", "")));
  const maxF1 = Math.max(0.045, ...f1Vals) * 1.15;

  const x = (i) => pl + (i / (items.length - 1)) * pw;
  const y = (val) => pt + ph * (1 - Math.max(0, val) / maxF1);

  // Line path
  const points = items.map((d, i) => `${x(i)},${y(parseFloat(d.f1_gain.replace("+", "")))}`).join(" ");

  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ maxWidth: 640 }} role="img" aria-label="Minimum Effective Repair Quantity Search">
      {/* Target threshold +0.0100 F1 */}
      <line x1={pl} x2={W - pr} y1={y(0.01)} y2={y(0.01)} stroke="var(--truth)" strokeDasharray="4 3" strokeWidth="1.5" />
      <text x={W - pr} y={y(0.01) - 5} textAnchor="end" fill="var(--truth)" fontSize="10" fontFamily="var(--mono)">
        +0.0100 Min Target Gain
      </text>

      {/* Grid line at 0 */}
      <line x1={pl} x2={W - pr} y1={y(0)} y2={y(0)} stroke="var(--line)" strokeWidth="1" />
      <text x={pl - 8} y={y(0) + 4} textAnchor="end" fill="var(--muted)" fontSize="10" fontFamily="var(--mono)">0.00</text>
      <text x={pl - 8} y={y(0.04) + 4} textAnchor="end" fill="var(--muted)" fontSize="10" fontFamily="var(--mono)">+0.04</text>

      {/* Gain curve */}
      <polyline fill="none" stroke="var(--accent)" strokeWidth="2.5" points={points} />

      {/* Data points */}
      {items.map((d, i) => {
        const val = parseFloat(d.f1_gain.replace("+", ""));
        const cx = x(i);
        const cy = y(val);
        const isSelected = d.status.includes("Minimum Effective");
        return (
          <g key={d.k}>
            <circle
              cx={cx}
              cy={cy}
              r={isSelected ? 6.5 : 4.5}
              fill={isSelected ? "var(--accent)" : "var(--surface)"}
              stroke="var(--accent)"
              strokeWidth={isSelected ? "3" : "2"}
            />
            <text x={cx} y={H - 12} textAnchor="middle" fill="var(--fg)" fontSize="11" fontFamily="var(--mono)">
              K={d.k}
            </text>
            <text x={cx} y={cy - 10} textAnchor="middle" fill={isSelected ? "var(--accent)" : "var(--muted)"} fontSize="10" fontWeight={isSelected ? "700" : "500"} fontFamily="var(--mono)">
              {d.f1_gain}
            </text>
            {isSelected && (
              <text x={cx} y={cy + 18} textAnchor="middle" fill="var(--ok-fg)" fontSize="9" fontWeight="600" fontFamily="var(--sans)">
                OPTIMAL
              </text>
            )}
          </g>
        );
      })}
    </svg>
  );
}

export function SessionHistoryTimeline({ runs = [], selected = -1, onSelect }) {
  if (!runs.length) return null;
  const W = 620, H = 84, pad = 12;

  return (
    <div className="scroll-x">
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ minWidth: 500 }} role="img" aria-label="Session repair runs history">
        {runs.map((r, i) => {
          const colW = (W - pad * 2) / Math.max(runs.length, 4);
          const cx = pad + i * colW;
          const isSel = i === selected;
          const isApp = r.out?.repair_package_status === "APPROVED";
          return (
            <g
              key={r.id || i}
              style={{ cursor: "pointer" }}
              tabIndex="0"
              role="button"
              aria-label={`Run ${i + 1}: ${r.out?.repair_package_status}`}
              onClick={() => onSelect(i)}
              onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onSelect(i); } }}
            >
              <rect
                x={cx + 4}
                y={6}
                width={colW - 8}
                height={H - 12}
                rx="6"
                fill={isSel ? "var(--tint)" : "var(--surface)"}
                stroke={isSel ? "var(--accent)" : "var(--line)"}
                strokeWidth={isSel ? "2" : "1"}
              />
              <text x={cx + colW / 2} y={24} textAnchor="middle" fill="var(--muted)" fontSize="10" fontFamily="var(--mono)">
                #{i + 1} · {r.out?.generator || "C3"}
              </text>
              <rect
                x={cx + colW / 2 - 32}
                y={32}
                width="64"
                height="16"
                rx="8"
                fill={isApp ? "var(--ok-bg)" : "var(--warn-bg)"}
                stroke={isApp ? "var(--ok-line)" : "var(--warn-line)"}
              />
              <text
                x={cx + colW / 2}
                y={43}
                textAnchor="middle"
                fill={isApp ? "var(--ok-fg)" : "var(--warn-fg)"}
                fontSize="9"
                fontWeight="700"
                fontFamily="var(--mono)"
              >
                {r.out?.repair_package_status || "UNKNOWN"}
              </text>
              <text x={cx + colW / 2} y={64} textAnchor="middle" fill="var(--fg)" fontSize="10" fontFamily="var(--mono)">
                {r.out?.n_synthetic_rows || 0} rows
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}
