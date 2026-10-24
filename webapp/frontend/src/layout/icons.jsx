// Sidebar and page icons (24x24, stroke). Add a new one here if your page needs it.
const P = { fill: "none", stroke: "currentColor", strokeWidth: 2, strokeLinecap: "round", strokeLinejoin: "round" };

const paths = {
  overview: (<><path d="M3 11 12 4l9 7" /><path d="M5 10v10h14V10" /></>),
  lab: (<path d="M9 3h6M10 3v6L4 20a1 1 0 0 0 .9 1.5h14.2A1 1 0 0 0 20 20l-6-11V3" />),
  runs: (<><circle cx="12" cy="12" r="9" /><path d="M12 7v5l3 3" /></>),
  detect: (<><circle cx="11" cy="11" r="7" /><path d="m20 20-3.5-3.5" /></>),
  diagnose: (<><path d="M6 3h9l4 4v14H6z" /><path d="M9 12h7M9 16h7" /></>),
  repair: (<path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z" />),
  release: (<><path d="M4.5 16.5c-1.5 1.26-2 5-2 5s3.74-.5 5-2c.71-.84.7-2.13-.09-2.91a2.18 2.18 0 0 0-2.91-.09z" /><path d="m12 15-3-3a22 22 0 0 1 2-3.95A12.88 12.88 0 0 1 22 2c0 2.72-.78 7.5-6 11a22.35 22.35 0 0 1-4 2z" /><path d="M9 12H4s.55-3.03 2-4c1.62-1.08 5 0 5 0" /><path d="M12 15v5s3.03-.55 4-2c1.08-1.62 0-5 0-5" /></>),
  pulse: (<path d="M2 12h4l3-8 4 16 3-8h6" />),
  warn: (<><path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z" /><path d="M12 9v4M12 17h.01" /></>),
  check: (<path d="m5 12 5 5 9-10" />),
  checkCircle: (<><circle cx="12" cy="12" r="9" /><path d="m8 12 3 3 5-6" /></>),
  minus: (<><circle cx="12" cy="12" r="9" /><path d="M8 12h8" /></>),
  clock: (<><circle cx="12" cy="12" r="9" /><path d="M12 7v5l3 3" /></>),
  bars: (<path d="M6 20V12M12 20V5M18 20v-9" />),
  db: (<><ellipse cx="12" cy="5" rx="8" ry="3" /><path d="M4 5v14c0 1.7 3.6 3 8 3s8-1.3 8-3V5M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3" /></>),
  layers: (<><path d="m12 3 9 5-9 5-9-5z" /><path d="m3 13 9 5 9-5" /></>),
  bolt: (<path d="M13 2 4 14h7l-1 8 9-12h-7z" />),
  shield: (<><path d="M12 3 4 6v6c0 5 3.5 8 8 9 4.5-1 8-4 8-9V6z" /><path d="m9 12 2 2 4-4" /></>),
  trend: (<><path d="m3 17 6-6 4 4 8-8" /><path d="M15 7h6v6" /></>),
  play: (<path d="M7 5v14l11-7z" fill="currentColor" />),
  upload: (<path d="M12 16V4M6 10l6-6 6 6M4 20h16" />),
  chevron: (<path d="m9 6 6 6-6 6" />),
  arrow: (<path d="M5 12h14M13 6l6 6-6 6" />),
  nodes: (<><circle cx="12" cy="5" r="2.5" /><circle cx="5" cy="19" r="2.5" /><circle cx="19" cy="19" r="2.5" /><path d="M11 7 6 17M13 7l5 10M7.5 19h9" /></>),
  bell: (<><path d="M6 9a6 6 0 0 1 12 0c0 6 2.5 7.5 2.5 7.5h-17S6 15 6 9z" /><path d="M10 20a2 2 0 0 0 4 0" /></>),
  target: (<><circle cx="12" cy="12" r="9" /><circle cx="12" cy="12" r="5" /><circle cx="12" cy="12" r="1" /></>),
  gear: (<><circle cx="12" cy="12" r="3" /><path d="M12 2v3M12 19v3M2 12h3M19 12h3M5 5l2 2M17 17l2 2M5 19l2-2M17 7l2-2" /></>),
  doc: (<><rect x="4" y="4" width="16" height="16" rx="2" /><path d="M8 9h8M8 13h8M8 17h5" /></>),
  code: (<><path d="M6 3h9l4 4v14H6z" /><path d="M9 13 7.5 14.5 9 16M13 13l1.5 1.5L13 16" /></>),
  info: (<><circle cx="12" cy="12" r="9" /><path d="M12 11v5M12 8h.01" /></>),
};

export function Icon({ name, size = 18, className = "ico", ...rest }) {
  return (
    <svg className={className} width={size} height={size} viewBox="0 0 24 24" aria-hidden="true" {...P} {...rest}>
      {paths[name]}
    </svg>
  );
}
