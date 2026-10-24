// Line icons for the C2 page and report (Tharmika N). Stroke icons in currentColor, decorative
// (aria-hidden): every icon sits next to a text label.
const P = {
  bars: <><rect x="4" y="11" width="3.5" height="9" rx="1" /><rect x="10.25" y="5" width="3.5" height="15" rx="1" /><rect x="16.5" y="9" width="3.5" height="11" rx="1" /></>,
  bell: <><path d="M6 8a6 6 0 0 1 12 0c0 7 3 8 3 8H3s3-1 3-8" /><path d="M10.3 21a1.9 1.9 0 0 0 3.4 0" /></>,
  clipboard: <><rect x="5" y="4" width="14" height="17" rx="2" /><path d="M9 4h6v3H9z" /><line x1="9" y1="11" x2="15" y2="11" /><line x1="9" y1="15" x2="15" y2="15" /></>,
  check: <polyline points="4,12 10,18 20,6" />,
  checkCircle: <><circle cx="12" cy="12" r="9" /><polyline points="7.5,12.5 10.5,15.5 16.5,9" /></>,
  clock: <><circle cx="12" cy="12" r="8.5" /><polyline points="12,7.5 12,12 15,14" /></>,
  warn: <><path d="M12 3.5 2.5 20h19L12 3.5z" /><line x1="12" y1="10" x2="12" y2="14" /><line x1="12" y1="17" x2="12" y2="17.2" /></>,
  search: <><circle cx="11" cy="11" r="6.5" /><line x1="16" y1="16" x2="20.5" y2="20.5" /></>,
  x: <><line x1="6" y1="6" x2="18" y2="18" /><line x1="18" y1="6" x2="6" y2="18" /></>,
  info: <><circle cx="12" cy="12" r="9.5" /><line x1="12" y1="11" x2="12" y2="16.5" /><line x1="12" y1="7.5" x2="12" y2="7.6" /></>,
  copy: <><rect x="9" y="9" width="11" height="11" rx="2" /><path d="M5 15V5a1 1 0 0 1 1-1h10" /></>,
  db: <><ellipse cx="12" cy="5.5" rx="7" ry="2.5" /><path d="M5 5.5v13c0 1.4 3.1 2.5 7 2.5s7-1.1 7-2.5v-13" /><path d="M5 12c0 1.4 3.1 2.5 7 2.5s7-1.1 7-2.5" /></>,
  sort: <><polyline points="8,9 12,5 16,9" /><polyline points="8,15 12,19 16,15" /></>,
  calendar: <><rect x="3.5" y="5" width="17" height="15.5" rx="2" /><line x1="3.5" y1="10" x2="20.5" y2="10" /><line x1="8" y1="3" x2="8" y2="7" /><line x1="16" y1="3" x2="16" y2="7" /></>,
  sparkle: <><path d="M10 2.5l1.8 5.2 5.2 1.8-5.2 1.8L10 16.5l-1.8-5.2L3 9.5l5.2-1.8z" /><path d="M18 13l.9 2.6 2.6.9-2.6.9L18 20l-.9-2.6-2.6-.9 2.6-.9z" /></>,
  trend: <><polyline points="3,17 9,11 13,15 21,7" /><polyline points="15,7 21,7 21,13" /></>,
  people: <><circle cx="9" cy="8" r="3.5" /><path d="M2.5 20c0-3.6 2.9-6 6.5-6s6.5 2.4 6.5 6" /><circle cx="17" cy="9" r="2.6" /><path d="M16 14.2c3 .2 5.5 2.3 5.5 5.3" /></>,
  person: <><circle cx="12" cy="8" r="3.5" /><path d="M5.5 20c0-3.6 2.9-6 6.5-6s6.5 2.4 6.5 6" /></>,
  target: <><circle cx="12" cy="12" r="9" /><circle cx="12" cy="12" r="5" /><circle cx="12" cy="12" r="1.5" /></>,
  chartUp: <><polyline points="3,4 3,20 21,20" /><polyline points="6,15 10,10 13,13 19,6" /></>,
  link: <><path d="M10 14a4 4 0 0 0 5.7 0l3-3a4 4 0 0 0-5.7-5.7l-1 1" /><path d="M14 10a4 4 0 0 0-5.7 0l-3 3a4 4 0 0 0 5.7 5.7l1-1" /></>,
  code: <><polyline points="8,7 3,12 8,17" /><polyline points="16,7 21,12 16,17" /><line x1="14" y1="4" x2="10" y2="20" /></>,
  doc: <><path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z" /><polyline points="14,3 14,8 19,8" /><line x1="9" y1="13" x2="15" y2="13" /><line x1="9" y1="17" x2="13" y2="17" /></>,
  pie: <><circle cx="12" cy="12" r="8.5" /><path d="M12 3.5V12h8.5" /></>,
  shield: <><path d="M12 3 4.5 6v6c0 4.5 3.2 7.8 7.5 9 4.3-1.2 7.5-4.5 7.5-9V6z" /><polyline points="9,12 11,14 15,10" /></>,
  bulb: <><path d="M9 18h6" /><path d="M10 21h4" /><path d="M12 3a6 6 0 0 0-3.5 10.9c.6.5 1 1.2 1 2.1h5c0-.9.4-1.6 1-2.1A6 6 0 0 0 12 3z" /></>,
  trophy: <><path d="M7 4h10v5a5 5 0 0 1-10 0z" /><path d="M7 6H4v1.5A3.5 3.5 0 0 0 7.5 11" /><path d="M17 6h3v1.5A3.5 3.5 0 0 1 16.5 11" /><line x1="12" y1="14" x2="12" y2="18" /><path d="M8.5 21h7l-1-3h-5z" /></>,
  cube: <><path d="M12 2.5 20.5 7.3v9.4L12 21.5 3.5 16.7V7.3z" /><path d="M3.5 7.3 12 12l8.5-4.7" /><line x1="12" y1="12" x2="12" y2="21.5" /></>,
  reset: <><path d="M3 12a9 9 0 1 0 3-6.7" /><polyline points="3,3 3,8 8,8" /></>,
  upload: <><path d="M12 15V4" /><polyline points="7,9 12,4 17,9" /><path d="M4 15v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3" /></>,
  flask: <><path d="M9 3h6" /><path d="M10 3v6L4.5 19a1.5 1.5 0 0 0 1.3 2h12.4a1.5 1.5 0 0 0 1.3-2L14 9V3" /></>,
  eye: <><path d="M2 12s3.6-7 10-7 10 7 10 7-3.6 7-10 7S2 12 2 12z" /><circle cx="12" cy="12" r="3" /></>,
  chevron: <polyline points="9,6 15,12 9,18" />,
  arrow: <><line x1="5" y1="12" x2="19" y2="12" /><polyline points="13,6 19,12 13,18" /></>,
  pulse: <polyline points="2,12 6,12 9,5 13,19 16,12 22,12" />,
  grid: <><rect x="4" y="4" width="7" height="7" rx="1" /><rect x="13" y="4" width="7" height="7" rx="1" /><rect x="4" y="13" width="7" height="7" rx="1" /><rect x="13" y="13" width="7" height="7" rx="1" /></>,
  list: <><line x1="9" y1="6" x2="20" y2="6" /><line x1="9" y1="12" x2="20" y2="12" /><line x1="9" y1="18" x2="20" y2="18" /><circle cx="5" cy="6" r="1" /><circle cx="5" cy="12" r="1" /><circle cx="5" cy="18" r="1" /></>,
  filter: <polygon points="3,4 21,4 14,12.5 14,19 10,21 10,12.5" />,
  gear: <><circle cx="12" cy="12" r="3" /><path d="M19.4 15a1.6 1.6 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.6 1.6 0 0 0-1.8-.3 1.6 1.6 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1a1.6 1.6 0 0 0-1-1.5 1.6 1.6 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.6 1.6 0 0 0 .3-1.8 1.6 1.6 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1a1.6 1.6 0 0 0 1.5-1 1.6 1.6 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.6 1.6 0 0 0 1.8.3H9a1.6 1.6 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.6 1.6 0 0 0 1 1.5 1.6 1.6 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.6 1.6 0 0 0-.3 1.8V9a1.6 1.6 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.6 1.6 0 0 0-1.5 1z" /></>,
  wand: <><path d="M4 20 14 10" /><path d="M15 4v2" /><path d="M15 12v2" /><path d="M19 8h2" /><path d="M11 8H9" /><path d="M18 5l1.5-1.5" /><path d="M18 11l1.5 1.5" /></>,
  hourglass: <><path d="M6 3h12" /><path d="M6 21h12" /><path d="M7 3c0 5 10 5 10 9s-10 4-10 9" /><path d="M17 3c0 5-10 5-10 9" /></>,
};

export function I({ n, size = 20, sw = 2, style }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={sw}
      strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" style={style}>{P[n]}</svg>
  );
}
