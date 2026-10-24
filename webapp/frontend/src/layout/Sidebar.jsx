import { NavLink } from "react-router-dom";
import { PIPELINE_PAGES, PLATFORM_PAGES } from "../pages.js";
import { componentInfo, componentState, usePlatform } from "../platform.jsx";
import { Icon } from "./icons.jsx";
import "./Sidebar.css";

const navClass = ({ isActive }) => "nav" + (isActive ? " active" : "");

export default function Sidebar() {
  const { health, datasets, connected, checked, alertsWaiting } = usePlatform();
  const c1 = componentInfo(health, "c1");
  const enginesLoaded = c1?.engines ? Object.values(c1.engines).filter((e) => e.loaded).length : 0;

  return (
    <nav className="side" aria-label="Platform">
      <div className="brand">
        <Icon name="pulse" size={30} className="" strokeWidth={2.2} style={{ color: "var(--side-accent)" }} />
        <div><b>Intelligent MLOps</b><small>J26-DS-361</small></div>
      </div>

      <div className="navgrp">Platform</div>
      {PLATFORM_PAGES.map((p) => (
        <NavLink key={p.path} to={p.path} end={p.path === "/"} className={navClass}>
          <span className="lft"><Icon name={p.icon} size={20} /><span className="txt">{p.label}{p.sub && <small>{p.sub}</small>}</span></span>
          <Icon name="chevron" size={16} className="chev" />
        </NavLink>
      ))}

      <div className="navsep" />
      <div className="navgrp">Pipeline</div>
      {PIPELINE_PAGES.map((p) => {
        const st = componentState(health, p.component);
        return (
          <NavLink key={p.path} to={p.path} className={navClass}>
            <span className="lft"><Icon name={p.icon} size={20} /><span className="txt">{p.label}<small>C{p.n} {p.title}</small></span></span>
            {p.component === "c2" && alertsWaiting > 0
              ? <span className="alertbadge" title={`${alertsWaiting} drift alert${alertsWaiting === 1 ? "" : "s"} waiting for diagnosis`}>
                  <Icon name="warn" size={12} className="" strokeWidth={2.6} /><span className="mono">{alertsWaiting}</span></span>
              : <span className={`dot ${st}`} title={st} />}
          </NavLink>
        );
      })}

      <div className="sidefoot">
        <div className="srow"><span className={`dot ${connected ? "real" : "off"}`} />
          {connected ? "Backend connected" : checked ? "Backend not reachable" : "Checking backend…"}</div>
        {connected && c1 && (
          <div className="srow mono" style={{ fontSize: 11 }}>
            C1 engine {c1.engine_version} · {enginesLoaded} of {datasets.length} datasets
          </div>
        )}
        <div className="legend2">
          <span><i className="dot real" />real</span><span><i className="dot fake" />fake</span><span><i className="dot off" />off</span>
        </div>
      </div>
    </nav>
  );
}
