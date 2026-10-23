import { createContext, useContext, useEffect, useState } from "react";
import { api } from "./api.js";

// Backend status shared by the sidebar and every page: which components are loaded, real or fake,
// and how many drift alerts wait for C2. Checked at start, every 15 seconds, and right away after
// anything that can change the alerts (refreshAlerts()).
const PlatformContext = createContext({ health: null, datasets: [], connected: false, alertsWaiting: 0 });
const ALERTS_EVENT = "j26:alerts";
export const refreshAlerts = () => window.dispatchEvent(new Event(ALERTS_EVENT));

export function PlatformProvider({ children }) {
  const [state, setState] = useState({ health: null, datasets: [], connected: false, checked: false, alertsWaiting: 0 });
  useEffect(() => {
    let alive = true;
    const check = async () => {
      try {
        const [health, datasets] = await Promise.all([api.health(), api.datasets()]);
        const alerts = await api.alerts().catch(() => null);
        if (alive) setState({ health, datasets, connected: true, checked: true, alertsWaiting: alerts?.waiting ?? 0 });
      } catch {
        if (alive) setState((s) => ({ ...s, connected: false, checked: true }));
      }
    };
    const recount = async () => {
      const alerts = await api.alerts().catch(() => null);
      if (alive && alerts) setState((s) => ({ ...s, alertsWaiting: alerts.waiting }));
    };
    check();
    const t = setInterval(check, 15000);
    window.addEventListener(ALERTS_EVENT, recount);
    return () => { alive = false; clearInterval(t); window.removeEventListener(ALERTS_EVENT, recount); };
  }, []);
  return <PlatformContext.Provider value={state}>{children}</PlatformContext.Provider>;
}

export const usePlatform = () => useContext(PlatformContext);

// "real", "fake" or "off" (not loaded / backend down) for one component name (c1..c4).
export function componentState(health, name) {
  const c = health?.components?.find((x) => x.name === name);
  if (!c || !c.loaded) return "off";
  return c.source === "real" ? "real" : "fake";
}

export function componentInfo(health, name) {
  return health?.components?.find((x) => x.name === name) || null;
}
