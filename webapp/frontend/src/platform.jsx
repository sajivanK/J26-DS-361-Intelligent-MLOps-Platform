import { createContext, useContext, useEffect, useState } from "react";
import { api } from "./api.js";

// Backend status shared by the sidebar and every page: which components are loaded, real or fake.
// Checked at start and every 15 seconds.
const PlatformContext = createContext({ health: null, datasets: [], connected: false });

export function PlatformProvider({ children }) {
  const [state, setState] = useState({ health: null, datasets: [], connected: false, checked: false });
  useEffect(() => {
    let alive = true;
    const check = async () => {
      try {
        const [health, datasets] = await Promise.all([api.health(), api.datasets()]);
        if (alive) setState({ health, datasets, connected: true, checked: true });
      } catch {
        if (alive) setState((s) => ({ ...s, connected: false, checked: true }));
      }
    };
    check();
    const t = setInterval(check, 15000);
    return () => { alive = false; clearInterval(t); };
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
