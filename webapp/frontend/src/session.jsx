import { createContext, useContext, useMemo, useState } from "react";

// Results kept while the browser tab is open, so they survive moving between pages.
// (Not saved anywhere: a page reload starts a new session.)
// Each page gets its own slot: const [runs, setRuns] = useSession("c1:uci", []).
const SessionContext = createContext(null);

export function SessionProvider({ children }) {
  const [store, setStore] = useState({});
  const value = useMemo(() => ({ store, setStore }), [store]);
  return <SessionContext.Provider value={value}>{children}</SessionContext.Provider>;
}

export function useSession(key, initial) {
  const { store, setStore } = useContext(SessionContext);
  const value = key in store ? store[key] : initial;
  const set = (next) =>
    setStore((s) => ({ ...s, [key]: typeof next === "function" ? next(key in s ? s[key] : initial) : next }));
  return [value, set];
}
