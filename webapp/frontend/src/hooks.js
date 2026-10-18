import { useCallback, useEffect, useState } from "react";

// Load data from the backend when a page opens (and again when `deps` change).
// Returns { data, error, loading, reload }.
export function useApi(fn, deps = []) {
  const [state, setState] = useState({ data: null, error: null, loading: true });
  const load = useCallback(() => {
    let alive = true;
    setState((s) => ({ ...s, loading: true, error: null }));
    fn().then(
      (data) => alive && setState({ data, error: null, loading: false }),
      (e) => alive && setState({ data: null, error: e.message, loading: false }),
    );
    return () => { alive = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
  useEffect(load, [load]);
  return { ...state, reload: load };
}

// Run an action on click (detect, run pipeline, ...). Returns [run, { busy, error }].
export function useAction(fn) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const run = useCallback(async (...args) => {
    setBusy(true); setError(null);
    try { return await fn(...args); }
    catch (e) { setError(e.message); return null; }
    finally { setBusy(false); }
  }, [fn]);
  return [run, { busy, error, setError }];
}
