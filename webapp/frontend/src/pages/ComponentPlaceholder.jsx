import { api } from "../api.js";
import { useApi } from "../hooks.js";
import { componentInfo, componentState, usePlatform } from "../platform.jsx";
import { JsonBox, PageHeader, SourceBadge } from "../ui/parts.jsx";

// Shown for C2, C3 and C4 until the owner builds the real page (see the UI guide shared in the team group).
const OUTPUTS = { c2: ["c2_to_c3", "c2_to_c4"], c3: ["c3"], c4: ["c4"] };

export default function ComponentPlaceholder({ step }) {
  const { health } = usePlatform();
  const st = componentState(health, step.component);
  const comp = componentInfo(health, step.component);
  const latest = useApi(api.latestRun, []);
  const outs = latest.data?.outputs || {};
  const mine = Object.fromEntries((OUTPUTS[step.component] || []).filter((k) => outs[k]).map((k) => [k, outs[k]]));
  return (
    <>
      <PageHeader eyebrow={`Component ${step.n} · ${step.owner}`} title={`C${step.n} ${step.title}`} subtitle={step.question} />
      <div className="card stack">
        <div className="row"><h2>Page not built yet</h2><SourceBadge source={st} /></div>
        <p className="muted">{step.owner} builds this page. The component itself already runs in the pipeline
          {st === "fake" ? " as a fake made from the contract example" : st === "real" ? " with real code" : ""}
          {comp?.error ? ` (error: ${comp.error})` : ""}. How to build the page: the UI guide shared in the team group.</p>
      </div>
      <JsonBox title="This component's output in the latest pipeline run" data={Object.keys(mine).length ? mine : null}
        note="Run the full pipeline in the Drift Lab to fill this." />
    </>
  );
}
