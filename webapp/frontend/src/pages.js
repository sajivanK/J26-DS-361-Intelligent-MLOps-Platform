// The page registry: the sidebar and the routes are both built from these two lists.
// To add your component page: create it under src/pages/, import it here and put it in PAGE_BY_STEP.
import C1Page from "./pages/c1/C1Page.jsx";
import DriftLab from "./pages/DriftLab.jsx";
import Overview from "./pages/Overview.jsx";
import { STEPS } from "./steps.js";

export const PLATFORM_PAGES = [
  { path: "/", label: "Overview", icon: "overview", page: Overview },
  { path: "/lab", label: "Drift Lab", sub: "test the system yourself", icon: "lab", page: DriftLab },
];

// component -> page. A step without a page shows a placeholder until its owner builds it.
const PAGE_BY_STEP = {
  c1: C1Page,
  // c2: C2Page,   (Tharmika)
  // c3: C3Page,   (Mathusan)
  // c4: C4Page,   (Kesigan)
};

export const PIPELINE_PAGES = STEPS.map((s) => ({ ...s, page: PAGE_BY_STEP[s.component] || null }));
