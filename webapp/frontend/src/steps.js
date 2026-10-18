// The four pipeline steps, in order. `component` matches the backend name in /api/health (c1..c4).
// Pages read this for titles and owners; src/pages.js attaches each step's page.
export const STEPS = [
  { component: "c1", n: 1, path: "/c1", label: "Detect", title: "Drift Detection", owner: "Sajivan K",
    icon: "detect", question: "Did the data drift, and will it hurt the model?", output: "harm · interval · extent" },
  { component: "c2", n: 2, path: "/c2", label: "Diagnose", title: "Root Cause Diagnosis", owner: "Tharmika N",
    icon: "diagnose", question: "Why did it happen, and which slice is affected?", output: "root cause · affected slice" },
  { component: "c3", n: 3, path: "/c3", label: "Repair", title: "Synthetic Data Repair", owner: "Mathusan K",
    icon: "repair", question: "What repair data fixes the affected slice?", output: "repair dataset · quality scores" },
  { component: "c4", n: 4, path: "/c4", label: "Release", title: "Release Control", owner: "Kesigan M",
    icon: "release", question: "Is the retrained model safe to release?", output: "AUTO_DEPLOY / HUMAN_REVIEW / ESCALATE" },
];

export const TEAM = [
  { name: "Sajivan K", role: "C1 Drift Detection · lead" },
  { name: "Tharmika N", role: "C2 Root Cause Diagnosis" },
  { name: "Mathusan K", role: "C3 Synthetic Data Repair" },
  { name: "Kesigan M", role: "C4 Release Control" },
];
