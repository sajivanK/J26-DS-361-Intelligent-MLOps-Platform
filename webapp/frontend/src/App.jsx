import { Route, Routes } from "react-router-dom";
import Sidebar from "./layout/Sidebar.jsx";
import { PIPELINE_PAGES, PLATFORM_PAGES } from "./pages.js";
import ComponentPlaceholder from "./pages/ComponentPlaceholder.jsx";

export default function App() {
  return (
    <div className="app">
      <Sidebar />
      <main className="main">
        <Routes>
          {PLATFORM_PAGES.map(({ path, page: Page }) => <Route key={path} path={path} element={<Page />} />)}
          {PIPELINE_PAGES.map((p) => (
            <Route key={p.path} path={p.path}
              element={p.page ? <p.page /> : <ComponentPlaceholder step={p} />} />
          ))}
          <Route path="*" element={<div className="empty">Page not found.</div>} />
        </Routes>
      </main>
    </div>
  );
}
