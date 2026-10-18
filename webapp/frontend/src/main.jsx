import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
// shared styles first, so page styles (imported by the pages) can override them
import "./styles/tokens.css";
import "./styles/base.css";
import App from "./App.jsx";
import { PlatformProvider } from "./platform.jsx";
import { SessionProvider } from "./session.jsx";

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <BrowserRouter>
      <PlatformProvider>
        <SessionProvider>
          <App />
        </SessionProvider>
      </PlatformProvider>
    </BrowserRouter>
  </React.StrictMode>,
);
