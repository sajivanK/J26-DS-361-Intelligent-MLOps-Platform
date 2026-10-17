import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import App from "./App.jsx";
import { PlatformProvider } from "./platform.jsx";
import { SessionProvider } from "./session.jsx";
import "./styles/tokens.css";
import "./styles/base.css";

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
