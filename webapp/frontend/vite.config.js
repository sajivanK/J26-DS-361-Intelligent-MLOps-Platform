import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The backend runs on port 8000; every /api call from the browser is forwarded to it.
export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy: { "/api": "http://127.0.0.1:8000" } },
});
