import { defineConfig } from "vite";

// Dev convenience only: proxy API paths to the local backend so the dev
// server stays same-origin (no CORS juggling). In production the built
// dist/ is served BY the backend, so this config is irrelevant there.
export default defineConfig({
  server: {
    proxy: {
      "/ask": "http://localhost:8000",
      "/transcribe": "http://localhost:8000",
      "/health": "http://localhost:8000",
    },
  },
});
