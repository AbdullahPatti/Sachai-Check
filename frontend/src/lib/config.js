// Sachai Check frontend config.
// Set VITE_API_BASE in a .env file (see .env.example) to use the real FastAPI pipelines,
// e.g. VITE_API_BASE=http://localhost:8000
// Leave it empty to run in demo mode (simulated OCR and verdicts).
const base = (import.meta.env.VITE_API_BASE || '').trim().replace(/\/+$/, '');

export const CONFIG = {
  apiBase: base || null,
  maxImageMB: Number(import.meta.env.VITE_MAX_IMAGE_MB) || 10
};

export const LIVE = !!CONFIG.apiBase;
