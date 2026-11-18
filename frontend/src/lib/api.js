import { CONFIG } from './config.js';

// Used only when CONFIG.apiBase is set.
export const API = {
  async ocr(file, name) {
    const fd = new FormData();
    fd.append('image', file, name);
    const r = await fetch(`${CONFIG.apiBase}/api/news/ocr`, { method: 'POST', body: fd });
    if (!r.ok) throw new Error('OCR failed');
    return r.json(); // { text, script, confidence }
  },
  async verifyNews(payload) {
    const r = await fetch(`${CONFIG.apiBase}/api/news/verify`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    if (!r.ok) throw new Error('Verify failed');
    return r.json(); // { label: 'fake'|'real', confidence, domain, explanation: [] }
  }
};
