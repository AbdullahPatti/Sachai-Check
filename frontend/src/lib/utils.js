import { NASTALIQ_SAMPLE } from './data.js';

export const store = {
  get(k) { try { return localStorage.getItem(k); } catch (e) { return null; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch (e) { /* ignore */ } }
};

export const wait = (ms) => new Promise((r) => setTimeout(r, ms));

export const reducedMotion = () =>
  typeof matchMedia !== 'undefined' && matchMedia('(prefers-reduced-motion: reduce)').matches;

export function fmtBytes(b) {
  return b > 1048576 ? (b / 1048576).toFixed(1) + ' MB' : Math.max(1, Math.round(b / 1024)) + ' KB';
}

export function fmtTime(s) {
  s = Math.max(0, Math.round(s));
  return String(Math.floor(s / 60)).padStart(2, '0') + ':' + String(s % 60).padStart(2, '0');
}

export function copyText(text) {
  const fallback = () => new Promise((res, rej) => {
    const t = document.createElement('textarea');
    t.value = text;
    document.body.appendChild(t);
    t.select();
    try { document.execCommand('copy'); res(); } catch (e) { rej(e); }
    t.remove();
  });
  try { return navigator.clipboard.writeText(text).catch(fallback); } catch (e) { return fallback(); }
}

function roundRect(x, l, t, w, h, r) {
  x.beginPath();
  x.moveTo(l + r, t);
  x.arcTo(l + w, t, l + w, t + h, r);
  x.arcTo(l + w, t + h, l, t + h, r);
  x.arcTo(l, t + h, l, t, r);
  x.arcTo(l, t, l + w, t, r);
  x.closePath();
}

// Sample image: a forwarded-post style card drawn on canvas.
export async function makeSampleImage() {
  try {
    if (document.fonts) {
      await document.fonts.load('700 46px "Noto Nastaliq Urdu"');
      await document.fonts.load('600 26px "IBM Plex Sans"');
    }
  } catch (e) { /* ignore */ }
  const W = 1080, H = 760, c = document.createElement('canvas');
  c.width = W; c.height = H;
  const x = c.getContext('2d');
  x.fillStyle = '#F7F0E4'; x.fillRect(0, 0, W, H);
  x.fillStyle = '#1F2430'; x.fillRect(0, 0, W, 130);
  x.fillStyle = '#C9F03A'; roundRect(x, W - 330, 34, 290, 62, 31); x.fill();
  x.fillStyle = '#1F2430'; x.direction = 'rtl'; x.textAlign = 'center'; x.textBaseline = 'middle';
  x.font = '700 30px "Noto Nastaliq Urdu", serif'; x.fillText('بریکنگ نیوز', W - 185, 66);
  x.direction = 'ltr'; x.textAlign = 'left'; x.fillStyle = '#F7F0E4';
  x.font = '600 26px "IBM Plex Sans", sans-serif'; x.fillText('Forwarded many times', 40, 66);
  x.direction = 'rtl'; x.textAlign = 'right'; x.fillStyle = '#1F2430';
  x.font = '700 46px "Noto Nastaliq Urdu", serif';
  const words = NASTALIQ_SAMPLE.split(' '), lines = [];
  let line = '';
  words.forEach((w) => {
    const t = line ? line + ' ' + w : w;
    if (x.measureText(t).width > W - 120) { lines.push(line); line = w; } else line = t;
  });
  if (line) lines.push(line);
  lines.forEach((l, i) => x.fillText(l, W - 60, 250 + i * 108));
  x.fillStyle = '#CDBFA8'; x.fillRect(60, H - 110, W - 120, 2);
  x.fillStyle = '#5E6B82'; x.font = '400 28px "Noto Nastaliq Urdu", serif';
  x.fillText('اسے زیادہ سے زیادہ شیئر کریں', W - 60, H - 58);
  return new Promise((res) => c.toBlob((b) => res(b), 'image/png'));
}
