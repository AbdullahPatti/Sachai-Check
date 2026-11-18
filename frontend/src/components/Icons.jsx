export const Arrow = ({ stroke = '#1F2430' }) => (
  <svg className="arr" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke={stroke} strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round"><path d="M5 12h14M13 6l6 6-6 6" /></svg>
);

export const Check = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#1F2430" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"><path d="M5 12.5l4.5 4.5L19 7.5" /></svg>
);

export const Play = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="#1F2430"><path d="M7 4l13 8-13 8z" /></svg>
);

export const Pause = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="#1F2430"><rect x="6" y="4" width="4" height="16" rx="1" /><rect x="14" y="4" width="4" height="16" rx="1" /></svg>
);

export const ImageIcon = ({ size = 16, stroke = 'currentColor' }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={stroke} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="3" y="4" width="18" height="16" rx="2" /><circle cx="9" cy="10" r="2" /><path d="M21 16l-5-5-9 9" /></svg>
);

export const TextIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M4 7V5h16v2M9 19h6M12 5v14" /></svg>
);

export const Upload = () => (
  <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#1F2430" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 16V4M6 10l6-6 6 6M4 20h16" /></svg>
);

export const Mic = () => (
  <svg width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="#1F2430" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="9" y="3" width="6" height="11" rx="3" /><path d="M5 11a7 7 0 0 0 14 0M12 18v3" /></svg>
);
