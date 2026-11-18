import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react';

const LINKS = [
  ['home', '#/', 'Home'],
  ['news', '#/news', 'Fake News'],
  ['audio', '#/audio', 'Audio Deepfake']
];

export default function Nav({ route }) {
  const refs = useRef({});
  const [ind, setInd] = useState({ left: 0, width: 0 });

  const measure = useCallback(() => {
    const el = refs.current[route];
    if (el) setInd({ left: el.offsetLeft, width: el.offsetWidth });
  }, [route]);

  useLayoutEffect(measure, [measure]);
  useEffect(() => {
    window.addEventListener('resize', measure);
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(measure);
    return () => window.removeEventListener('resize', measure);
  }, [measure]);

  return (
    <header className="nav">
      <div className="wrap">
        <a className="brand" href="#/">
          <span className="logo">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#1F2430" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round"><path d="M4 12.5l5 5L20 6.5" /></svg>
          </span>
          <b>Sachai Check</b><span className="ur" lang="ur">سچائی چیک</span>
        </a>
        <nav className="tabs" aria-label="Modes">
          <span className="ind" aria-hidden="true" style={{ left: ind.left, width: ind.width }} />
          {LINKS.map(([name, href, label]) => (
            <a key={name} href={href} ref={(el) => { refs.current[name] = el; }} aria-current={route === name ? 'page' : undefined}>
              {label}
            </a>
          ))}
        </nav>
        <span className="navtag">FAST-NUCES · FYP 2026</span>
      </div>
    </header>
  );
}
