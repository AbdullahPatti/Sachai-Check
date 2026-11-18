import { useEffect, useRef, useState } from 'react';
import { Check } from './Icons.jsx';
import { copyText } from '../lib/utils.js';

// Animated step list shown while a check runs. Calls onDone after the last step.
export function StepRunner({ urLabel, labels, onDone }) {
  const [i, setI] = useState(0);
  const doneRef = useRef(onDone);
  doneRef.current = onDone;

  useEffect(() => {
    if (i > labels.length) return undefined;
    if (i === labels.length) {
      const t = setTimeout(() => { setI(i + 1); doneRef.current(); }, 350);
      return () => clearTimeout(t);
    }
    const t = setTimeout(() => setI(i + 1), 760);
    return () => clearTimeout(t);
  }, [i, labels.length]);

  return (
    <div className="loading fade">
      <div className="top"><span>Analysing</span><span className="ur" lang="ur">{urLabel}</span></div>
      <div className="stl" style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
        {labels.map((l, k) => (
          <div key={l} className={'st' + (k < i ? ' ok' : k === i ? ' on' : '')}>
            <span className="ic">
              {k < i ? <span className="ck"><Check /></span> : k === i ? <span className="sp" /> : <span className="pd" />}
            </span>
            <span>{l}</span>
          </div>
        ))}
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 12, paddingTop: 6 }}>
        <span className="shim" style={{ width: '92%' }} />
        <span className="shim" style={{ width: '78%' }} />
        <span className="shim" style={{ width: '62%' }} />
      </div>
    </div>
  );
}

export function EmptySide({ children }) {
  return (
    <div className="empty">
      <h3>Your verdict appears here</h3>
      <p className="ur" lang="ur" style={{ fontSize: 20, lineHeight: 2, color: 'var(--pigeon)', alignSelf: 'stretch', textAlign: 'right', margin: 0 }}>نتیجہ یہاں ظاہر ہوگا</p>
      <p>{children}</p>
    </div>
  );
}

export function ErrorSide() {
  return (
    <div className="empty fade">
      <h3>Something went wrong</h3>
      <p>The server did not return a verdict. Check that the backend is running, then try again.</p>
    </div>
  );
}

export function CopyButton({ getText }) {
  const [copied, setCopied] = useState(false);
  useEffect(() => {
    if (!copied) return undefined;
    const t = setTimeout(() => setCopied(false), 1600);
    return () => clearTimeout(t);
  }, [copied]);
  return (
    <button type="button" className="btn btn-ghost" onClick={() => copyText(getText()).then(() => setCopied(true), () => {})}>
      {copied ? 'Copied' : 'Copy report'}
    </button>
  );
}

export function DemoSwitch({ a, b, labelA, labelB, value, onChange }) {
  return (
    <div className="demo">
      <span>Demo outcome</span>
      <div className="mini-seg">
        <button type="button" aria-pressed={value === a} onClick={() => onChange(a)}>{labelA}</button>
        <button type="button" aria-pressed={value === b} onClick={() => onChange(b)}>{labelB}</button>
      </div>
    </div>
  );
}

export function PageHead({ kicker, title, ur }) {
  return (
    <div className="wrap phead">
      <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
        <span className="kicker r">{kicker}</span>
        <h1 className="display r" style={{ '--d': '.08s' }}>{title}</h1>
      </div>
      <p className="ur r" lang="ur" style={{ '--d': '.16s' }}>{ur}</p>
    </div>
  );
}

export function PageFoot() {
  return (
    <footer className="pfoot">
      <div className="wrap">
        <span>Research prototype. Treat every verdict as a signal, not proof.</span>
        <span className="ur" lang="ur">نتیجہ ایک اشارہ ہے، حتمی ثبوت نہیں</span>
      </div>
    </footer>
  );
}

// Scroll the result column into view on narrow screens when a check starts.
export function scrollSideIntoView(ref) {
  if (window.innerWidth < 981 && ref.current) ref.current.scrollIntoView({ behavior: 'smooth', block: 'start' });
}
