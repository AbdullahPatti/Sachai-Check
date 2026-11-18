import { useEffect, useRef, useState } from 'react';
import Segmented from '../components/Segmented.jsx';
import { Arrow, Mic, Pause, Play, Upload } from '../components/Icons.jsx';
import { StepRunner, EmptySide, CopyButton, DemoSwitch, PageHead, PageFoot, scrollSideIntoView } from '../components/Shared.jsx';
import { AUDIO_OUT } from '../lib/data.js';
import { store, fmtTime } from '../lib/utils.js';

const BARS = typeof window !== 'undefined' && window.innerWidth < 640 ? 56 : 96;
const SRC_OPTS = [{ v: 'upload', label: 'Upload file' }, { v: 'record', label: 'Record' }];
const STEPS = ['Resampling and trimming audio', 'Scoring each segment', 'Writing the explanation'];

function demoPeaks(n) {
  const p = [];
  for (let i = 0; i < n; i++) p.push(0.14 + Math.abs(Math.sin(i * 0.37 * 96 / n) * Math.cos(i * 0.11 * 96 / n)) * 0.78 + (i % 5) * 0.02);
  return p;
}
const flaggedIdx = (i, n) => { const x = i / n; return (x > 0.32 && x < 0.48) || (x > 0.69 && x < 0.81); };

export default function Audio({ active }) {
  const [src, setSrcState] = useState('upload');
  const [outcome, setOutcome] = useState(() => store.get('sc1-audio') || 'synthetic');
  const [peaks, setPeaks] = useState(null);
  const [hasBuffer, setHasBuffer] = useState(false);
  const [fname, setFname] = useState('sample_voice_note.wav');
  const [fmeta, setFmeta] = useState('Demo clip');
  const [analysed, setAnalysed] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [playK, setPlayK] = useState(-1);
  const [over, setOver] = useState(false);
  const [recording, setRecording] = useState(false);
  const [secs, setSecs] = useState(0);
  const [recNote, setRecNote] = useState('');
  const [side, setSide] = useState({ phase: 'empty', run: 0 });

  const buffer = useRef(null);
  const ctxRef = useRef(null);
  const node = useRef(null);
  const raf = useRef(0);
  const demoStop = useRef(0);
  const media = useRef(null);
  const chunks = useRef([]);
  const recTimer = useRef(null);
  const secsRef = useRef(0);
  const fileRef = useRef(null);
  const sideRef = useRef(null);
  const toggleRecRef = useRef(null);

  const getCtx = () => {
    if (!ctxRef.current) {
      const C = window.AudioContext || window.webkitAudioContext;
      if (!C) return null;
      ctxRef.current = new C();
    }
    return ctxRef.current;
  };

  function stopPlayback() {
    if (node.current) { try { node.current.onended = null; node.current.stop(); } catch (e) { /* ignore */ } node.current = null; }
    cancelAnimationFrame(raf.current);
    clearTimeout(demoStop.current);
    setPlaying(false); setPlayK(-1);
  }

  useEffect(() => { if (!active) stopPlayback(); }, [active]);
  useEffect(() => () => { stopPlayback(); clearInterval(recTimer.current); }, []);

  function togglePlay() {
    if (playing) { stopPlayback(); return; }
    setPlaying(true);
    const buf = buffer.current;
    if (buf) {
      const ctx = getCtx(); if (!ctx) return;
      if (ctx.resume) ctx.resume();
      const n = ctx.createBufferSource();
      n.buffer = buf; n.connect(ctx.destination);
      const t0 = ctx.currentTime; n.start(); node.current = n;
      n.onended = () => stopPlayback();
      const step = () => {
        const p = (ctx.currentTime - t0) / buf.duration;
        setPlayK(Math.floor(p * BARS));
        raf.current = requestAnimationFrame(step);
      };
      step();
    } else {
      demoStop.current = setTimeout(stopPlayback, 6000);
    }
  }

  function showEmpty() { setSide((s) => ({ phase: 'empty', run: s.run + 1 })); }

  async function loadArrayBuffer(ab, name) {
    stopPlayback();
    setFname(name); setFmeta('Reading audio…');
    try {
      const ctx = getCtx();
      if (!ctx) throw new Error('no audio context');
      const buf = await new Promise((res, rej) => { const p = ctx.decodeAudioData(ab, res, rej); if (p && p.then) p.then(res, rej); });
      buffer.current = buf;
      const data = buf.getChannelData(0), size = Math.floor(data.length / BARS) || 1, pk = [];
      for (let i = 0; i < BARS; i++) {
        let m = 0;
        for (let j = i * size; j < (i + 1) * size && j < data.length; j += 32) m = Math.max(m, Math.abs(data[j]));
        pk.push(m);
      }
      const max = Math.max(...pk, 0.01);
      setPeaks(pk.map((p) => 0.08 + p / max * 0.92));
      setHasBuffer(true);
      setFmeta(`${fmtTime(buf.duration)} · ${(buf.sampleRate / 1000).toFixed(1)} kHz`);
    } catch (e) {
      buffer.current = null; setPeaks(null); setHasBuffer(false);
      setFmeta('This format could not be read here. Try WAV or MP3.');
    }
    setAnalysed(false); showEmpty();
  }

  const loadFile = (f) => f.arrayBuffer().then((ab) => loadArrayBuffer(ab, f.name));

  /* ---------- recording ---------- */
  function stopRec() {
    clearInterval(recTimer.current); recTimer.current = null;
    setRecording(false);
    if (media.current && media.current.state !== 'inactive') media.current.stop();
  }

  async function toggleRec() {
    if (recTimer.current) {
      stopRec();
      if (!media.current) {
        buffer.current = null; setPeaks(null); setHasBuffer(false);
        setFname('recording_demo.wav'); setFmeta(`${fmtTime(secsRef.current)} · simulated`);
        setAnalysed(false); showEmpty();
      }
      return;
    }
    secsRef.current = 0; setSecs(0); setRecNote('');
    media.current = null; chunks.current = [];
    try {
      if (!navigator.mediaDevices || !window.MediaRecorder) throw new Error('unsupported');
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mr = new MediaRecorder(stream); media.current = mr;
      mr.ondataavailable = (e) => { if (e.data.size) chunks.current.push(e.data); };
      mr.onstop = () => {
        stream.getTracks().forEach((t) => t.stop());
        const blob = new Blob(chunks.current, { type: mr.mimeType });
        blob.arrayBuffer().then((ab) => loadArrayBuffer(ab, 'recording.webm'));
        media.current = null;
      };
      mr.start();
    } catch (e) {
      setRecNote('Microphone access is not available here, so this recording is simulated.');
    }
    setRecording(true);
    recTimer.current = setInterval(() => {
      secsRef.current++; setSecs(secsRef.current);
      if (secsRef.current >= 60) toggleRecRef.current();
    }, 1000);
  }
  toggleRecRef.current = toggleRec;

  const setSrc = (v) => { setSrcState(v); if (v !== 'record') stopRec(); };

  /* ---------- verify ---------- */
  function verify() {
    stopPlayback();
    setSide((s) => ({ phase: 'loading', run: s.run + 1 }));
    scrollSideIntoView(sideRef);
  }
  function done(run) { setAnalysed(true); setSide({ phase: 'result', run }); }

  /* ---------- render ---------- */
  const syn = outcome === 'synthetic';
  const wavePeaks = peaks || demoPeaks(BARS);
  const o = AUDIO_OUT[outcome];
  const dur = buffer.current ? fmtTime(buffer.current.duration) : '00:14';
  const report = () => `Sachai Check · Audio deepfake\nFile: ${fname}\nVerdict: ${o.en} (${o.ur})\nSpoof score: ${(o.w / 100).toFixed(2)} (demo)\n\n${o.why.map((t, i) => `${i + 1}. ${t}`).join('\n')}`;
  const k = `${side.run}-${outcome}`;

  return (
    <>
      <PageHead kicker="Mode 2 · Speech pipeline" title="Audio Deepfake Detection" ur="آڈیو ڈیپ فیک کی جانچ" />
      <div className="wrap grid2">
        <div className="panel dark r" style={{ '--d': '.2s' }}>
          <div className="row">
            <span className="lbl">Source</span>
            <Segmented options={SRC_OPTS} value={src} onChange={setSrc} />
          </div>

          <label className={'drop fade' + (over ? ' over' : '')} htmlFor="audioFile" hidden={src !== 'upload'}
            onDragEnter={(e) => { e.preventDefault(); setOver(true); }}
            onDragOver={(e) => { e.preventDefault(); setOver(true); }}
            onDragLeave={(e) => { e.preventDefault(); setOver(false); }}
            onDrop={(e) => { e.preventDefault(); setOver(false); const f = e.dataTransfer.files[0]; if (f && f.type.startsWith('audio')) loadFile(f); }}>
            <span className="up"><Upload /></span>
            <strong>Drop an audio file or browse</strong>
            <span className="ur" lang="ur">آڈیو فائل یہاں چھوڑیں</span>
            <small>WAV · MP3 · M4A · OGG</small>
          </label>
          <input id="audioFile" ref={fileRef} className="sr" type="file" accept="audio/*"
            onChange={(e) => { const f = e.target.files[0]; if (f) loadFile(f); e.target.value = ''; }} />

          {src === 'record' && (
            <div className="rec fade">
              <button type="button" className={'recbtn' + (recording ? ' live' : '')} aria-label={recording ? 'Stop recording' : 'Start recording'} onClick={toggleRec}>
                <Mic /><span className="sq" />
              </button>
              <span className="clock">{fmtTime(secs)}</span>
              <span className="ur" lang="ur">{recording ? 'ریکارڈنگ جاری ہے' : 'ریکارڈنگ شروع کرنے کے لیے دبائیں'}</span>
              <small>{recNote}</small>
            </div>
          )}

          <div className="clip">
            <div className="row">
              <div style={{ display: 'flex', alignItems: 'center', gap: 14, minWidth: 0 }}>
                <button type="button" className="play" aria-label={playing ? 'Pause clip' : 'Play clip'} onClick={togglePlay}>
                  {playing ? <Pause /> : <Play />}
                </button>
                <div className="fname"><b>{fname}</b><span>{fmeta}</span></div>
              </div>
              <button type="button" className="ghost-s" onClick={() => { setSrc('upload'); fileRef.current && fileRef.current.click(); }}>Replace</button>
            </div>
            <div className={'wave' + (playing && !hasBuffer ? ' playing' : '')}>
              {wavePeaks.map((p, i) => {
                const f = syn && analysed && flaggedIdx(i, BARS);
                const pl = playing && hasBuffer && !f && i <= playK;
                return (
                  <b key={i} className={(f ? 'f' : '') + (pl ? ' pl' : '')}
                    style={{ height: `${Math.max(6, Math.round(p * 68))}px`, animationDelay: `${((i % 12) * 0.07).toFixed(2)}s` }} />
                );
              })}
            </div>
          </div>

          <div className="actions">
            <span className="hint">Urdu speech only</span>
            <button type="button" className="btn btn-lime" style={{ border: 'none' }} onClick={verify}>
              Analyse voice <span className="ur" lang="ur">آواز جانچیں</span><Arrow />
            </button>
          </div>
        </div>

        <div className="side r" style={{ '--d': '.3s' }} ref={sideRef} aria-live="polite">
          {side.phase === 'empty' && (
            <EmptySide>Upload or record a clip, then press Analyse voice. You'll see the verdict, the segments that raised flags and the judge's explanation.</EmptySide>
          )}
          {side.phase === 'loading' && (
            <StepRunner key={side.run} urLabel="آواز کی جانچ جاری ہے" labels={STEPS} onDone={() => done(side.run)} />
          )}
          {side.phase === 'result' && (
            <>
              <div className={`verdict ${o.cls} fade`} key={'v' + k}>
                <div className="meta"><span>Verdict · demo data</span></div>
                <div><h2>{o.en}</h2><p className="ur v" lang="ur">{o.ur}</p></div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                  <div className="meta"><span>Spoof score</span><span style={{ fontFamily: 'var(--mono)' }}>{(o.w / 100).toFixed(2)} demo</span></div>
                  <div className="bar"><i style={{ width: `${o.w}%` }} /></div>
                </div>
              </div>
              <div className="why fade" key={'w' + k}>
                <div className="row"><h3 style={{ display: 'block' }}>Suspicious segments</h3><span className="legend"><i />flagged</span></div>
                <div className="segs">
                  {Array.from({ length: 24 }, (_, i) => (
                    <i key={i} className={syn && flaggedIdx(i, 24) ? 'f' : ''} style={{ animationDelay: `${(0.15 + i * 0.025).toFixed(3)}s` }} />
                  ))}
                </div>
                <div className="times"><span>00:00</span><span>{dur}</span></div>
                <h3 style={{ marginTop: 6 }}>Why this verdict <span className="ur" lang="ur">وضاحت</span></h3>
                <ol>{o.why.map((t, i) => <li key={i} style={{ animationDelay: `${0.25 + i * 0.1}s` }}>{t}</li>)}</ol>
                <p className="note">Written by the LLM judge from the model's output. It explains, it does not decide.</p>
              </div>
              <div className="btnrow">
                <CopyButton getText={report} />
                <button type="button" className="btn btn-ink" onClick={verify}>Run again</button>
              </div>
              <DemoSwitch a="synthetic" b="genuine" labelA="Synthetic" labelB="Genuine" value={outcome}
                onChange={(v) => { setOutcome(v); store.set('sc1-audio', v); }} />
            </>
          )}
        </div>
      </div>
      <PageFoot />
    </>
  );
}
