import { useEffect, useRef, useState } from 'react';
import Segmented from '../components/Segmented.jsx';
import { Arrow, ImageIcon, TextIcon } from '../components/Icons.jsx';
import { StepRunner, EmptySide, ErrorSide, CopyButton, DemoSwitch, PageHead, PageFoot, scrollSideIntoView } from '../components/Shared.jsx';
import { CONFIG, LIVE } from '../lib/config.js';
import { API } from '../lib/api.js';
import { DOMAINS, NEWS_OUT, NASTALIQ_SAMPLE, ROMAN_SAMPLE, ROMAN_TRANSLIT } from '../lib/data.js';
import { store, wait, fmtBytes, makeSampleImage } from '../lib/utils.js';

const REFADE = { animation: 'fade .4s ease both' };
const TRANSLIT_PENDING = 'Transliteration appears here once the backend is connected.';

const INPUT_OPTS = [
  { v: 'text', label: <><TextIcon />Type text</> },
  { v: 'image', label: <><ImageIcon />Upload image</> }
];
const SCRIPT_OPTS = [
  { v: 'nastaliq', label: <span className="ur" lang="ur">اردو نستعلیق</span> },
  { v: 'roman', label: 'Roman Urdu' }
];

const scriptProps = (script) => script === 'roman'
  ? { className: 'latin', dir: 'ltr', lang: 'en' }
  : { className: 'urdu', dir: 'rtl', lang: 'ur' };

export default function News({ active }) {
  const [input, setInputState] = useState('text');
  const [script, setScript] = useState('nastaliq');
  const [texts, setTexts] = useState({ nastaliq: NASTALIQ_SAMPLE, roman: ROMAN_SAMPLE });
  const [domain, setDomain] = useState('politics');
  const [outcome, setOutcome] = useState(() => store.get('sc1-news') || 'fake');

  const [image, setImage] = useState(null); // { file, name, url, w, h, size, sample }
  const [imgErr, setImgErr] = useState('');
  const [over, setOver] = useState(false);
  const [ocr, setOcr] = useState(null); // { busy, text, script, failed, note }
  const ocrRun = useRef(0);

  const [side, setSide] = useState({ phase: 'empty', run: 0 }); // phase: empty | loading | result | error
  const [result, setResult] = useState(null);
  const pending = useRef({});
  const sideRef = useRef(null);
  const fileRef = useRef(null);

  const text = texts[script];
  const ocrText = ocr ? ocr.text : '';
  const checkedText = (input === 'image' ? ocrText : text).trim();

  // status line + verify button state
  let count, canVerify;
  if (input === 'text') {
    const n = text.trim().length; canVerify = n > 0;
    count = n ? `${n} characters · typed text` : 'Paste or type a news text to begin';
  } else if (!image) { canVerify = false; count = 'Upload or paste an image to begin'; }
  else if (!ocr || ocr.busy) { canVerify = false; count = 'Reading text from the image…'; }
  else {
    const n = ocrText.trim().length; canVerify = n > 0;
    count = n ? `${n} characters read · edit before verifying` : 'No text found. Type it in the box above.';
  }

  const resetSide = () => { pending.current = {}; setSide((s) => ({ phase: 'empty', run: s.run + 1 })); setResult(null); };

  const setInput = (v) => { if (v !== input) { setInputState(v); resetSide(); } };

  /* ---------- image + OCR ---------- */
  async function runOcr(im) {
    const run = ++ocrRun.current;
    setOcr({ busy: true, text: '', script: 'nastaliq', failed: false, note: '' });
    let t = '', sc = 'nastaliq', note = '', failed = false;
    try {
      if (LIVE) {
        const r = await API.ocr(im.file, im.name);
        t = r.text || ''; sc = r.script || 'nastaliq';
      } else {
        await wait(1600);
        t = NASTALIQ_SAMPLE;
        note = im.sample
          ? 'Demo OCR output for the sample image.'
          : 'Demo OCR output. The real text from your image appears once the OCR backend is connected.';
      }
    } catch (e) { failed = true; }
    if (run !== ocrRun.current) return;
    if (failed) note = 'The image could not be read. Type the text in the box, or try a sharper image.';
    setOcr({ busy: false, text: t, script: sc, failed, note });
  }

  function loadImage(file, name, sample = false) {
    setImgErr('');
    if (!file || !(file.type || '').startsWith('image/')) { setImgErr('That file is not an image. Use PNG, JPG or WEBP.'); return; }
    if (file.size > CONFIG.maxImageMB * 1048576) { setImgErr(`Image is larger than ${CONFIG.maxImageMB} MB.`); return; }
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      const im = { file, name: name || file.name || 'image.png', url, w: img.naturalWidth, h: img.naturalHeight, size: file.size, sample };
      setImage((prev) => { if (prev) URL.revokeObjectURL(prev.url); return im; });
      resetSide();
      runOcr(im);
    };
    img.onerror = () => { URL.revokeObjectURL(url); setImgErr('This image could not be opened. Try another file.'); };
    img.src = url;
  }

  function removeImage() {
    ocrRun.current++;
    if (image) URL.revokeObjectURL(image.url);
    setImage(null); setOcr(null);
    resetSide();
  }

  // Ctrl+V pastes a screenshot while this view is open
  const pasteRef = useRef(null);
  pasteRef.current = (e) => {
    const item = Array.from((e.clipboardData && e.clipboardData.items) || []).find((i) => i.type && i.type.startsWith('image/'));
    if (!item) return;
    e.preventDefault();
    if (input !== 'image') setInput('image');
    loadImage(item.getAsFile(), 'pasted_image.png');
  };
  useEffect(() => {
    if (!active) return undefined;
    const h = (e) => pasteRef.current(e);
    document.addEventListener('paste', h);
    return () => document.removeEventListener('paste', h);
  }, [active]);

  const dragProps = {
    onDragEnter: (e) => { e.preventDefault(); setOver(true); },
    onDragOver: (e) => { e.preventDefault(); setOver(true); },
    onDragLeave: (e) => { e.preventDefault(); setOver(false); },
    onDrop: (e) => { e.preventDefault(); setOver(false); const f = e.dataTransfer.files[0]; if (f) loadImage(f); }
  };

  /* ---------- verify ---------- */
  const isImg = input === 'image';
  const usedScript = isImg ? (ocr ? ocr.script : 'nastaliq') : script;
  const labels = isImg
    ? ['Cleaning OCR text', usedScript === 'roman' ? 'Transliterating to Nastaliq' : 'Normalising Nastaliq text', 'Running the classifier', 'Writing the explanation']
    : usedScript === 'roman'
      ? ['Transliterating to Nastaliq', 'Running the classifier', 'Writing the explanation']
      : ['Normalising Nastaliq text', 'Running the classifier', 'Writing the explanation'];

  function finishLive(run) {
    const p = pending.current;
    if (p.run !== run || !p.stepsDone || (!p.res && !p.err)) return;
    if (p.err) { setSide({ phase: 'error', run }); return; }
    setResult(p.res);
    setSide({ phase: 'result', run });
  }

  function verify() {
    if (!canVerify) return;
    const run = side.run + 1;
    setResult(null);
    setSide({ phase: 'loading', run, labels });
    scrollSideIntoView(sideRef);
    if (!LIVE) return;
    pending.current = { run };
    API.verifyNews({ text: checkedText, script: usedScript, domain, source: isImg ? 'image' : 'text' })
      .then((r) => { if (pending.current.run === run) { pending.current.res = r; finishLive(run); } },
        (e) => { if (pending.current.run === run) { pending.current.err = e; finishLive(run); } });
  }

  function stepsDone(run) {
    if (!LIVE) { setSide({ phase: 'result', run }); return; }
    if (pending.current.run === run) { pending.current.stepsDone = true; finishLive(run); }
  }

  /* ---------- result ---------- */
  function renderResult() {
    const r = result;
    const base = NEWS_OUT[r ? r.label : outcome] || NEWS_OUT.fake;
    const o = r ? { ...base, w: Math.round((r.confidence || 0) * 100), why: r.explanation && r.explanation.length ? r.explanation : base.why } : base;
    const d = DOMAINS.find((x) => x[0] === (r && r.domain ? r.domain : domain)) || DOMAINS[7];
    const src = isImg ? 'Image · OCR' : script === 'roman' ? 'Roman Urdu text' : 'Nastaliq text';
    const srcLine = isImg ? `Source: Image (${image ? image.name : 'upload'}), text read with OCR` : `Source: ${src}`;
    const report = () => `Sachai Check · Fake news\n${srcLine}\nVerdict: ${o.en} (${o.ur})\nDomain: ${d[1]}\nConfidence: ${o.w}%${LIVE ? '' : ' (demo)'}\n\nText checked:\n${checkedText}\n\n${o.why.map((t, i) => `${i + 1}. ${t}`).join('\n')}`;
    const k = `${side.run}-${outcome}`;
    return (
      <>
        <div className={`verdict ${o.cls} fade`} key={'v' + k}>
          <div className="meta">
            <span>Verdict · {src}{LIVE ? '' : ' · demo data'}</span>
            <span className="pill">{d[1]} · <span className="ur" lang="ur" style={{ lineHeight: 1.6 }}>{d[2]}</span></span>
          </div>
          <div><h2>{o.en}</h2><p className="ur v" lang="ur">{o.ur}</p></div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            <div className="meta">
              <span>Model confidence <span className="ur" lang="ur" style={{ lineHeight: 1.6 }}>اعتماد</span></span>
              <span style={{ fontFamily: 'var(--mono)' }}>{o.w}%{LIVE ? '' : ' demo'}</span>
            </div>
            <div className="bar"><i style={{ width: `${o.w}%` }} /></div>
          </div>
        </div>
        <div className="why fade" key={'w' + k}>
          <h3>Why this verdict <span className="ur" lang="ur">وضاحت</span></h3>
          <ol>{o.why.map((t, i) => <li key={i} style={{ animationDelay: `${0.15 + i * 0.1}s` }}>{t}</li>)}</ol>
          <p className="note">
            Written by the LLM judge. The judge explains the classifier's verdict and cannot change it.
            {isImg ? ' The text was read from the image with OCR, so a misread word can shift the verdict. Check the extracted text before relying on it.' : ''}
          </p>
        </div>
        <div className="btnrow">
          <CopyButton getText={report} />
          <button type="button" className="btn btn-ink" onClick={verify}>Run again</button>
        </div>
        {!LIVE && (
          <DemoSwitch a="fake" b="real" labelA="Fake" labelB="Real" value={outcome}
            onChange={(v) => { setOutcome(v); store.set('sc1-news', v); }} />
        )}
      </>
    );
  }

  return (
    <>
      <PageHead kicker="Mode 1 · Text and image pipeline" title="Fake News Detection" ur="جعلی خبر کی جانچ" />
      <div className="wrap grid2">
        <div className="panel light r" style={{ '--d': '.2s' }}>
          <div className="row">
            <span className="lbl">Input</span>
            <Segmented options={INPUT_OPTS} value={input} onChange={setInput} />
          </div>

          {input === 'text' && (
            <div className="stack fade" key="textBlock">
              <div className="row">
                <span className="lbl">Script</span>
                <Segmented options={SCRIPT_OPTS} value={script} onChange={setScript} />
              </div>
              <label className="row lbl" htmlFor="newsText">
                <span>News text</span>
                <span className="ur" lang="ur" style={{ fontWeight: 400, fontSize: 15, lineHeight: 1.8, color: 'var(--pigeon)' }}>خبر یہاں لکھیں یا چسپاں کریں</span>
              </label>
              <textarea id="newsText" key={script} style={REFADE} {...scriptProps(script)} value={text}
                onChange={(e) => setTexts((t) => ({ ...t, [script]: e.target.value }))} />
              {script === 'roman' && (
                <div className="translit fade">
                  <span>Transliterated to Nastaliq before classification</span>
                  <p lang="ur">{text.trim() === ROMAN_SAMPLE ? ROMAN_TRANSLIT : TRANSLIT_PENDING}</p>
                </div>
              )}
            </div>
          )}

          {input === 'image' && (
            <div className="stack fade" key="imgBlock">
              <label className={'idrop' + (over ? ' over' : '')} htmlFor="imgFile" hidden={!!image} {...dragProps}>
                <span className="up"><ImageIcon size={28} stroke="#C9F03A" /></span>
                <strong>Drop a news screenshot or browse</strong>
                <span className="ur" lang="ur">خبر کی تصویر یہاں چھوڑیں یا چسپاں کریں</span>
                <small>PNG · JPG · WEBP · up to {CONFIG.maxImageMB} MB<br />Headlines, forwarded posts, newspaper clippings · Ctrl+V pastes too</small>
                {imgErr && <span className="err">{imgErr}</span>}
              </label>
              <input id="imgFile" ref={fileRef} className="sr" type="file" accept="image/png,image/jpeg,image/webp,image/*"
                onChange={(e) => { const f = e.target.files[0]; if (f) loadImage(f); e.target.value = ''; }} />

              {!image && (
                <div className="isample">
                  <span className="hint">No screenshot handy?</span>
                  <button type="button" className="linkbtn"
                    onClick={async () => { const b = await makeSampleImage(); if (b) loadImage(b, 'sample_forwarded_post.png', true); }}>
                    Try a sample image
                  </button>
                </div>
              )}

              {image && (
                <div className="ipreview fade" key={image.url}>
                  <div className={'thumb' + (ocr && ocr.busy ? ' scanning' : '')}><img src={image.url} alt="Uploaded news image" /></div>
                  <div className="imeta">
                    <b>{image.name}</b>
                    <span>{image.w}×{image.h} · {fmtBytes(image.size)}{image.sample ? ' · sample' : ''}</span>
                    <div className="ibtns">
                      <button type="button" className="ghost-l" onClick={() => fileRef.current && fileRef.current.click()}>Replace</button>
                      <button type="button" className="ghost-l" onClick={removeImage}>Remove</button>
                    </div>
                  </div>
                </div>
              )}

              {image && ocr && (
                <div className="ocr fade">
                  <div className="row">
                    <label className="lbl" htmlFor="ocrText">Text read from image <small>· edit if OCR misread</small></label>
                    <span className="ostat">
                      {ocr.busy ? <><span className="dot pulse" />Reading text…</>
                        : ocr.failed ? 'OCR failed'
                          : <><span className="dot" />{ocr.script === 'roman' ? 'Roman Urdu' : 'Nastaliq'} detected{LIVE ? '' : ' · demo'}</>}
                    </span>
                  </div>
                  {ocr.busy ? (
                    <div className="oshim" aria-hidden="true">
                      <span className="shim" style={{ width: '94%' }} /><span className="shim" style={{ width: '82%' }} /><span className="shim" style={{ width: '66%' }} />
                    </div>
                  ) : (
                    <textarea id="ocrText" style={REFADE} {...scriptProps(ocr.script)} value={ocr.text}
                      onChange={(e) => setOcr((o) => ({ ...o, text: e.target.value }))} />
                  )}
                  {!ocr.busy && ocr.note && <p className="onote">{ocr.note}</p>}
                </div>
              )}
            </div>
          )}

          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            <span className="lbl">Domain <small>· auto-detected, tap to override</small></span>
            <div className="domains">
              {DOMAINS.map(([id, en, ur]) => (
                <button key={id} type="button" className="chip" aria-pressed={id === domain} onClick={() => setDomain(id)}>
                  {en} <span className="ur" lang="ur">{ur}</span>
                </button>
              ))}
            </div>
          </div>

          <div className="actions">
            <span className="hint">{count}</span>
            <button type="button" className="btn btn-lime" disabled={!canVerify} style={{ opacity: canVerify ? 1 : 0.5 }} onClick={verify}>
              Verify <span className="ur" lang="ur">تصدیق کریں</span><Arrow />
            </button>
          </div>
        </div>

        <div className="side r" style={{ '--d': '.3s' }} ref={sideRef} aria-live="polite">
          {side.phase === 'empty' && (
            <EmptySide>
              {isImg
                ? 'Upload a screenshot of a headline, post or clipping. OCR pulls the Urdu text out, you can correct it, then press Verify.'
                : 'Press Verify to run the check.'}{' '}
              You'll see the verdict, a confidence score and the judge's explanation.
            </EmptySide>
          )}
          {side.phase === 'loading' && (
            <StepRunner key={side.run} urLabel="جانچ جاری ہے" labels={side.labels} onDone={() => stepsDone(side.run)} />
          )}
          {side.phase === 'error' && <ErrorSide />}
          {side.phase === 'result' && renderResult()}
        </div>
      </div>
      <PageFoot />
    </>
  );
}
