import { Arrow } from '../components/Icons.jsx';
import { TICKER } from '../lib/data.js';

export default function Home() {
  return (
    <>
      <div className="wrap hero">
        <div style={{ display: 'flex', flexDirection: 'column', gap: 26, flex: '1 1 720px' }}>
          <span className="kicker r"><span className="dot pulse" />Urdu verification for text, images and speech</span>
          <h1 className="display r" style={{ '--d': '.08s' }}>Check the news.<br />Check the <mark>voice.</mark></h1>
          <p className="lead r" style={{ '--d': '.16s' }}>
            Two independent pipelines. One reads Urdu news in Nastaliq or Roman script, typed or pulled from a screenshot with OCR. The other listens for synthetic Urdu speech. Every verdict comes with a plain explanation.
          </p>
        </div>
        <p className="ur big r" lang="ur" style={{ '--d': '.24s' }}>خبر بھی پرکھیں،<br />آواز بھی پرکھیں</p>
      </div>

      <div className="ticker r" style={{ '--d': '.3s' }} aria-hidden="true">
        <div className="track">
          {TICKER.concat(TICKER).map(([en, ur], i) => (
            <span key={i}>{en}<span className="ur" lang="ur">{ur}</span><i /></span>
          ))}
        </div>
      </div>

      <div className="wrap modes">
        <a className="card text r" href="#/news" style={{ '--d': '.34s' }}>
          <div className="top">
            <span className="icon" style={{ background: '#1F2430' }}>
              <svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="#C9F03A" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><rect x="4" y="3" width="16" height="18" rx="2" /><path d="M8 8h8M8 12h8M8 16h5" /></svg>
            </span>
            <span className="tag">Mode 1 · Text and image</span>
          </div>
          <div><h2>Fake News Detection</h2><p className="ur sub" lang="ur" style={{ color: 'var(--pigeon)' }}>جعلی خبر کی جانچ</p></div>
          <p>Paste a headline, a forwarded message or a full article, or upload a screenshot and let OCR read it. Roman Urdu is transliterated to Nastaliq before the classifier reads it.</p>
          <div className="chips"><span>Nastaliq</span><span>Roman Urdu</span><span>Image OCR</span><span>8 news domains</span><span>Judge explanation</span></div>
          <span className="btn btn-ink go">Check a news item <Arrow stroke="#C9F03A" /></span>
        </a>

        <a className="card voice r" href="#/audio" style={{ '--d': '.42s' }}>
          <div className="top">
            <span className="icon mini" style={{ background: '#C9F03A', justifyContent: 'center' }}>
              <b style={{ height: 14 }} />
              <b style={{ height: 26, animationDelay: '.12s' }} />
              <b style={{ height: 34, animationDelay: '.24s' }} />
              <b style={{ height: 22, animationDelay: '.36s' }} />
              <b style={{ height: 12, animationDelay: '.48s' }} />
            </span>
            <span className="tag">Mode 2 · Speech</span>
          </div>
          <div><h2 style={{ color: '#F7F0E4' }}>Audio Deepfake Detection</h2><p className="ur sub" lang="ur" style={{ color: '#C9F03A' }}>آڈیو ڈیپ فیک کی جانچ</p></div>
          <p>Upload a voice note or record one here. The model flags cloned or text-to-speech Urdu voices and marks the segments that gave it away.</p>
          <div className="chips"><span>Upload or record</span><span>Urdu speech</span><span>Segment timeline</span><span>Judge explanation</span></div>
          <span className="btn btn-lime go">Check an audio clip <Arrow /></span>
        </a>
      </div>

      <div className="wrap how" id="how">
        <div className="head r" style={{ '--d': '.5s' }}><h3>How a check works</h3><p className="ur" lang="ur" style={{ margin: 0 }}>جانچ کیسے ہوتی ہے</p></div>
        <div className="steps3">
          <div className="r" style={{ '--d': '.55s' }}><span className="n">01</span><h4>Submit</h4><p className="ur" lang="ur" style={{ color: 'var(--pigeon)' }}>جمع کریں</p><p>Text in either script, a news screenshot, or an audio clip. Pick the mode, nothing is fused.</p></div>
          <div className="r" style={{ '--d': '.62s' }}><span className="n">02</span><h4>Classify</h4><p className="ur" lang="ur" style={{ color: 'var(--pigeon)' }}>درجہ بندی</p><p>A dedicated model per pipeline returns the verdict and a confidence score. Images go through OCR first.</p></div>
          <div className="r" style={{ '--d': '.69s' }}><span className="n">03</span><h4>Explain</h4><p className="ur" lang="ur" style={{ color: 'var(--pigeon)' }}>وضاحت</p><p>An LLM judge writes the reasoning. It explains the verdict, it never overrides it.</p></div>
        </div>
      </div>

      <footer className="foot">
        <div className="wrap">
          <div><b>Sachai Check</b>Research prototype · National University of Computer and Emerging Sciences, Lahore</div>
          <div style={{ textAlign: 'right' }}>Muhammad Abdullah Haroon · Ikhlas Ahmad · Zayan Amjad<br /><span className="adv">Advisor: Ms. Sukhan Amir</span></div>
        </div>
      </footer>
    </>
  );
}
