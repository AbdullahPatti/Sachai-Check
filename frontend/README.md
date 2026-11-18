# Sachai Check · Frontend (React + Vite)

React 18 app built with Vite. Same design and behaviour as the original plain HTML version.

## Run

Needs Node.js 18 or newer.

```
npm install
npm run dev
```

Opens at http://localhost:5173

Production build:

```
npm run build     # output in dist/
npm run preview   # serve the built app locally
```

## Connect the FastAPI backend

Copy `.env.example` to `.env` and set:

```
VITE_API_BASE=http://localhost:8000
```

Restart `npm run dev` after editing `.env`. With `VITE_API_BASE` empty the app runs in demo mode: OCR text and verdicts are simulated.

FastAPI needs CORS enabled for `http://localhost:5173`.

## Structure

| Path | What it is |
|---|---|
| `src/main.jsx` | Entry point |
| `src/App.jsx` | Hash router (`#/`, `#/news`, `#/audio`) and view transitions |
| `src/components/Nav.jsx` | Top bar with sliding tab indicator |
| `src/components/Segmented.jsx` | Pill switch used for input, script and source |
| `src/components/Shared.jsx` | Step runner, empty state, copy button, demo switch, page header/footer |
| `src/views/Home.jsx` | Landing page |
| `src/views/News.jsx` | Fake news: text input, image upload, OCR, verdict |
| `src/views/Audio.jsx` | Audio deepfake: upload, record, waveform, verdict |
| `src/lib/config.js` | Reads `VITE_API_BASE` and `VITE_MAX_IMAGE_MB` |
| `src/lib/api.js` | `ocr()` and `verifyNews()` fetch calls |
| `src/lib/data.js` | Sample texts, domains, demo verdicts |
| `src/lib/utils.js` | Helpers and the canvas sample image |
| `src/styles.css` | All styling, light and dark themes |

## Backend contract (when `VITE_API_BASE` is set)

### `POST /api/news/ocr`
`multipart/form-data` with field `image`.

```json
{ "text": "حکومت نے اعلان کیا ہے ...", "script": "nastaliq", "confidence": 0.91 }
```
`script` is `nastaliq` or `roman`.

### `POST /api/news/verify`
```json
{ "text": "...", "script": "nastaliq", "domain": "politics", "source": "image" }
```
`source` is `text` or `image`. Response:
```json
{ "label": "fake", "confidence": 0.82, "domain": "politics", "explanation": ["...", "...", "..."] }
```

The audio pipeline is still demo only.
