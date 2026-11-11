# Sachai Check

**A multi-script, multi-domain fake news and audio deepfake detection system for the Urdu language.**

Final Year Project F26-034 · FAST-NUCES Lahore, School of Computing

Sachai means truth. Sachai Check does two separate things: it checks an Urdu news claim against dated evidence, and it checks whether an Urdu voice recording is real or synthetic.

## Why this exists

Urdu has over 246 million speakers and ranks among the ten most spoken languages in the world, yet its tools for text and speech verification lag far behind English. Urdu misinformation reaches people in four forms:

| Form | What it looks like |
|---|---|
| Nastaliq news | Native-script articles and headlines |
| Roman Urdu | WhatsApp-style forwards in Latin script |
| Social posts | Copied, cropped, stripped of context |
| Voice notes | Cloned or text-to-speech Urdu voices |

No existing Urdu system checks claims reliably across sources, scripts and recording conditions while keeping evidence apart from model guesswork.

The published numbers hide the problem. A detector can report an F1 of 0.97 inside one Urdu corpus, then label 99.7% of items fake once it is moved to a different dataset. On the only Urdu audio deepfake benchmark, a standard anti-spoofing model sits at roughly 49.5% equal error rate, which is close to chance.

## Design principles

1. **Two verdicts, never one.** A real voice can carry a false claim. A fake voice can say something true. Text truth and voice authenticity stay separate.
2. **Evidence over confidence.** Weak evidence never becomes a fake label. When evidence is thin, the answer is Unverified.
3. **The LLM explains, it does not decide.** It only describes outputs that have already been validated.
4. **Honest when unsure.** Bad input, missing evidence and service outages each get a different answer.

## How it works

```
                 React web interface
          (input, conversion confirm, results)
                        |
                  FastAPI service
           (validate, orchestrate, bound load)
                 /                \
        Text module            Audio module
   classifier + evidence     acoustic detector
            |                       |
  RSS feeds + LLM adapter   Temp audio storage
                            (deleted after processing)
```

### Text pipeline

1. Input in Urdu script or Roman Urdu
2. An LLM converts Roman Urdu to Urdu script, and the user confirms the conversion
3. XLM-R classifier with an out-of-distribution signal
4. Evidence retrieval from RSS news feeds, with sources and dates
5. Evidence adequacy rules

Possible outcomes: **Supported, Refuted, Mixed, Unverified, Cannot assess**

### Audio pipeline

1. Upload a clip or record in the browser, then decode checks
2. Resample to 16 kHz and split into windows
3. Score each window with an anti-spoofing model (AASIST-L, RawNet2)
4. Aggregate the window scores
5. Apply the threshold and a quality gate

Possible outcomes: **Bona fide, Synthetic, Inconclusive**

The result shows the verdict, a spoof score and the flagged segments.

## Scope

| In scope | Out of scope |
|---|---|
| Single claims in Urdu script and Roman Urdu | Image and video deepfakes |
| 8 news domains, from politics to sports | One merged verdict for text and audio |
| Evidence retrieval with sources and dates | Speaker identity or intent judgments |
| Synthetic voice detection for short clips | Live social monitoring, automatic takedowns |
| Web app with a FastAPI backend and private processing | Accounts, monetization, production scale |

## Input limits

| Input | Limit |
|---|---|
| Text claim | 4 to 1,000 words, never silently cut |
| Audio clip | WAV or MP3, 3 to 60 seconds, up to 20 MiB |
| Stored results | None. Audio is deleted after processing |

## Data

| Corpus | Type | Published size |
|---|---|---|
| Hook and Bait | Text | 78,409 items |
| CSALT-LUMS Urdu deepfake corpus | Audio | 37,281 clips |
| Ax-to-Grind | Text | 10,083 items |
| UrduFake family | Text | 1,600 items |
| Manual claim set | Text | 45 hand-verified claims |

Text preprocessing:

- Manifest every source with its origin, hash and license
- Audit labels so an original claim and its debunk are not confused
- Group near-duplicate stories (cosine similarity of 0.90 or higher) before splitting
- Freeze 70 / 15 / 15 splits
- Fit features and calibration on train and dev only

Audio robustness conditions: speaker-disjoint splits, added noise at 20, 10 and 5 dB, MP3 compression at 64 and 32 kb/s, reverberation, speed and pitch changes, and a held-out generator.

## Research questions

| # | Question | Test | Measure |
|---|---|---|---|
| RQ1 | Do dataset controls improve transfer? | Leave-one-dataset-out folds with story grouping | Macro-F1, per class |
| RQ2 | Does Roman Urdu keep its meaning after conversion? | Natural Roman Urdu pairs and round trips | Conversion fidelity |
| RQ3 | Does evidence-led assessment help? | Classifier vs evidence vs combined on the same claims | Correctness, coverage |
| RQ4 | How robust is audio detection? | Speaker-disjoint splits and distortions | EER, ROC-AUC |
| RQ5 | Are outputs usable and traceable? | 5-user task study and an explanation audit | Completion, citations |

No held-out label tunes any model, threshold or prompt. Abstentions are reported, never dropped.

## Project status

| Component | Status |
|---|---|
| Literature review, SRS, architecture and low-level design | Done |
| Frontend (text and audio flows) | Built, not yet connected to the backend |
| Audio deepfake pipeline | Experiments in progress |
| Fake news pipeline | Planned for FYP-2 |
| End-to-end integration | Planned for FYP-2 |

This repository currently holds the audio deepfake module.

## Tech stack

- **Backend:** Python, FastAPI
- **Frontend:** React
- **Text:** XLM-R, RSS evidence retrieval, LLM adapter for script conversion and explanations
- **Audio:** AASIST-L, RawNet2, 16 kHz windowed scoring

## Known limitations

- The audio data covers only two speech generators, so unseen generators are the main risk
- Few human-reviewed labels are available for Urdu claims
- GPU time and API budget are limited
- Stale or failed RSS feeds return Unverified instead of a guess

## Team

| Name | Roll number |
|---|---|
| Muhammad Abdullah Haroon | 23L-0734 |
| Ikhlas Ahmad | 23L-0638 |
| Zayan Amjad | 23L-0721 |

**Supervisor:** Ms. Sukhan Amir

National University of Computer and Emerging Sciences (FAST-NUCES), Lahore

## Acknowledgements

The audio work builds on the Urdu deepfake audio corpus from CSALT, LUMS (Munir et al.), and on the AASIST and RawNet2 anti-spoofing models. The text work draws on the Bend the Truth, UrduFake, Ax-to-Grind and Hook and Bait datasets.
