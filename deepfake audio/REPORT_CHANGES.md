# Report changes for the audio module (Deliverable II, LaTeX-ready)

Why: the report planned an AASIST-L vs RawNet2 comparison and cites 37,281 utterances. The project now builds one
pipeline (fine-tuned XLS-R with a layer-weighted head) on the Hugging Face release, which holds 6,794 clips.
Every block below replaces the named passage. Numbers marked `\todo{}` come from `runs/xlsr300m-sls-v1/report/`
after the GPU run; do not fill them from anywhere else.

---

## Abstract (replace the sentence on acoustic models)
```latex
The text pipeline pairs classifiers trained on native-script Urdu with LLM-based conversion of Roman Urdu and with
retrieved evidence, while a fine-tuned multilingual speech transformer (XLS-R) judges whether speech is synthetic and
a grounded LLM layer explains its measured output without being able to change it.
```

## Executive Summary (replace the paragraph starting "On the audio side")
```latex
On the audio side, a single detector makes the decision: XLS-R 300M, pretrained on 128 languages including Urdu,
fine-tuned with a learned weighting over all of its hidden layers. Scores are calibrated on held-out development
speakers and mapped to three outcomes: bona fide, synthetic, or inconclusive. An LLM turns the detector's validated
measurements into a short explanation, a second LLM audits that explanation against the same measurements, and a
deterministic template is used whenever either step fails. Neither LLM can alter the acoustic verdict.
```

## Table 1.1, row "Audio and user interface"
```latex
Audio pipeline and audio UI & Implemented code: data cache, speaker-disjoint split, resumable training, calibration,
API, explanation layer and wired interface & Implementation complete; final metrics pending the full GPU run. \\
```

## Table 2.1, row O5 (and O6 wording)
```latex
O5 & Build the independent audio branch & Fine-tuned XLS-R detector evaluated on speaker-disjoint partitions,
per spoofing attack and per speaker, with grouped bootstrap confidence intervals. \\
O6 & Measure robustness and confidence & Audio results under noise, reverberation, telephone band, MP3 and Opus with a
frozen policy; calibration error before and after temperature scaling; text calibration and OOD rejection analysis. \\
```

## Section 2.5.1, third bullet
```latex
\item Detection of synthetic speech in Urdu uploads of 3 to 60 seconds (WAV, MP3, M4A, OGG/Opus, WebM), using the
CSALT Tacotron and VITS benchmark, with a segment timeline, an out-of-distribution flag and an LLM explanation that is
grounded in measured values.
```

## Section 3.2.4.1, replace the summary sentence and the last paragraph
```latex
Munir et al. present an Urdu deepfake corpus of bona fide speech and Tacotron and VITS clones and report 37,281
utterances in the paper. The release hosted on Hugging Face, which this project uses, contains 6,794 clips from
17 speakers: 3,398 bona fide (two parts per speaker), 1,698 VITS and 1,698 Tacotron. File names in the second bona
fide part match the synthetic file names of the same speaker, so each synthetic clip has a genuine recording of the
same sentence by the same voice. They evaluate AASIST-L and RawNet2 and also run a human study [6].
```
```latex
The paired structure makes leakage easy: a random clip-level split would place a genuine sentence in training and its
clone in testing. Sachai Check therefore splits by speaker, keeps all of a speaker's genuine and cloned clips in one
partition, and resamples sentence groups when estimating confidence intervals. Because the original paper already
reports AASIST-L and RawNet2, the project does not repeat that comparison; it builds one stronger detector and
evaluates it under the stricter protocol.
```

## Section 3.5.3, replace the paragraph
```latex
Audio tests should separate dependence on the speaker, the generator, and the channel. Repeating the AASIST-L and
RawNet2 comparison would add nothing new. The defensible contribution is a single, carefully controlled detector:
a multilingual self-supervised encoder fine-tuned with layer weighting [27], [28], evaluated on unseen speakers,
calibrated on separate speakers, stress-tested with real codecs, and paired with an explanation layer whose claims
are checked against measured values. A result from two generators is still a bounded result
[6], [7], [8], [9], [10], [29].
```

## Section 4.2, FR-07, FR-08, FR-09
```latex
FR-07 / Reader & Accept WAV, MP3, M4A, OGG/Opus or WebM audio up to 20 MiB with a decoded duration of 3 to 60 seconds
and user-attested Urdu speech. Reject empty, undecodable or out-of-range files with a specific message. &
Format, size and duration boundary tests. \\
FR-08 / Reader & Return bona fide, synthetic or inconclusive from the calibrated detector score and the dev-fitted
band; return inconclusive when the quality gate fails (less than 1.5 s of speech or a very low signal-to-noise
estimate). Report two-second segment scores and an out-of-distribution flag. & Returned model version, temperature,
thresholds and score agree with the decision. \\
FR-09 / Reader & Build explanations only from numbered measured facts. An LLM drafts bullets that cite fact ids; rule
checks reject numbers absent from the facts and contradictions of the verdict; a second LLM judges faithfulness; a
deterministic template replaces the output when fewer than two bullets survive. & Unsupported-number and
contradiction test cases; human rubric review of a sample. \\
```

## Table 4.8, rows "Training workstation" and "Demonstration server"
```latex
Training workstation & RTX 5060 laptop GPU (8 GB) with bf16 and gradient checkpointing, 16 GB RAM, about 15 GB free SSD
for data, cache and checkpoints; the university RTX A4000 (16 GB) allows larger batches. \\
Demonstration server & 4 CPU cores, 16 GB RAM and any CUDA GPU with at least 4 GB; CPU-only inference is supported
with higher measured latency. \\
```

## Section 4.8.5 to 4.8.7 (append one sentence to each)
```latex
% 4.8.5
The page now sends the selected file to \texttt{POST /api/audio} and accepts WAV, MP3, M4A, OGG/Opus and WebM.
% 4.8.6
Browser recordings are sent as WebM/Opus and decoded on the server.
% 4.8.7
Verdict, spoof score, thresholds, segment timeline, explanation, Urdu summary and caveats are rendered from the live
API response; no demo values remain on the audio page.
```

## Table 4.9, row "GPU/API budget"
```latex
GPU/API budget / medium & Incomplete runs after power cuts & Team: atomic checkpoints every 8 minutes and every
epoch with a fallback copy, deterministic batch order for exact resumption, cached scores for every evaluation step. \\
```

## Table 5.1, row RQ4
```latex
RQ4: How robust is audio detection? & One fine-tuned XLS-R detector on unseen speakers, per attack, and under noise,
reverberation, telephone band, MP3 and Opus with a frozen policy. & EER, ROC-AUC, macro-F1, FAR/FRR at the operating
point, coverage and selective risk per condition. \\
```

## Table 5.2, row "CSALT-LUMS Urdu audio"
```latex
CSALT-LUMS Urdu audio [6] & Hugging Face release: 6,794 clips, 17 speakers (3,398 bona fide, 1,698 VITS,
1,698 Tacotron); the paper reports 37,281 & Speaker-disjoint split 12/2/3 frozen in \texttt{splits.json}; sentence
pairs grouped for bootstrapping. \\
```

## Section 5.7, replace the whole section
```latex
\section{Audio Preparation, Training, and Robustness}

\textbf{Preparation.} Every clip is decoded, converted to mono, resampled to 16 kHz and trimmed of leading and trailing
silence (40 dB below peak) for both classes, because unequal silence padding is a known shortcut in anti-spoofing
data [29]. Clips shorter than 0.5 s after trimming are excluded. A metadata-only logistic regression (duration,
silence, loudness) is trained on the training speakers and scored on the development speakers as a leakage check;
its AUC is reported with the results (\todo{metadata AUC}).

\textbf{Split.} The 17 speakers are divided into 12 training, 2 development and 3 test speakers with a fixed seed.
All genuine and cloned clips of a speaker stay in one partition, so test voices and test clone models are never seen
during training or calibration.

\textbf{Model.} The detector is XLS-R 300M [27], fine-tuned end to end with its convolutional waveform encoder frozen.
Instead of the last layer alone, a softmax-weighted sum of all 25 hidden states is learned, following the observation
that spoofing cues concentrate in lower and middle layers [28]. Frame features are projected to 256 dimensions,
pooled with attentive statistics pooling [30] and classified by a two-layer MLP. Inputs are normalised per window to
zero mean and unit variance, which removes loudness as a cue.

\textbf{Training.} Each batch uses a random crop length of 2, 3 or 4 s. With probability 0.75 a crop receives one or two
label-blind channel transforms in the spirit of RawBoost [8]: coloured noise (5 to 30 dB SNR), synthetic
reverberation, band-limiting, 8-bit mu-law quantisation, random peaking filters and impulsive noise. Real codecs are
excluded from training so that MP3 and Opus remain unseen test conditions. Optimisation uses AdamW (learning rate
$10^{-5}$ for the encoder, $5\times10^{-4}$ for the head), warm-up and cosine decay, label smoothing 0.05, an
effective batch of 32 and bf16 precision. Early stopping monitors development EER with a patience of 5 epochs.
A fixed clean subset of training clips is scored every epoch next to the development set, so underfitting (high
training EER) and overfitting (a widening train and development gap) are visible in one plot.

\textbf{Inference and decision.} A clip is scored with 4 s windows and a 2 s hop; window log-odds are averaged.
Temperature scaling [17] is fitted on development speakers. The development set also fixes two cut-offs: the widest
low and high scores that keep each decided side at 99\% precision, with a minimum band of $\pm0.10$ around the EER
threshold. Scores between them are reported as inconclusive. A separate pass with 2 s windows and a 1 s hop produces
the segment timeline shown to users.

\textbf{Coverage signal.} Class-conditional Gaussians with a shrunk shared covariance are fitted on clean training
embeddings, and the Mahalanobis distance threshold is the 99th percentile on development speakers [18]. A larger
distance marks the recording as outside the tested conditions; it adds a caveat and never flips the verdict.

\textbf{Robustness.} The test speakers are re-scored under the conditions in Table~\ref{tab:audio-conditions} with the
policy frozen; nothing is re-tuned per condition.

\textbf{Reproducibility and fault tolerance.} Batches are a deterministic function of seed, epoch and batch index.
A full checkpoint (weights, optimiser, scheduler, scaler, sampler position, history) is written atomically every
8 minutes and at every epoch, with the previous checkpoint kept as a fallback, so a power cut resumes at the same
batch. Every evaluation step caches its scores.
```

## Table 5.5, replace the table body (keep `\label{tab:audio-conditions}`)
```latex
Clean reference & Unaltered held-out audio & Baseline on unseen speakers. \\
Additive noise & Pink noise at 20, 10 and 5 dB SNR, seeded per clip & Noise sensitivity, not every real environment. \\
Reverberation & Synthetic room impulse responses, RT60 0.3 s and 0.8 s & Room effects; a related transform appears in training. \\
Telephone band & Down-sampled to 8 kHz and back & Narrow-band calls. \\
MP3 & Re-encoded at 64 and 32 kb/s with LAME & Unseen codec in training. \\
Opus & Re-encoded at 16 kb/s & Messaging-app voice notes; unseen codec in training. \\
```
Remove the speed/pitch and generator-holdout rows: they need extra training runs and the project uses one pipeline.
Report per-attack results (bona fide vs Tacotron, bona fide vs VITS) instead.

## Section 5.8, replace the audio sentence
```latex
For audio we report EER, ROC-AUC, accuracy and macro-F1 at the EER threshold, false acceptance and false rejection,
coverage and selective risk under the three-way policy, and expected calibration error before and after temperature
scaling, with synthetic as the positive class. Confidence intervals come from 1,000 bootstrap resamples of sentence
groups, so a genuine sentence and its clones move together. Results are broken down by attack and by test speaker.
```

## Table 5.6, last row
```latex
Audio, API and UI not supplied as finished components & Audio pipeline, \texttt{POST /api/audio}, temporary-file
cleanup and the audio page are implemented; the text endpoint and its wiring remain. \\
```

## Section 6.3.2, replace the paragraph
```latex
Uploads are decoded by FFmpeg to 16 kHz mono, trimmed, and checked against the quality gate. The fine-tuned XLS-R
detector scores 4 s windows for the verdict and 2 s windows for the timeline; window log-odds are averaged and
temperature-scaled, and the dev-fitted band yields bona fide, synthetic or inconclusive. In parallel, seven
interpretable descriptors (pitch variation, pitch movement, voiced share, pause share, pause background level,
high-band energy ratio, spectral flatness) are measured and placed against their genuine and synthetic training
distributions. These numbered facts feed the explanation layer; they are context and never part of the decision.
```

## Table 6.1, row AudioDetectorAdapter, plus a new row
```latex
AudioDetectorAdapter & Decoded waveform $\rightarrow$ calibrated score, verdict, segment scores, OOD distance,
quality flags, descriptors, versions & No content-truth or speaker-identity claim; loaded once at start-up. \\
ExplanationComposer & Numbered facts $\rightarrow$ judged bullets, Urdu summary, caveats & Cites fact ids only; rule
checks and an independent LLM judge; template fallback; cannot modify the verdict. \\
```

## Table 6.2, rows for audio and readiness
```latex
POST /api/audio & Multipart audio with permission confirmation $\rightarrow$ verdict, score, thresholds, segments,
quality, OOD flag, explanation, caveats, versions & 413 too large; 415 unsupported or undecodable; 422 empty, too short
or too long; 429 busy; 503 model unavailable; 504 timeout. \\
GET /health/ready & None $\rightarrow$ model versions and device & 503 with the load error if the model is missing. \\
```

## Figure 6.3 caption
```latex
\caption{Audio processing: FFmpeg decoding, trimming and the quality gate; XLS-R window scoring, averaging,
temperature scaling and the three-way band; measured descriptors and the OOD flag feed a grounded explanation that
cannot change the verdict.}
```

## Results paragraph for the next deliverable (template)
```latex
On the three unseen test speakers the detector reaches an EER of \todo{x.xx}\% (95\% CI \todo{a}--\todo{b}) and a
ROC-AUC of \todo{0.xxx}; \todo{xx.x}\% of clips fall outside the inconclusive band, with a selective risk of
\todo{x.xx}\%. Per attack, the EER is \todo{} for Tacotron and \todo{} for VITS. Under Opus voice-note compression
the EER is \todo{}\%. The learned layer weights peak at layer \todo{}, which supports [28].
```

## New references
```latex
[27] A. Babu et al., ``XLS-R: Self-supervised Cross-lingual Speech Representation Learning at Scale,''
     in Proc. Interspeech, 2022. Available: https://arxiv.org/abs/2111.09296.
[28] Q. Zhang, S. Wen, and T. Hu, ``Audio Deepfake Detection with Self-Supervised XLS-R and SLS Classifier,''
     in Proc. ACM Multimedia, 2024. doi:10.1145/3664647.3681345.
[29] N. M. M\"uller, F. Dieckmann, P. Czempin, R. Canals, K. B\"ottinger, and J. Williams, ``Speech is Silver,
     Silence is Golden: What do ASVspoof-trained Models Really Learn?,'' in Proc. ASVspoof 2021 Workshop, 2021.
     Available: https://arxiv.org/abs/2106.12914.
[30] K. Okabe, T. Koshinaka, and K. Shinoda, ``Attentive Statistics Pooling for Deep Speaker Embedding,''
     in Proc. Interspeech, 2018. Available: https://arxiv.org/abs/1803.10963.
```
