# Sylva Cortex — Build Specification

**Event:** Ingénieux 2026 Hackathon · Track 2 (Forests & Wildlife Intelligent Sensing)
**Host:** IEEE CIS × IEEE SPCOM · Mahindra University · 4 November 2026
**Duration:** 12 hours
**Repo:** `sylva-cortex`

---

## 0. Mission

Build a prototype **adaptive bioacoustic threat-detection agent** for forest rangers.

The system continuously listens to forest audio, separates biologically meaningful
signals from environmental noise, classifies events (animal calls, chainsaws,
gunshots, vehicles, weather), and an **agent** decides what to do about each event —
while adjusting its own sensitivity based on operator feedback.

Two equal-weight pillars:

1. **Signal processing** — real, named, measurable DSP. Not a wrapper around an API.
2. **Agentic AI** — a persistent observation → reasoning → action → adaptation loop
   with memory and state. Not a single LLM call.

---

## 1. Non-Negotiable Requirements

These map directly to the judging rubric. Do not skip any.

| # | Requirement | Rubric target |
|---|-------------|---------------|
| R1 | Named DSP algorithms with tunable parameters exposed in config | Signal processing (12) |
| R2 | Agent loop with explicit state, memory, and self-adjusting thresholds | Agentic AI (13) |
| R3 | Quantitative evaluation with baselines and ablations | Results (20) |
| R4 | Full decision trace logged and replayable in the UI | Integration depth (25) |
| R5 | Documented limitations and next research questions | Novelty + Potential (30) |

---

## 2. Architecture

Two halves, deliberately decoupled:

```
┌──────────────────────── PYTHON (offline) ─────────────────────────┐
│  dataset loader → synthetic mixer → DSP → features → classifier   │
│  → agent policy → evaluation harness → results/*.json + figures/  │
└───────────────────────────────────────────────────────────────────┘
                                │ exports
                    model.onnx + config.json + results
                                ▼
┌──────────────────────── REACT (live demo) ────────────────────────┐
│  Web Audio AudioWorklet → same DSP (JS port) → ONNX Runtime Web   │
│  → agent policy (JS port) → console UI                            │
└───────────────────────────────────────────────────────────────────┘
```

**Why this shape:** the Python side produces the numbers you get graded on. The
React side produces the live demo that must not crash on stage. The browser path
uses the microphone/file input with **zero network dependency** — no server to
fall over during judging.

Keep the browser DSP deliberately simpler than the Python DSP. Porting risk is real;
budget for it.

---

## 3. Signal Processing Module (`src/dsp/`)

Implement each of these as a separate, unit-testable function.

### 3.1 Preprocessing chain
1. **DC removal** — subtract running mean.
2. **Pre-emphasis** — first-order high-pass, coefficient 0.97.
3. **Bandpass filter** — Butterworth, 4th order, 80 Hz – 8000 Hz. Rationale:
   below 80 Hz is mostly wind and handling noise; above 8 kHz is largely insect
   hiss and sensor self-noise for the target classes.
4. **Framing** — 25 ms window, 10 ms hop, Hann taper.

### 3.2 Denoising (must be a real algorithm, not a low-pass)
- **Noise floor estimation:** take the 10th-percentile energy across all frames per
  frequency bin. This is the standard assumption that the quietest frames are
  noise-only.
- **Spectral subtraction with over-subtraction factor α ≈ 2.0** and a **spectral
  floor β ≈ 0.02** to prevent musical noise artefacts.
- **Wiener-style gain smoothing** across adjacent frames to reduce residual
  artefacts.
- Expose α and β in `config.yaml` — you will sweep them in the ablation.

### 3.3 Feature extraction
Per detected event segment, compute:

- **Log-Mel spectrogram** — 40 mel bands, log-scaled. This is your primary input.
- **MFCC** — 13 coefficients + Δ + ΔΔ.
- **Spectral centroid** — brightness; distinguishes tonal calls from broadband noise.
- **Spectral flatness** — near 1.0 for noise-like (rain, wind), near 0 for tonal
  (bird, whistle).
- **Spectral rolloff** (85%).
- **Zero-crossing rate.**
- **Crest factor** — peak/RMS. High for impulsive events (gunshot, branch snap).
- **Attack time** — onset sharpness in ms. Gunshots are near-instant; chainsaws ramp.

### 3.4 Event segmentation
Energy-based onset detection with adaptive threshold = median frame energy + k·MAD.
Merge onsets closer than 150 ms. Discard segments under 60 ms. Emit
`(start_time, end_time, features)` tuples.

### 3.5 Harmonic–percussive separation (optional, high payoff)
Apply median-filter HPSS on the spectrogram to split tonal vs percussive content.
Animal vocalisations concentrate in the harmonic component; chainsaws and gunshots
in the percussive component. Report this as an ablation if implemented.

---

## 4. Classification (`src/models/`)

**Stage 1 (do this first, ship it):** gradient-boosted trees (LightGBM or sklearn
`HistGradientBoostingClassifier`) on the engineered feature vector. Trains in
seconds, interpretable, gives you a working number within the first three hours.

**Stage 2 (only if time remains):** small CNN on log-Mel patches (~4 conv blocks,
global average pooling). Compare against Stage 1 — this comparison *is* an ablation
row and costs you nothing extra to report.

Target classes: `bird_call`, `mammal_call`, `insect`, `rain`, `wind`,
`chainsaw`, `gunshot`, `vehicle`, `human_speech`, `unknown`.

Export the chosen model to **ONNX** for the browser via `onnxruntime-web`.

---

## 5. Agent Module (`src/agent/`) — the differentiator

This must be a genuine stateful loop. Structure it as follows and make every
component inspectable.

### 5.1 State
```python
@dataclass
class AgentState:
    belief: dict[str, float]        # class -> posterior, decays over time
    thresholds: dict[str, float]    # per-class decision thresholds (adaptive)
    event_memory: deque             # rolling window of recent events
    precision_track: dict[str, float]  # running precision per class
    escalation_level: int           # 0 calm .. 3 emergency
    context: dict                   # time_of_day, weather, location, battery
```

### 5.2 The loop
Each tick:

1. **OBSERVE** — ingest new event segments + features from the DSP module.
2. **UPDATE BELIEF** — Bayesian update of class posteriors with temporal decay
   (older evidence weighs less). Detections arriving close together reinforce.
3. **CONTEXTUALISE** — modulate priors using context. A chainsaw at 14:00 near a
   marked logging concession is lower risk than the same sound at 02:00 in a
   protected core zone.
4. **DECIDE** — select action from `{IGNORE, LOG, ALERT, ESCALATE}` by comparing
   posterior against the *current adaptive threshold* for that class.
5. **ACT** — emit the action, write a structured trace entry with the full
   reasoning chain.
6. **ADAPT** — when the operator confirms or dismisses an alert, update
   `precision_track[class]`. If precision drops below target, raise that class's
   threshold; if precision is high and recall is low, lower it. Implement as a
   simple proportional controller. **Document the update rule explicitly.**

### 5.3 Trace format
Every decision writes one JSON line:
```json
{"t": 41.2, "event_id": 17, "observed": {"class": "chainsaw", "p": 0.81},
 "belief": {"chainsaw": 0.88}, "context_note": "02:14, core zone",
 "threshold_at_decision": 0.72, "action": "ESCALATE",
 "reason": "posterior exceeded adaptive threshold; night + core zone multiplier 1.3"}
```
This trace is what you replay in the UI. It is your proof of agentic behaviour.

### 5.4 Optional LLM narration layer
Use an LLM **only** to convert the structured trace into a human-readable situation
report. It must never sit in the control path — the loop must run identically with
the LLM disabled. State this explicitly in the report; it reads as engineering
discipline, not a limitation.

---

## 6. Data (`data/`)

Candidate public sources — **verify each licence yourself before committing anything**
and record the licence in `THIRD_PARTY.md`:

- **ESC-50** — 2,000 labelled environmental clips, 50 classes (includes chainsaw,
  rain, insects, animal calls).
- **UrbanSound8K** — includes `gun_shot`, `jackhammer`, `drilling`, `siren`.
- **Xeno-canto** — large bird vocalisation archive.
- **FSD50K** — broad AudioSet-derived corpus.

Do **not** commit raw audio. Commit only:
- `data/manifest.csv` — file → label → source → licence
- `scripts/download_data.py` — reproducible fetch
- `data/samples/` — 5–10 short clips for the demo

### Synthetic SNR sweep (your headline experiment)
Build `scripts/make_mixed_set.py`: overlay each threat clip onto forest ambience at
SNR ∈ {−10, −5, 0, 5, 10, 15, 20} dB, 20 mixtures per (class, SNR) cell. Seed the
RNG and record it. This gives you a controlled degradation curve no other team will
have.

---

## 7. Evaluation Harness (`src/eval/`)

Output every number to `results/metrics.json`. **Never hand-write a metric value
into a document — always reference the generated file.**

### Required experiments
| ID | Experiment | Produces |
|----|------------|----------|
| E1 | Precision / Recall / F1 vs SNR, per class | Headline curve |
| E2 | Raw audio vs denoised (α,β sweep) | DSP contribution |
| E3 | Fixed threshold vs adaptive agent policy | Agent contribution |
| E4 | Energy-threshold detector baseline | Floor baseline |
| E5 | Classifier-alone baseline (no agent) | Second baseline |
| E6 | Confusion matrix at 0 dB | Failure-mode analysis |
| E7 | End-to-end latency (ms per event) | Deployability |

E2 and E3 are the ones that prove your two pillars actually earn their points.
Without them, judges cannot tell whether your DSP or your agent did anything.

Generate all figures programmatically into `figures/` as PDF/PNG. Caption each one
in the report with the exact command that regenerated it.

---

## 8. Frontend (`web/`) — build this LAST

Stack: **Vite + React + TypeScript + Tailwind**.

### Screens
1. **Live Console** — the main demo surface.
   - Centre: **live spectrogram** rendered on `<canvas>` from the AudioWorklet FFT.
     Custom-written. This is the visual centrepiece and it is real data.
   - Left rail: pipeline stage indicators (Capture → Filter → Denoise → Segment →
     Classify → Decide) lighting up as audio flows through.
   - Right rail: scrolling event log with class, posterior, action taken.
   - Bottom strip: KPI counters — events processed, alerts raised, running precision.
2. **Decision Trace Replay** — scrub a timeline, watch the agent's belief state and
   thresholds evolve, see exactly why each action fired.
3. **Results** — embedded figures from `figures/`, rendered from `results/metrics.json`.

### ReactBits usage — cap it at four components
Install as source via the CLI (`npx shadcn@latest add @react-bits/<Name>-TS-TW`).
Suggested picks:

- **AnimatedList** — the event log rows.
- **CountUp** (micro) — KPI numbers animating on update.
- One **background** component for the idle/landing state only.
- One **text animation** for the title.

Hard constraints:
- **Do not run a WebGL/shader background behind the live spectrogram canvas.**
  Frame rate will collapse during the demo. Use the animated background on the
  landing screen and switch to a static dark surface on the console.
- Respect `prefers-reduced-motion` (ReactBits 2.0 handles much of this, but verify).
- Target 60 fps with the spectrogram running. Profile before the demo, not during it.

---

## 9. Repository Layout

```
sylva-cortex/
├── README.md                 # what it is, how to run, results summary
├── SPEC.md                   # this file
├── LIMITATIONS.md            # honest failure modes
├── THIRD_PARTY.md            # every dataset/model/tool + licence
├── LICENSE                   # MIT
├── .gitignore                # Python template + ML artefacts
├── requirements.txt
├── config.yaml               # all DSP + agent hyperparameters
├── src/
│   ├── dsp/                  # filters, denoise, features, segmentation
│   ├── models/               # classifier training + ONNX export
│   ├── agent/                # state, policy, adaptation, trace
│   ├── eval/                 # harness + figure generation
│   └── utils/
├── scripts/                  # download_data.py, make_mixed_set.py, run_all.sh
├── data/                     # manifest + samples only (.gitkeep elsewhere)
├── results/                  # metrics.json, traces/
├── figures/                  # generated plots
├── web/                      # Vite React console
├── slides/                   # 5–7 slide deck (PDF)
├── report/                   # 1–2 page technical report (PDF)
└── evidence/                 # screenshots, logs, demo recording
```

---

## 10. Guardrails — read these before writing any code

1. **Never fabricate a metric.** Every number in the README, report, or slides must
   be traceable to `results/metrics.json`. If an experiment hasn't run, write
   `NOT YET MEASURED`, not a plausible-looking value.
2. **Label provenance.** Each reported figure gets a source annotation: measured /
   from literature (cited) / assumed. Assumptions go in `LIMITATIONS.md`.
3. **Commit every 20–30 minutes** with a descriptive message. This evidences
   authorship within the hackathon window.
4. **Time-box aggressively.** If a module isn't working by its deadline, ship the
   simpler version and note the upgrade path. A working baseline beats a broken
   ambitious system in every rubric row.
5. **Do not refactor working code** late in the session. Add features beside it.
6. **Run `scripts/run_all.sh` end to end** before touching the frontend. If the
   pipeline doesn't reproduce from a clean checkout, nothing else matters.
7. **Record the demo video once the console works**, not at the end. Screen
   recordings are the fallback if the live demo fails.

---

## 11. Build Order

| Phase | Deliverable | Gate |
|-------|-------------|------|
| P1 | Repo scaffold, config, README skeleton | — |
| P2 | DSP chain + unit tests on one audio file | Spectrogram renders correctly |
| P3 | Dataset loader + synthetic SNR mixer | `manifest.csv` populated |
| P4 | Baseline classifier + E1 harness | First real F1 number exists |
| P5 | Agent loop + trace logging | Trace replays correctly |
| P6 | Ablations E2–E7 + figures | `metrics.json` complete |
| P7 | React console + Web Audio port | Live mic demo works offline |
| P8 | Slides + report + demo video | Submission checklist green |

Phases P2–P6 are where the points live. P7–P8 are presentation.

---

## 12. Definition of Done

- [ ] `git clone` → `pip install -r requirements.txt` → `bash scripts/run_all.sh`
      reproduces every figure from scratch
- [ ] `results/metrics.json` contains all seven experiments
- [ ] Agent trace replayable in the UI showing belief + threshold evolution
- [ ] Live demo runs from microphone with no network connection
- [ ] README states results, limitations, and next research questions
- [ ] `THIRD_PARTY.md` lists every external asset with its licence
- [ ] Slide deck (5–7 slides) and technical report (1–2 pages) committed
- [ ] Demo recording in `evidence/`
