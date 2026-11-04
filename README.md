# Sylva Cortex

**Ingénieux 2026 Hackathon · Track 2 — Forests & Wildlife Intelligent Sensing**
IEEE CIS × IEEE SPCOM · Mahindra University · 4 November 2026

An adaptive bioacoustic threat-detection agent for forest rangers. The system continuously listens to forest audio, separates biologically meaningful sound from environmental noise, classifies events (animal calls, chainsaws, gunshots, vehicles, weather), and an agent decides how to respond — adjusting its sensitivity dynamically based on operator feedback.

---

## Two Pillars

1. **Named, Measurable DSP:** Deterministic classical signal processing (Butterworth bandpass, spectral subtraction with noise floor estimation, Hann-tapered framing, mel filterbanks, energy MAD segmentation, and HPSS).
2. **Persistent Agentic AI:** A stateful observation → belief update → contextualization → decision → action → adaptation loop maintaining a memory window and adaptive per-class precision controllers.

---

## Status & Build Roadmap

| Phase | Deliverable | Status |
|-------|-------------|--------|
| P1 | Repo scaffold, config, documentation | Completed |
| P2 | DSP chain + unit tests | In progress |
| P3 | Dataset loader + synthetic SNR mixer | Pending |
| P4 | Baseline classifier + E1 harness | Pending |
| P5 | Agent loop + trace logging | Pending |
| P6 | Ablations E2–E7 + figure generation | Pending |
| P7 | React console + Web Audio port | Pending |
| P8 | Slides, report, demo video | Pending |

---

## Quickstart

### Prerequisites
- Python 3.11+
- Virtual environment with dependencies from `requirements.txt`:
  ```bash
  python -m venv .venv
  source .venv/bin/activate  # Or on Windows: .venv\Scripts\activate
  pip install -r requirements.txt
  ```

### Running Unit Tests
```bash
python tests/test_dsp.py
```

---

## Repository Structure

```
sylva-cortex/
├── README.md                 # Project overview, quickstart, roadmap
├── SPEC.md                   # Full system specification
├── LIMITATIONS.md            # Honest failure modes and edge cases
├── THIRD_PARTY.md            # External datasets and library licences
├── LICENSE                   # MIT Licence
├── .gitignore                # Source control ignore rules
├── requirements.txt          # Python dependencies
├── config.yaml               # Central DSP and agent hyperparameters
├── src/
│   ├── dsp/                  # DSP: preprocess, denoise, features, segment, hpss
│   ├── models/               # Classifier models & ONNX export
│   ├── agent/                # Agent state, decision policy, adaptation
│   ├── eval/                 # Evaluation harness & ablation metrics
│   └── utils/
├── scripts/                  # Data fetch and pipeline automation scripts
├── data/                     # Manifest and audio samples
├── results/                  # metrics.json and decision traces
├── figures/                  # Generated plots and spectrograms
├── web/                      # React/Vite audio console
├── slides/                   # Presentation deck
├── report/                   # Technical report
└── evidence/                 # Demo recordings and logs
```

---

## Results Summary

> [!NOTE]
> Per the project guardrails, every metric is generated programmatically into `results/metrics.json`. No metric is hand-written or fabricated.

| Experiment | Description | Status |
|---|---|---|
| E1 | Precision / Recall / F1 vs SNR | NOT YET MEASURED |
| E2 | Raw audio vs Denoised (α, β sweep) | NOT YET MEASURED |
| E3 | Fixed threshold vs Adaptive agent | NOT YET MEASURED |
| E4 | Energy-threshold detector baseline | NOT YET MEASURED |
| E5 | Classifier-alone baseline | NOT YET MEASURED |
| E6 | Confusion matrix at 0 dB SNR | NOT YET MEASURED |
| E7 | End-to-end latency (ms per event) | NOT YET MEASURED |
