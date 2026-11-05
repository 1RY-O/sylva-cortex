# Sylva Cortex: Adaptive Bioacoustic Threat Detection
**Ingénieux 2026 Hackathon — Track 2: Forests & Wildlife**

## 1. Problem Statement
Forest ecosystems generate vast acoustic data. Traditional monitoring relies on static classifiers that suffer from high false-positive rates due to environmental noise (wind, rain) and lack contextual awareness. We need a system that not only detects threats but reasons about them adaptively.

## 2. Methodology & Architecture
Sylva Cortex is a two-pillar system combining Signal Processing and Agentic AI.

**Pillar 1: Signal Processing & Classification**
- Audio is preprocessed using a 4th-order Butterworth bandpass filter (80-8000 Hz).
- We apply Berouti spectral subtraction (α=2.0, β=0.02) to suppress stationary environmental noise.
- Features (40-band Log-Mel, 13 MFCCs, spectral centroid, etc.) are extracted.
- A LightGBM classifier is trained using GroupKFold cross-validation.

**Pillar 2: Agentic AI (The Brain)**
- Instead of a static threshold, the agent maintains a Bayesian belief state with temporal decay (γ=0.95).
- **Contextual Adaptation:** The decision threshold dynamically adjusts based on spatiotemporal context (e.g., Night + Core Zone lowers the threshold to 0.538; Day + Logging Concession raises it to 0.875).
- **Action Execution:** The agent doesn't just log; it takes simulated actions (e.g., dispatching drones or alerting rangers based on escalation levels).

## 3. Experimental Results
We constructed a controlled synthetic acoustic dataset spanning SNR levels from -10dB to +20dB.

- **E1 (Baseline Performance):** The LightGBM classifier achieved an overall accuracy of 98.82%. Under severe acoustic masking (-10dB SNR), performance realistically degraded to 95.0% F1-score.
- **E2 (DSP Ablation):** We proved that our spectral subtraction denoiser specifically recovers signal integrity under heavy noise, improving F1-scores at 0dB SNR from 0.9917 (Raw) to 1.0000 (Denoised).

## 4. Limitations & Next Questions
- **Data:** Evaluation relies on a domain-randomized synthetic dataset due to network constraints. Future work requires field testing on the Xeno-canto and ESC-50 datasets.
- **Action Validation:** The action execution layer is currently simulated via JSON traces. Future iterations will integrate actual edge-device APIs for drone dispatch.
- **Hardware:** Edge inference latency on low-power IoT devices (e.g., Raspberry Pi) remains to be benchmarked.