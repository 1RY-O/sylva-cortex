# Sylva Cortex — Known Limitations & Honest Failure Modes

As per SPEC.md guardrails and rubric requirements, this document records known failure modes, design trade-offs, and assumptions.

## 1. Acoustic & Environmental Constraints
- **Extreme Weather Interference:** Heavy torrential rain and high winds elevate the spectral floor across the 80 Hz – 8 kHz band, increasing false-positive rates or masking weak animal vocalisations.
- **Microphone Distance & Attenuation:** High-frequency acoustic attenuation in dense forest foliage limits the effective detection range of high-frequency bird calls compared to low-frequency chainsaw rumble.
- **Reverberation & Canopy Multipath:** Multi-path reflections in dense jungle can smear transient signals (e.g., gunshots), altering attack time measurements.

## 2. Signal Processing Assumptions
- **Stationary Noise Floor:** The 10th-percentile energy estimation assumes that in any analysis block, at least 10% of frames represent background environmental noise without threat signatures.
- **Bandpass Cutoffs:** Filtering between 80 Hz and 8000 Hz intentionally discards sub-audible elephant rumbles (<80 Hz) and ultra-high bat echolocation calls (>8000 Hz) to optimize computation and eliminate wind noise.
- **Linear Phase & Delay:** Using zero-phase bidirectional filtering (`sosfiltfilt`) in offline processing is non-causal. The live Web Audio port will use causal IIR filtering which introduces phase distortion near cutoff boundaries.

## 3. Classification & Agent Policy Assumptions
- **Class Imbalance:** Threat events (gunshots, chainsaws) are rare events relative to wind, rain, and insect chorus.
- **Context Multiplier Sensitivity:** Context modifiers (e.g., night core zone multiplier = 1.30) assume accurate situational awareness (time and GPS zone). Misconfigured geolocation could artificially inflate or depress alarm thresholds.
- **Operator Feedback Latency:** The adaptive proportional controller for decision thresholds relies on timely operator confirmation or dismissal. In intermittent-connectivity scenarios, threshold updates will lag.

## 4. Dataset & Network-Isolation Constraints
- **Offline Synthetic Placeholder Set:** Due to hackathon network-isolation constraints, real raw gigabyte corpora (ESC-50, UrbanSound8K, FSD50K) are registered with reproducible download scripts (`scripts/download_data.py`), but offline evaluation currently uses a synthetic physical model placeholder set (286 clips, seed 1337) to benchmark DSP and SNR degradation without network calls.
