# Third-Party Dependencies and Licences

This document records all third-party software, libraries, and dataset sources utilised in the Sylva Cortex project along with their licences, conforming to Section 6 and Section 10 of SPEC.md.

## 1. Python Libraries

| Package | Version Target | Licence | Purpose |
|---------|----------------|---------|---------|
| `numpy` | >=1.26 | BSD-3-Clause | Core multidimensional array manipulation and numerical math |
| `scipy` | >=1.11 | BSD-3-Clause | Butterworth filter design, DCT, signal processing algorithms |
| `librosa` | >=0.10 | ISC | Audio feature extraction reference, mel filterbank design |
| `soundfile` | >=0.12 | BSD-3-Clause | Audio I/O based on libsndfile (LGPLv2.1+) |
| `scikit-learn` | >=1.3 | BSD-3-Clause | Tabular classification models, evaluation metrics |
| `lightgbm` | >=4.0 | MIT | Gradient-boosted decision trees for threat classification |
| `matplotlib` | >=3.8 | PSF | Metric visualization and figure generation |
| `pyyaml` | >=6.0 | MIT | Parsing central hyperparameter configuration |
| `pandas` | >=2.0 | BSD-3-Clause | Dataset manifest manipulation and tabular metrics |
| `onnx` | >=1.15 | Apache-2.0 | Model interchange format for web export |
| `onnxruntime` | >=1.17 | MIT | Runtime engine for running ONNX models in browser |

## 2. Audio Datasets & Benchmarks (Candidate Sources)

| Dataset | Primary Content | Licence | Source / Citation |
|---------|-----------------|---------|-------------------|
| **ESC-50** | 2,000 environmental audio recordings (chainsaw, rain, animals) | CC-BY-NC 3.0 | Piczak (2015), ACM Multimedia |
| **UrbanSound8K** | 8,732 urban sound excerpts (gunshots, sirens, drilling) | CC-BY-NC 3.0 | Salamon, Jacoby, Bello (2014), ACM MM |
| **Xeno-canto** | Global avian and wildlife vocalisations | CC-BY-NC / CC-BY-SA | Xeno-canto Foundation |
| **FSD50K** | 51,197 Freesound audio clips under AudioSet ontology | CC-BY 4.0 | Fonseca et al. (2021), IEEE/ACM TASLP |

## 3. Web & Frontend Stack

| Library | Licence | Description |
|---------|---------|-------------|
| React 18 / 19 | MIT | UI component library |
| Vite | MIT | Frontend build tool |
| Tailwind CSS | MIT | Utility-first CSS framework |
| ONNX Runtime Web | MIT | In-browser model inference |
