"""Sylva Cortex — Digital Signal Processing (DSP) Module.

Contains deterministic, classical audio signal processing implementations:
- preprocess: DC removal, pre-emphasis, 4th-order Butterworth bandpass, framing.
- denoise: Quantile noise floor estimation, Berouti spectral subtraction, Wiener gain smoothing.
- features: Mel filterbanks, Log-Mel spectrogram, MFCC + Δ + ΔΔ, spectral centroid,
            flatness, rolloff, ZCR, crest factor, attack time.
- segment: Adaptive energy MAD event segmentation and gap merging.
- hpss: Harmonic-percussive source separation via 2D median filtering.
"""

from src.dsp.preprocess import (
    remove_dc,
    pre_emphasis,
    design_butterworth_bandpass,
    bandpass_filter,
    frame_signal,
    preprocess,
)
from src.dsp.denoise import (
    estimate_noise_floor,
    spectral_subtraction,
    smooth_spectral_gain,
    denoise,
)
from src.dsp.features import (
    hz_to_mel,
    mel_to_hz,
    mel_filterbank,
    compute_log_mel_spectrogram,
    compute_deltas,
    compute_mfcc,
    compute_spectral_centroid,
    compute_spectral_flatness,
    compute_spectral_rolloff,
    compute_zero_crossing_rate,
    compute_crest_factor,
    compute_attack_time,
    extract_features,
)
from src.dsp.segment import (
    compute_frame_energies,
    compute_adaptive_threshold,
    merge_and_filter_segments,
    segment_events,
)
from src.dsp.hpss import (
    hpss_spectrogram,
    hpss,
)

__all__ = [
    "remove_dc",
    "pre_emphasis",
    "design_butterworth_bandpass",
    "bandpass_filter",
    "frame_signal",
    "preprocess",
    "estimate_noise_floor",
    "spectral_subtraction",
    "smooth_spectral_gain",
    "denoise",
    "hz_to_mel",
    "mel_to_hz",
    "mel_filterbank",
    "compute_log_mel_spectrogram",
    "compute_deltas",
    "compute_mfcc",
    "compute_spectral_centroid",
    "compute_spectral_flatness",
    "compute_spectral_rolloff",
    "compute_zero_crossing_rate",
    "compute_crest_factor",
    "compute_attack_time",
    "extract_features",
    "compute_frame_energies",
    "compute_adaptive_threshold",
    "merge_and_filter_segments",
    "segment_events",
    "hpss_spectrogram",
    "hpss",
]

