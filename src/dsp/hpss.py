"""Harmonic-Percussive Source Separation (HPSS) using 2D median filtering.

Algorithm:
Separates acoustic signals into harmonic (horizontal, sustained tonal structures)
and percussive (vertical, broadband transient structures) components on the STFT
magnitude spectrogram using directional median filtering.
Reference: Fitzgerald, D. (2010). Harmonic/percussive separation using median
filtering. In Proceedings of the 13th International Conference on Digital Audio
Effects (DAFx-10), Graz, Austria.
"""

from typing import Tuple
import numpy as np
import scipy.signal as signal
from scipy.ndimage import median_filter


def hpss_spectrogram(
    magnitude_spectrogram: np.ndarray,
    kernel_harmonic: int = 15,
    kernel_percussive: int = 15,
    power: float = 2.0,
    eps: float = 1e-10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Separates a magnitude spectrogram into harmonic and percussive representations.

    Algorithm:
    1. Filter magnitude spectrogram along time frames (axis=1) with length
       kernel_harmonic to enhance horizontal harmonic trajectories: H_med = median_filter(S, (1, k_h)).
    2. Filter magnitude spectrogram along frequency bins (axis=0) with length
       kernel_percussive to enhance vertical transient clicks: P_med = median_filter(S, (k_p, 1)).
    3. Soft Wiener-style masking:
       M_H = (H_med ^ power) / (H_med ^ power + P_med ^ power + eps)
       M_P = (P_med ^ power) / (H_med ^ power + P_med ^ power + eps)
    Reference: Fitzgerald, D. (2010). DAFx-10.

    Args:
        magnitude_spectrogram: 2D array of shape (n_bins, n_frames).
        kernel_harmonic: Odd integer median filter length along time axis (default 15).
        kernel_percussive: Odd integer median filter length along freq axis (default 15).
        power: Exponent for soft masking (default 2.0).
        eps: Small positive constant for numerical stability.

    Returns:
        Tuple of (harmonic_spec, percussive_spec, mask_harmonic, mask_percussive):
            - harmonic_spec: 2D array (n_bins, n_frames)
            - percussive_spec: 2D array (n_bins, n_frames)
            - mask_harmonic: 2D array (n_bins, n_frames) with values in [0, 1]
            - mask_percussive: 2D array (n_bins, n_frames) with values in [0, 1]
    """
    s = np.asarray(magnitude_spectrogram, dtype=np.float64)
    if s.ndim != 2:
        raise ValueError(f"Expected 2D magnitude spectrogram, got shape {s.shape}")
    if s.size == 0:
        zeros = np.zeros_like(s)
        return zeros, zeros, zeros, zeros

    # Median filter along time frames (harmonic)
    h_med = median_filter(s, size=(1, kernel_harmonic), mode="nearest")
    # Median filter along frequency bins (percussive)
    p_med = median_filter(s, size=(kernel_percussive, 1), mode="nearest")

    h_pow = h_med ** power
    p_pow = p_med ** power
    denom = h_pow + p_pow + eps

    mask_h = h_pow / denom
    mask_p = p_pow / denom

    harm_spec = mask_h * s
    perc_spec = mask_p * s
    return harm_spec, perc_spec, mask_h, mask_p


def hpss(
    signal_arr: np.ndarray,
    sample_rate: int = 16000,
    n_fft: int = 400,
    hop_length: int = 160,
    kernel_harmonic: int = 15,
    kernel_percussive: int = 15,
    power: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Separates a 1D audio waveform into harmonic and percussive time-domain signals.

    Algorithm:
    1. Short-Time Fourier Transform (STFT) with Hann windowing.
    2. Spectrogram decomposition using 2D median filtering (Fitzgerald 2010).
    3. Soft mask multiplication against complex STFT bins.
    4. Overlap-add synthesis via inverse STFT (iSTFT).
    Reference: Fitzgerald, D. (2010).

    Args:
        signal_arr: 1D audio waveform.
        sample_rate: Sampling frequency in Hz.
        n_fft: FFT window length in samples (400 for 25 ms @ 16 kHz).
        hop_length: Frame hop in samples (160 for 10 ms @ 16 kHz).
        kernel_harmonic: Filter length along time axis for sustained tones.
        kernel_percussive: Filter length along frequency axis for transients.
        power: Masking exponent (default 2.0).

    Returns:
        Tuple of (harmonic_signal, percussive_signal) 1D arrays matching input length.
    """
    x = np.asarray(signal_arr, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(f"Expected 1D audio array, got shape {x.shape}")
    orig_len = len(x)
    if orig_len == 0:
        return x.copy(), x.copy()

    if orig_len < n_fft:
        pad_len = n_fft - orig_len
        x_pad = np.pad(x, (0, pad_len), mode="constant")
    else:
        x_pad = x

    noverlap = n_fft - hop_length
    _, _, zxx = signal.stft(
        x_pad,
        fs=sample_rate,
        window="hann",
        nperseg=n_fft,
        noverlap=noverlap,
        padded=True,
    )

    mag = np.abs(zxx)
    _, _, mask_h, mask_p = hpss_spectrogram(
        mag,
        kernel_harmonic=kernel_harmonic,
        kernel_percussive=kernel_percussive,
        power=power,
    )

    zxx_h = mask_h * zxx
    zxx_p = mask_p * zxx

    _, x_h = signal.istft(
        zxx_h,
        fs=sample_rate,
        window="hann",
        nperseg=n_fft,
        noverlap=noverlap,
    )
    _, x_p = signal.istft(
        zxx_p,
        fs=sample_rate,
        window="hann",
        nperseg=n_fft,
        noverlap=noverlap,
    )

    # Trim or pad to original length
    def fit_length(arr: np.ndarray) -> np.ndarray:
        if len(arr) >= orig_len:
            return arr[:orig_len]
        return np.pad(arr, (0, orig_len - len(arr)), mode="constant")

    return fit_length(x_h), fit_length(x_p)

