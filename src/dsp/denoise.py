"""Denoising module using multi-band spectral subtraction and temporal gain smoothing.

Algorithms implemented:
1. 10th-percentile stationary noise floor estimation across time frames.
   Reference: Martin, R. (2001). Noise power spectral density estimation based on
   optimal smoothing and minimum statistics. IEEE Transactions on Speech and
   Audio Processing, 9(5), 504-512.
2. Generalized spectral subtraction with over-subtraction and spectral floor.
   Reference: Berouti, M., Schwartz, R., & Makhoul, J. (1979). Enhancement of
   speech corrupted by acoustic noise. IEEE ICASSP, 4, 208-211.
   Reference: Boll, S. (1979). Suppression of acoustic noise in speech using
   spectral subtraction. IEEE Trans. ASSP, 27(2), 113-120.
3. Wiener-style temporal gain smoothing across adjacent frames.
   Reference: Ephraim, Y., & Malah, D. (1984). Speech enhancement using a
   minimum-mean square error short-time spectral amplitude estimator.
   IEEE Trans. ASSP, 32(6), 1109-1121.
"""

from typing import Tuple
import numpy as np
import scipy.signal as signal
from scipy.ndimage import uniform_filter1d


def estimate_noise_floor(
    power_spectrogram: np.ndarray,
    percentile: float = 10.0,
) -> np.ndarray:
    """Estimates stationary noise floor power per frequency bin.

    Algorithm: Quantile-based noise tracking. Computes the p-th percentile energy
    across all analysis frames for each frequency bin, under the assumption that
    signal events are sparse in time and the quietest frames represent ambient
    background noise.
    Reference: Martin, R. (2001). IEEE Trans. Speech Audio Process., 9(5), 504-512.

    Args:
        power_spectrogram: 2D array of shape (n_freq_bins, n_time_frames) containing
            power spectral values (|STFT|^2).
        percentile: Percentile value in [0, 100] (default 10.0).

    Returns:
        1D array of shape (n_freq_bins,) representing estimated noise power.
    """
    p = np.asarray(power_spectrogram, dtype=np.float64)
    if p.ndim != 2:
        raise ValueError(f"Expected 2D power spectrogram, got shape {p.shape}")
    if p.shape[1] == 0:
        return np.zeros(p.shape[0], dtype=np.float64)

    noise_floor = np.percentile(p, percentile, axis=1)
    return np.maximum(noise_floor, 1e-12)


def spectral_subtraction(
    power_spectrogram: np.ndarray,
    noise_power: np.ndarray,
    alpha: float = 2.0,
    beta: float = 0.02,
) -> Tuple[np.ndarray, np.ndarray]:
    """Applies Berouti generalized spectral subtraction with spectral flooring.

    Algorithm: Subtracted power spectrum:
        P_clean(f, t) = max(P_noisy(f, t) - alpha * N(f), beta * N(f))
    where alpha >= 1 is an over-subtraction factor designed to push residual noise
    peaks below the human hearing / classifier threshold, and beta >= 0 sets a
    spectral floor to prevent musical noise flutter.
    Reference: Berouti, M., Schwartz, R., & Makhoul, J. (1979). IEEE ICASSP, 4, 208-211.

    Args:
        power_spectrogram: 2D array of shape (n_freq_bins, n_time_frames).
        noise_power: 1D array of shape (n_freq_bins,) with noise floor estimate.
        alpha: Over-subtraction factor (typically 1.0 - 4.0, default 2.0).
        beta: Spectral floor factor (typically 0.01 - 0.05, default 0.02).

    Returns:
        Tuple of (clean_power_spectrogram, gain_filter):
            - clean_power_spectrogram: 2D array of shape (n_freq_bins, n_time_frames).
            - gain_filter: 2D array of shape (n_freq_bins, n_time_frames) with values in [0, 1].
    """
    p = np.asarray(power_spectrogram, dtype=np.float64)
    n = np.asarray(noise_power, dtype=np.float64)[:, np.newaxis]

    subtracted = p - alpha * n
    floor_level = beta * n
    clean_power = np.maximum(subtracted, floor_level)

    # Gain G(f, t) = sqrt(clean_power / noisy_power)
    denom = np.maximum(p, 1e-12)
    gain = np.sqrt(clean_power / denom)
    gain = np.clip(gain, 0.0, 1.0)
    return clean_power, gain


def smooth_spectral_gain(
    gain_matrix: np.ndarray,
    smoothing_frames: int = 3,
) -> np.ndarray:
    """Applies Wiener-style temporal smoothing across adjacent time frames.

    Algorithm: Moving-average filtering across the time axis (frames) on the
    gain suppression function G(f, t). Eliminates isolated musical noise tones
    and rapid frame-to-frame gain chatter.
    Reference: Ephraim, Y., & Malah, D. (1984). IEEE Trans. ASSP, 32(6), 1109-1121.

    Args:
        gain_matrix: 2D array of shape (n_freq_bins, n_time_frames).
        smoothing_frames: Odd integer filter length (default 3).

    Returns:
        Smoothed 2D gain matrix of same shape.
    """
    g = np.asarray(gain_matrix, dtype=np.float64)
    if smoothing_frames <= 1 or g.shape[1] <= 1:
        return g.copy()

    smoothed = uniform_filter1d(g, size=smoothing_frames, axis=1, mode="nearest")
    return np.clip(smoothed, 0.0, 1.0)


def denoise(
    signal_arr: np.ndarray,
    sample_rate: int = 16000,
    n_fft: int = 400,
    hop_length: int = 160,
    noise_percentile: float = 10.0,
    oversubtraction_alpha: float = 2.0,
    spectral_floor_beta: float = 0.02,
    gain_smoothing_frames: int = 3,
) -> np.ndarray:
    """Executes full spectral subtraction denoising pipeline on a 1D audio array.

    Pipeline:
    1. STFT computation via Hann-windowed framing.
    2. 10th-percentile stationary noise floor estimation per bin.
    3. Over-subtraction and spectral flooring.
    4. Temporal gain smoothing across adjacent frames.
    5. Phase-preserving inverse STFT synthesis.

    Args:
        signal_arr: 1D input audio waveform.
        sample_rate: Sampling frequency in Hz.
        n_fft: FFT window size in samples (default 400 for 25 ms @ 16 kHz).
        hop_length: Frame hop in samples (default 160 for 10 ms @ 16 kHz).
        noise_percentile: Quantile for noise floor tracking (default 10.0).
        oversubtraction_alpha: Alpha parameter (default 2.0).
        spectral_floor_beta: Beta parameter (default 0.02).
        gain_smoothing_frames: Temporal smoothing length (default 3).

    Returns:
        Denoised 1D audio waveform of identical length.
    """
    x = np.asarray(signal_arr, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(f"Expected 1D audio array, got shape {x.shape}")
    orig_len = len(x)
    if orig_len == 0:
        return x.copy()

    # If signal is too short for even one STFT frame, return as is
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

    power = np.abs(zxx) ** 2
    noise_power = estimate_noise_floor(power, percentile=noise_percentile)
    _, raw_gain = spectral_subtraction(
        power,
        noise_power,
        alpha=oversubtraction_alpha,
        beta=spectral_floor_beta,
    )
    smooth_gain = smooth_spectral_gain(raw_gain, smoothing_frames=gain_smoothing_frames)

    # Reconstruct complex spectrum preserving original phase
    zxx_clean = smooth_gain * zxx

    _, reconstructed = signal.istft(
        zxx_clean,
        fs=sample_rate,
        window="hann",
        nperseg=n_fft,
        noverlap=noverlap,
    )

    # Match original length
    if len(reconstructed) >= orig_len:
        return reconstructed[:orig_len]
    return np.pad(reconstructed, (0, orig_len - len(reconstructed)), mode="constant")
