"""Feature extraction module for bioacoustic event analysis.

Algorithms implemented:
1. Log-Mel Filterbank Spectrogram (40 bands).
   Reference: Davis, S., & Mermelstein, P. (1980). Comparison of parametric
   representations for monosyllabic word recognition in continuously spoken
   sentences. IEEE Trans. ASSP, 28(4), 357-366.
2. Mel-Frequency Cepstral Coefficients (MFCC, 13 coefficients) + Delta (Δ) + Delta-Delta (ΔΔ).
   Reference: Rabiner, L. R., & Juang, B. H. (1993). Fundamentals of Speech
   Recognition. Prentice Hall.
3. Spectral Centroid (spectral brightness).
   Reference: Peeters, G. (2004). A large set of audio features for sound
   description. CUIDADO Project Technical Report.
4. Spectral Flatness (Wiener entropy).
   Reference: Johnston, J. D. (1988). Transform coding of audio signals using
   perceptual noise criteria. IEEE JSAC, 6(2), 314-323.
5. Spectral Rolloff (85% energy concentration).
   Reference: Scheirer, E., & Slaney, M. (1997). Construction and evaluation
   of a robust multifeature speech/music discriminator. IEEE ICASSP, 2, 1331-1334.
6. Zero-Crossing Rate (ZCR).
   Reference: Kedem, B. (1986). Spectral analysis and discrimination by
   zero-crossings. Proceedings of the IEEE, 74(11), 1477-1493.
7. Crest Factor (peak-to-RMS ratio).
   Reference: ISO 1996-1:2016 Acoustics — Description, measurement and
   assessment of environmental noise.
8. Attack Time (onset transient sharpness in milliseconds).
   Reference: Peeters, G. et al. (2011). The Caterpillar system: acoustic features
   for expressive sound analysis. J. New Music Research, 40(2), 127-140.
"""

from typing import Dict, Any, Tuple
import numpy as np
import scipy.signal as signal
import scipy.fft as fft


def hz_to_mel(hz: float | np.ndarray) -> float | np.ndarray:
    """Converts frequency in Hertz to the Mel scale.

    Algorithm: m = 2595 * log10(1 + f / 700).
    Reference: O'Shaughnessy, D. (1987). Speech Communication: Human and Machine.
    Addison-Wesley.
    """
    hz_arr = np.asarray(hz, dtype=np.float64)
    return 2595.0 * np.log10(1.0 + np.maximum(hz_arr, 0.0) / 700.0)


def mel_to_hz(mel: float | np.ndarray) -> float | np.ndarray:
    """Converts pitch in Mel scale back to frequency in Hertz.

    Algorithm: f = 700 * (10^(m / 2595) - 1).
    Reference: O'Shaughnessy, D. (1987).
    """
    mel_arr = np.asarray(mel, dtype=np.float64)
    return 700.0 * (10.0 ** (mel_arr / 2595.0) - 1.0)


def mel_filterbank(
    n_mels: int = 40,
    n_fft: int = 400,
    sample_rate: int = 16000,
    f_min: float = 0.0,
    f_max: float | None = None,
) -> np.ndarray:
    """Constructs a triangular Mel-frequency filterbank matrix.

    Algorithm: Linearly spaced points in Mel scale are mapped to FFT frequency
    bins. Triangular weighting filters are placed with peak response 1.0 at each
    center frequency, ramping linearly from adjacent filter centers.
    Reference: Davis, S., & Mermelstein, P. (1980). IEEE Trans. ASSP, 28(4), 357-366.

    Args:
        n_mels: Number of Mel bands (default 40).
        n_fft: FFT window size (default 400).
        sample_rate: Sampling frequency in Hz (default 16000).
        f_min: Lowest frequency in Hz (default 0.0).
        f_max: Highest frequency in Hz (defaults to sample_rate / 2).

    Returns:
        2D array of shape (n_mels, n_fft // 2 + 1).
    """
    if f_max is None:
        f_max = sample_rate / 2.0
    if f_min < 0 or f_max > sample_rate / 2.0 or f_min >= f_max:
        raise ValueError(f"Invalid frequency range [{f_min}, {f_max}] for sample rate {sample_rate}")

    mel_min = float(hz_to_mel(f_min))
    mel_max = float(hz_to_mel(f_max))
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    n_bins = n_fft // 2 + 1
    fft_freqs = np.linspace(0.0, sample_rate / 2.0, n_bins)

    weights = np.zeros((n_mels, n_bins), dtype=np.float64)
    for m in range(n_mels):
        f_left = hz_points[m]
        f_center = hz_points[m + 1]
        f_right = hz_points[m + 2]

        rise_mask = (fft_freqs >= f_left) & (fft_freqs <= f_center)
        if f_center > f_left:
            weights[m, rise_mask] = (fft_freqs[rise_mask] - f_left) / (f_center - f_left)

        fall_mask = (fft_freqs >= f_center) & (fft_freqs <= f_right)
        if f_right > f_center:
            weights[m, fall_mask] = (f_right - fft_freqs[fall_mask]) / (f_right - f_center)

    return weights


def compute_log_mel_spectrogram(
    signal_arr: np.ndarray,
    sample_rate: int = 16000,
    n_fft: int = 400,
    hop_length: int = 160,
    n_mels: int = 40,
    f_min: float = 0.0,
    f_max: float | None = None,
    eps: float = 1e-10,
) -> np.ndarray:
    """Computes log-scaled Mel spectrogram.

    Algorithm: STFT -> Power spectrum (|STFT|^2) -> Mel filterbank multiplication
    -> Logarithmic compression: 10 * log10(max(Mel_energy, eps)).
    Reference: Davis & Mermelstein (1980).

    Args:
        signal_arr: 1D audio waveform.
        sample_rate: Sampling frequency in Hz.
        n_fft: FFT window length.
        hop_length: Step size between frames.
        n_mels: Number of Mel frequency bands.
        f_min: Minimum frequency cutoff.
        f_max: Maximum frequency cutoff.
        eps: Small positive constant for numerical safety on silence.

    Returns:
        2D array of shape (n_frames, n_mels).
    """
    x = np.asarray(signal_arr, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(f"Expected 1D audio array, got shape {x.shape}")

    if len(x) < n_fft:
        pad_len = n_fft - len(x)
        x = np.pad(x, (0, pad_len), mode="constant")

    noverlap = n_fft - hop_length
    _, _, zxx = signal.stft(
        x,
        fs=sample_rate,
        window="hann",
        nperseg=n_fft,
        noverlap=noverlap,
        boundary=None,
        padded=True,
    )
    # Shape of zxx: (n_bins, n_frames)
    power_spec = np.abs(zxx) ** 2
    filterbank = mel_filterbank(
        n_mels=n_mels,
        n_fft=n_fft,
        sample_rate=sample_rate,
        f_min=f_min,
        f_max=f_max,
    )

    # mel_energies shape: (n_mels, n_frames)
    mel_energies = np.dot(filterbank, power_spec)
    log_mel = 10.0 * np.log10(np.maximum(mel_energies, eps))
    # Return as (n_frames, n_mels)
    return log_mel.T


def compute_deltas(features: np.ndarray, width: int = 5) -> np.ndarray:
    """Computes delta (velocity / first temporal derivative) of feature frames.

    Algorithm: Orthogonal polynomial linear regression over a moving window:
        delta_t = sum_{n=1}^N n * (c_{t+n} - c_{t-n}) / (2 * sum_{n=1}^N n^2)
    Reference: Young, S. et al. (2002). The HTK Book (for HTK Version 3.2).
    Cambridge University Engineering Department.

    Args:
        features: 2D array of shape (n_frames, n_features).
        width: Odd integer window width >= 3 (default 5).

    Returns:
        2D array of shape (n_frames, n_features).
    """
    feat = np.asarray(features, dtype=np.float64)
    if feat.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {feat.shape}")
    n_frames, n_dims = feat.shape
    if n_frames <= 1:
        return np.zeros_like(feat)

    half = width // 2
    n = np.arange(-half, half + 1, dtype=np.float64)
    denom = 2.0 * float(np.sum(np.arange(1, half + 1, dtype=np.float64) ** 2))
    padded = np.pad(feat, ((half, half), (0, 0)), mode="edge")

    deltas = np.zeros_like(feat)
    for t in range(n_frames):
        window = padded[t : t + 2 * half + 1]
        deltas[t] = np.sum(n[:, np.newaxis] * window, axis=0) / denom
    return deltas


def compute_mfcc(
    log_mel_spectrogram: np.ndarray,
    n_mfcc: int = 13,
    include_deltas: bool = True,
) -> Tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Computes Mel-Frequency Cepstral Coefficients (MFCC) and optional Δ/ΔΔ.

    Algorithm: Type-II discrete cosine transform (DCT-II) applied along the Mel
    dimension, retaining the first n_mfcc cepstral coefficients.
    Reference: Davis & Mermelstein (1980); Rabiner & Juang (1993).

    Args:
        log_mel_spectrogram: 2D array of shape (n_frames, n_mels).
        n_mfcc: Number of cepstral coefficients to keep (default 13).
        include_deltas: If True, computes delta and delta-delta features.

    Returns:
        Tuple of (mfcc, deltas, delta_deltas):
            - mfcc: 2D array (n_frames, n_mfcc)
            - deltas: 2D array (n_frames, n_mfcc) or None
            - delta_deltas: 2D array (n_frames, n_mfcc) or None
    """
    log_mel = np.asarray(log_mel_spectrogram, dtype=np.float64)
    if log_mel.ndim != 2:
        raise ValueError(f"Expected 2D log-Mel spectrogram, got shape {log_mel.shape}")

    # Orthogonal DCT-II along frequency axis
    cepstra = fft.dct(log_mel, type=2, axis=-1, norm="ortho")
    mfcc_coeffs = cepstra[:, :n_mfcc]

    if not include_deltas:
        return mfcc_coeffs, None, None

    deltas = compute_deltas(mfcc_coeffs, width=5)
    delta_deltas = compute_deltas(deltas, width=5)
    return mfcc_coeffs, deltas, delta_deltas


def compute_spectral_centroid(
    magnitude_spectrogram: np.ndarray,
    sample_rate: int = 16000,
    n_fft: int = 400,
    eps: float = 1e-10,
) -> np.ndarray:
    """Computes spectral centroid (center of mass / brightness) per frame.

    Algorithm: Centroid(t) = sum(f_k * |X(f_k, t)|) / (sum(|X(f_k, t)|) + eps).
    Measures perceptual brightness; tonal animal vocalisations show concentrated
    centroids, whereas broadband wind and rain exhibit wide distributions.
    Reference: Peeters, G. (2004). CUIDADO Project Technical Report.

    Args:
        magnitude_spectrogram: 2D array of shape (n_bins, n_frames).
        sample_rate: Sampling frequency in Hz.
        n_fft: FFT window size.
        eps: Small positive constant for numerical safety on zero/silent frames.

    Returns:
        1D array of shape (n_frames,) in Hz.
    """
    mag = np.asarray(magnitude_spectrogram, dtype=np.float64)
    n_bins, n_frames = mag.shape
    freqs = np.linspace(0.0, sample_rate / 2.0, n_bins)[:, np.newaxis]

    numerator = np.sum(freqs * mag, axis=0)
    denominator = np.sum(mag, axis=0) + eps
    return numerator / denominator


def compute_spectral_flatness(
    magnitude_spectrogram: np.ndarray,
    eps: float = 1e-10,
) -> np.ndarray:
    """Computes spectral flatness (Wiener entropy) per frame.

    Algorithm: Ratio of geometric mean of power spectrum to arithmetic mean:
        Flatness(t) = exp(mean(ln(P(t) + eps))) / (mean(P(t)) + eps)
    Values range from 0.0 (pure tone) to 1.0 (white noise).
    Reference: Johnston, J. D. (1988). IEEE JSAC, 6(2), 314-323.

    Args:
        magnitude_spectrogram: 2D array of shape (n_bins, n_frames).
        eps: Small positive constant for numerical stability.

    Returns:
        1D array of shape (n_frames,) with values in [0.0, 1.0].
    """
    mag = np.asarray(magnitude_spectrogram, dtype=np.float64)
    power = mag ** 2

    log_power = np.log(np.maximum(power, eps))
    geom_mean = np.exp(np.mean(log_power, axis=0))
    arith_mean = np.mean(power, axis=0) + eps

    flatness = geom_mean / arith_mean
    return np.clip(flatness, 0.0, 1.0)


def compute_spectral_rolloff(
    magnitude_spectrogram: np.ndarray,
    sample_rate: int = 16000,
    n_fft: int = 400,
    rolloff_percent: float = 0.85,
) -> np.ndarray:
    """Computes spectral rolloff frequency per frame.

    Algorithm: Frequency below which rolloff_percent (default 85%) of total
    spectral energy is contained. Distinguishes high-frequency insect stridulation
    from low-frequency mammalian or motor rumble.
    Reference: Scheirer, E., & Slaney, M. (1997). IEEE ICASSP, 2, 1331-1334.

    Args:
        magnitude_spectrogram: 2D array of shape (n_bins, n_frames).
        sample_rate: Sampling frequency in Hz.
        n_fft: FFT window size.
        rolloff_percent: Fraction of total energy in (0.0, 1.0) (default 0.85).

    Returns:
        1D array of shape (n_frames,) in Hz.
    """
    mag = np.asarray(magnitude_spectrogram, dtype=np.float64)
    power = mag ** 2
    n_bins, n_frames = power.shape
    freqs = np.linspace(0.0, sample_rate / 2.0, n_bins)

    cum_energy = np.cumsum(power, axis=0)
    total_energy = cum_energy[-1, :]

    rolloff = np.zeros(n_frames, dtype=np.float64)
    for t in range(n_frames):
        tot = total_energy[t]
        if tot <= 1e-12:
            rolloff[t] = 0.0
        else:
            threshold = rolloff_percent * tot
            idx = np.searchsorted(cum_energy[:, t], threshold)
            idx = min(idx, n_bins - 1)
            rolloff[t] = freqs[idx]

    return rolloff


def compute_zero_crossing_rate(signal_arr: np.ndarray) -> float:
    """Computes zero-crossing rate over the audio time waveform.

    Algorithm: Rate of sign changes per sample:
        ZCR = sum(|sgn(x[n]) - sgn(x[n-1])|) / (2 * (N - 1)).
    Reference: Kedem, B. (1986). Proceedings of the IEEE, 74(11), 1477-1493.

    Args:
        signal_arr: 1D audio waveform.

    Returns:
        Scalar ZCR value in [0.0, 1.0].
    """
    x = np.asarray(signal_arr, dtype=np.float64)
    if len(x) <= 1:
        return 0.0
    signs = np.sign(x)
    # Treat 0 as positive sign to avoid spurious boundary fluctuations
    signs[signs == 0] = 1
    diffs = np.abs(np.diff(signs))
    return float(np.sum(diffs) / (2.0 * (len(x) - 1)))


def compute_crest_factor(signal_arr: np.ndarray, eps: float = 1e-10) -> float:
    """Computes crest factor (peak-to-RMS ratio).

    Algorithm: Peak amplitude divided by root-mean-square amplitude:
        Crest = max(|x|) / (sqrt(mean(x^2)) + eps).
    Impulsive threat events (gunshots, tree branch snaps) exhibit high crest
    factors; continuous signals (chainsaws, wind) exhibit low crest factors.
    Reference: ISO 1996-1:2016 Acoustics.

    Args:
        signal_arr: 1D audio waveform.
        eps: Small constant for numerical stability on silence.

    Returns:
        Scalar crest factor value >= 0.0.
    """
    x = np.asarray(signal_arr, dtype=np.float64)
    if len(x) == 0:
        return 0.0
    peak = float(np.max(np.abs(x)))
    rms = float(np.sqrt(np.mean(x ** 2)))
    if rms <= eps:
        return 0.0
    return float(peak / (rms + eps))


def compute_attack_time(
    signal_arr: np.ndarray,
    sample_rate: int = 16000,
    low_thresh_ratio: float = 0.10,
    high_thresh_ratio: float = 0.90,
) -> float:
    """Computes onset attack time in milliseconds.

    Algorithm: Measures the duration (in milliseconds) required for the signal
    envelope to rise from low_thresh_ratio (10%) to high_thresh_ratio (90%) of
    its peak amplitude. Impulsive events (gunshots) transition in < 5 ms;
    gradual mechanical events (chainsaw throttle) transition over > 50 ms.
    Reference: Peeters, G. et al. (2011). J. New Music Research, 40(2), 127-140.

    Args:
        signal_arr: 1D audio waveform.
        sample_rate: Sampling frequency in Hz.
        low_thresh_ratio: Lower envelope fraction (default 0.10).
        high_thresh_ratio: Upper envelope fraction (default 0.90).

    Returns:
        Attack time in milliseconds >= 0.0.
    """
    x = np.asarray(signal_arr, dtype=np.float64)
    if len(x) <= 1:
        return 0.0

    # Analytic envelope via Hilbert transform
    analytic = signal.hilbert(x)
    envelope = np.abs(analytic)
    peak = np.max(envelope)
    if peak <= 1e-12:
        return 0.0

    peak_idx = int(np.argmax(envelope))
    if peak_idx == 0:
        return 0.0

    # Look backwards from the peak to find 10% and 90% crossing
    sub_env = envelope[: peak_idx + 1]
    low_val = low_thresh_ratio * peak
    high_val = high_thresh_ratio * peak

    low_indices = np.where(sub_env <= low_val)[0]
    high_indices = np.where(sub_env >= high_val)[0]

    t_low = low_indices[-1] if len(low_indices) > 0 else 0
    t_high = high_indices[0] if len(high_indices) > 0 else peak_idx

    sample_diff = max(0, t_high - t_low)
    return float((sample_diff / sample_rate) * 1000.0)


def extract_features(
    signal_arr: np.ndarray,
    sample_rate: int = 16000,
    n_fft: int = 400,
    hop_length: int = 160,
    n_mels: int = 40,
    n_mfcc: int = 13,
    include_deltas: bool = True,
    compute_flatness: bool = True,
    compute_crest: bool = True,
    compute_attack_time_flag: bool = True,
) -> Dict[str, Any]:
    """Extracts the complete feature representation for an audio segment.

    All returned values are guaranteed to be finite even on completely silent
    or constant input signals (no NaN, no inf, no divide-by-zero).

    Args:
        signal_arr: 1D audio waveform segment.
        sample_rate: Sampling frequency in Hz.
        n_fft: FFT window length in samples.
        hop_length: Hop length in samples.
        n_mels: Number of Mel bands.
        n_mfcc: Number of MFCC coefficients.
        include_deltas: Whether to include Δ and ΔΔ MFCCs.
        compute_flatness: Whether to compute spectral flatness.
        compute_crest: Whether to compute crest factor.
        compute_attack_time_flag: Whether to compute attack time.

    Returns:
        Dictionary containing:
            - 'log_mel': 2D array (n_frames, n_mels)
            - 'mfcc': 2D array (n_frames, n_mfcc)
            - 'mfcc_deltas': 2D array (n_frames, n_mfcc) or None
            - 'mfcc_delta_deltas': 2D array (n_frames, n_mfcc) or None
            - 'spectral_centroid': 1D array (n_frames,)
            - 'spectral_flatness': 1D array (n_frames,) or None
            - 'spectral_rolloff': 1D array (n_frames,)
            - 'zero_crossing_rate': float
            - 'crest_factor': float or None
            - 'attack_time_ms': float or None
            - 'feature_vector': 1D flattened summary vector for classification
    """
    x = np.asarray(signal_arr, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(f"Expected 1D audio array, got shape {x.shape}")

    # 1. Log-Mel spectrogram
    log_mel = compute_log_mel_spectrogram(
        x,
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )

    # 2. MFCC + deltas
    mfcc_coeffs, deltas, delta_deltas = compute_mfcc(
        log_mel,
        n_mfcc=n_mfcc,
        include_deltas=include_deltas,
    )

    # 3. Spectral features from STFT magnitude
    if len(x) < n_fft:
        pad_len = n_fft - len(x)
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
        boundary=None,
        padded=True,
    )
    mag_spec = np.abs(zxx)

    centroid = compute_spectral_centroid(
        mag_spec,
        sample_rate=sample_rate,
        n_fft=n_fft,
    )
    rolloff = compute_spectral_rolloff(
        mag_spec,
        sample_rate=sample_rate,
        n_fft=n_fft,
        rolloff_percent=0.85,
    )

    flatness = compute_spectral_flatness(mag_spec) if compute_flatness else None
    zcr = compute_zero_crossing_rate(x)
    crest = compute_crest_factor(x) if compute_crest else None
    att_time = compute_attack_time(x, sample_rate=sample_rate) if compute_attack_time_flag else None

    # Construct tabular 1D feature vector (mean + std pooling)
    vec_parts = [
        np.mean(mfcc_coeffs, axis=0),
        np.std(mfcc_coeffs, axis=0),
    ]
    if include_deltas and deltas is not None and delta_deltas is not None:
        vec_parts.extend([
            np.mean(deltas, axis=0),
            np.std(deltas, axis=0),
            np.mean(delta_deltas, axis=0),
            np.std(delta_deltas, axis=0),
        ])
    vec_parts.extend([
        np.array([np.mean(centroid), np.std(centroid)], dtype=np.float64),
        np.array([np.mean(rolloff), np.std(rolloff)], dtype=np.float64),
        np.array([zcr], dtype=np.float64),
    ])
    if flatness is not None:
        vec_parts.append(np.array([np.mean(flatness), np.std(flatness)], dtype=np.float64))
    if crest is not None:
        vec_parts.append(np.array([crest], dtype=np.float64))
    if att_time is not None:
        vec_parts.append(np.array([att_time], dtype=np.float64))

    feature_vector = np.concatenate(vec_parts)
    # Ensure numerical sanity
    feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)

    return {
        "log_mel": log_mel,
        "mfcc": mfcc_coeffs,
        "mfcc_deltas": deltas,
        "mfcc_delta_deltas": delta_deltas,
        "spectral_centroid": centroid,
        "spectral_flatness": flatness,
        "spectral_rolloff": rolloff,
        "zero_crossing_rate": zcr,
        "crest_factor": crest,
        "attack_time_ms": att_time,
        "feature_vector": feature_vector,
    }
