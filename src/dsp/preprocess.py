"""Preprocessing module for bioacoustic audio signals.

Algorithms implemented:
1. DC offset removal via mean subtraction or running mean.
   Reference: Oppenheim, A. V., & Schafer, R. W. (2009). Discrete-Time Signal
   Processing (3rd ed.). Pearson.
2. First-order high-pass pre-emphasis filter.
   Reference: Rabiner, L. R., & Schafer, R. W. (1978). Digital Processing of
   Speech Signals. Prentice Hall.
3. 4th-order Butterworth bandpass filter (80 Hz - 8000 Hz).
   Reference: Butterworth, S. (1930). On the Theory of Filter Amplifiers.
   Experimental Wireless and the Wireless Engineer, 7, 536-541.
4. Overlapping window framing with Hann taper.
   Reference: Harris, F. J. (1978). On the use of windows for harmonic analysis
   with the discrete Fourier transform. Proceedings of the IEEE, 66(1), 51-83.
"""

from typing import Literal, Tuple
import numpy as np
import scipy.signal as signal


def remove_dc(
    signal_arr: np.ndarray,
    method: Literal["mean", "running_mean"] = "mean",
    running_window: int | None = None,
) -> np.ndarray:
    """Removes DC bias from a 1D audio signal.

    Algorithm: Subtracts either global arithmetic mean or a running moving average
    from the input time-series to eliminate sensor bias and sub-audible DC drift.
    Reference: Oppenheim, A. V., & Schafer, R. W. (2009). Discrete-Time Signal
    Processing (3rd ed.). Pearson.

    Args:
        signal_arr: 1D input audio waveform.
        method: 'mean' for global mean subtraction, 'running_mean' for local box filter.
        running_window: Window size in samples for running mean (required if method='running_mean').

    Returns:
        DC-centered 1D array of same shape and float dtype.
    """
    x = np.asarray(signal_arr, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(f"Expected 1D audio array, got shape {x.shape}")
    if len(x) == 0:
        return x.copy()

    if method == "mean":
        return x - np.mean(x)
    elif method == "running_mean":
        if running_window is None or running_window <= 1:
            raise ValueError("running_window must be an integer > 1 for running_mean")
        window = np.ones(running_window, dtype=np.float64) / running_window
        running_avg = np.convolve(x, window, mode="same")
        return x - running_avg
    else:
        raise ValueError(f"Unknown DC removal method: {method}")


def pre_emphasis(signal_arr: np.ndarray, coeff: float = 0.97) -> np.ndarray:
    """Applies a first-order FIR high-pass pre-emphasis filter.

    Algorithm: Difference equation y[n] = x[n] - coeff * x[n-1].
    Flattens the spectral tilt (6 dB/octave falloff typical of natural bioacoustic
    and vocal sources), improving numerical sensitivity for higher frequency bands.
    Reference: Rabiner, L. R., & Schafer, R. W. (1978). Digital Processing of
    Speech Signals. Prentice Hall.

    Args:
        signal_arr: 1D input audio array.
        coeff: Filter coefficient alpha in [0.0, 1.0). Default is 0.97.

    Returns:
        Pre-emphasised 1D audio array of same length.
    """
    x = np.asarray(signal_arr, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(f"Expected 1D audio array, got shape {x.shape}")
    if len(x) == 0:
        return x.copy()

    y = np.empty_like(x)
    y[0] = x[0] * (1.0 - coeff)
    y[1:] = x[1:] - coeff * x[:-1]
    return y


def design_butterworth_bandpass(
    sample_rate: int,
    low_hz: float = 80.0,
    high_hz: float = 8000.0,
    order: int = 4,
) -> np.ndarray:
    """Designs a Butterworth bandpass filter in second-order sections (SOS).

    Algorithm: Butterworth maximally flat magnitude filter design. When high_hz
    equals or exceeds the Nyquist frequency (sample_rate / 2), the upper cutoff is
    either safely clamped below Nyquist or converted to a high-pass filter,
    avoiding invalid digital filter specs.
    Reference: Butterworth, S. (1930). On the Theory of Filter Amplifiers.
    Experimental Wireless and the Wireless Engineer, 7, 536-541.

    Args:
        sample_rate: Sampling frequency in Hz.
        low_hz: Lower passband cutoff frequency in Hz.
        high_hz: Upper passband cutoff frequency in Hz.
        order: Butterworth filter order (per side).

    Returns:
        Second-order sections (SOS) array representation.
    """
    nyquist = sample_rate / 2.0
    if low_hz <= 0 or low_hz >= nyquist:
        raise ValueError(f"low_hz ({low_hz}) must be between 0 and Nyquist ({nyquist})")

    if high_hz >= nyquist:
        # If high cutoff reaches Nyquist, a highpass filter at low_hz passes all
        # frequencies up to Nyquist with a maximally flat passband.
        sos = signal.butter(
            N=order,
            Wn=low_hz,
            btype="highpass",
            fs=sample_rate,
            output="sos",
        )
    else:
        sos = signal.butter(
            N=order,
            Wn=[low_hz, high_hz],
            btype="bandpass",
            fs=sample_rate,
            output="sos",
        )
    return sos


def bandpass_filter(
    signal_arr: np.ndarray,
    sample_rate: int = 16000,
    low_hz: float = 80.0,
    high_hz: float = 8000.0,
    order: int = 4,
) -> np.ndarray:
    """Applies zero-phase 4th-order Butterworth bandpass filtering.

    Algorithm: Forward-backward zero-phase digital filtering (sosfiltfilt)
    using numerically stable second-order sections. Suppresses out-of-band wind,
    handling noise below 80 Hz, and high-frequency sensor hiss above 8 kHz.
    Reference: Butterworth, S. (1930); Oppenheim & Schafer (2009).

    Args:
        signal_arr: 1D audio waveform.
        sample_rate: Sampling frequency in Hz.
        low_hz: Lower corner frequency in Hz (default 80 Hz).
        high_hz: Upper corner frequency in Hz (default 8000 Hz).
        order: Filter order (default 4).

    Returns:
        Filtered 1D audio waveform of same length.
    """
    x = np.asarray(signal_arr, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(f"Expected 1D audio array, got shape {x.shape}")
    if len(x) == 0:
        return x.copy()

    sos = design_butterworth_bandpass(
        sample_rate=sample_rate,
        low_hz=low_hz,
        high_hz=high_hz,
        order=order,
    )
    # Check minimum signal length required for sosfiltfilt padlen
    n_sections = sos.shape[0]
    min_len = 3 * (2 * n_sections + 1)
    if len(x) < min_len:
        return signal.sosfilt(sos, x)
    return signal.sosfiltfilt(sos, x)


def frame_signal(
    signal_arr: np.ndarray,
    frame_length: int = 400,
    hop_length: int = 160,
    window: str = "hann",
) -> Tuple[np.ndarray, np.ndarray]:
    """Frames a 1D audio signal into overlapping windowed frames.

    Algorithm: Short-time sliding window analysis with Hann tapering to minimise
    spectral leakage and discontinuities at frame boundaries.
    Reference: Harris, F. J. (1978). On the use of windows for harmonic analysis
    with the discrete Fourier transform. Proceedings of the IEEE, 66(1), 51-83.

    Args:
        signal_arr: 1D audio waveform.
        frame_length: Length of each frame in samples (e.g., 400 for 25 ms @ 16 kHz).
        hop_length: Step size between consecutive frames (e.g., 160 for 10 ms @ 16 kHz).
        window: Window taper name ('hann', 'hamming', 'boxcar').

    Returns:
        Tuple of (frames, window_vector):
            - frames: 2D array of shape (num_frames, frame_length), window applied.
            - window_vector: 1D array of shape (frame_length,) with the window taper.
    """
    x = np.asarray(signal_arr, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(f"Expected 1D audio array, got shape {x.shape}")

    if window == "hann":
        win = signal.windows.hann(frame_length, sym=False)
    elif window == "hamming":
        win = signal.windows.hamming(frame_length, sym=False)
    elif window == "boxcar" or window == "rect":
        win = np.ones(frame_length, dtype=np.float64)
    else:
        win = signal.get_window(window, frame_length, fftbins=True)

    if len(x) < frame_length:
        pad_width = frame_length - len(x)
        x_padded = np.pad(x, (0, pad_width), mode="constant")
        return (x_padded * win)[np.newaxis, :], win

    n_frames = 1 + (len(x) - frame_length) // hop_length
    shape = (n_frames, frame_length)
    strides = (x.strides[0] * hop_length, x.strides[0])
    raw_frames = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    windowed_frames = raw_frames * win
    return windowed_frames, win


def preprocess(
    signal_arr: np.ndarray,
    sample_rate: int = 16000,
    dc_removal: bool = True,
    preemphasis_coeff: float = 0.97,
    bandpass_low_hz: float = 80.0,
    bandpass_high_hz: float = 8000.0,
    bandpass_order: int = 4,
) -> np.ndarray:
    """Executes the full 1D preprocessing chain on an audio waveform.

    Chain order:
    1. DC removal via mean subtraction.
    2. First-order high-pass pre-emphasis.
    3. 4th-order Butterworth bandpass filtering.

    Args:
        signal_arr: Raw 1D input audio waveform.
        sample_rate: Sampling frequency in Hz.
        dc_removal: If True, subtracts signal mean.
        preemphasis_coeff: High-pass pre-emphasis factor.
        bandpass_low_hz: Bandpass lower cutoff in Hz.
        bandpass_high_hz: Bandpass upper cutoff in Hz.
        bandpass_order: Butterworth filter order.

    Returns:
        Preprocessed 1D audio array.
    """
    x = np.asarray(signal_arr, dtype=np.float64)
    if dc_removal:
        x = remove_dc(x, method="mean")
    if preemphasis_coeff > 0.0:
        x = pre_emphasis(x, coeff=preemphasis_coeff)
    if bandpass_low_hz > 0.0:
        x = bandpass_filter(
            x,
            sample_rate=sample_rate,
            low_hz=bandpass_low_hz,
            high_hz=bandpass_high_hz,
            order=bandpass_order,
        )
    return x
