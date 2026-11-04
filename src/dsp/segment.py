"""Event segmentation module using robust adaptive energy thresholding.

Algorithms implemented:
1. Short-time frame energy computation.
   Reference: Rabiner, L. R., & Schafer, R. W. (2010). Theory and Applications
   of Digital Speech Processing. Pearson.
2. Hampel identifier adaptive thresholding: Median + k * MAD (Median Absolute Deviation).
   Reference: Hampel, F. R. (1974). The influence curve and its role in robust
   estimation. Journal of the American Statistical Association, 69(346), 383-393.
3. Temporal gap merging and minimum-duration pruning.
   Reference: Duda, R. O., Hart, P. E., & Stork, D. G. (2001). Pattern
   Classification (2nd ed.). Wiley-Interscience.
"""

from typing import List, Tuple, Dict, Any, Callable
import numpy as np

from src.dsp.features import extract_features


def compute_frame_energies(
    signal_arr: np.ndarray,
    frame_length: int = 400,
    hop_length: int = 160,
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes short-time frame energy and time stamps.

    Algorithm: Slides an analysis window across the 1D signal and computes the sum
    of squared sample values: E[t] = sum_{n=0}^{N-1} x[t * H + n]^2.
    Reference: Rabiner, L. R., & Schafer, R. W. (2010).

    Args:
        signal_arr: 1D audio waveform.
        frame_length: Window size in samples.
        hop_length: Hop size between frames in samples.

    Returns:
        Tuple of (energies, frame_times):
            - energies: 1D array of shape (n_frames,)
            - frame_times: 1D array of shape (n_frames,) in seconds
    """
    x = np.asarray(signal_arr, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(f"Expected 1D audio array, got shape {x.shape}")
    if len(x) < frame_length:
        return np.array([float(np.sum(x ** 2))], dtype=np.float64), np.array([0.0], dtype=np.float64)

    n_frames = 1 + (len(x) - frame_length) // hop_length
    shape = (n_frames, frame_length)
    strides = (x.strides[0] * hop_length, x.strides[0])
    frames = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)

    energies = np.sum(frames ** 2, axis=-1)
    frame_indices = np.arange(n_frames, dtype=np.float64)
    return energies, frame_indices


def compute_adaptive_threshold(
    frame_energies: np.ndarray,
    k_mad: float = 3.0,
) -> float:
    """Computes robust adaptive energy threshold using Median Absolute Deviation (MAD).

    Algorithm: Threshold = median(E) + k * MAD(E), where
        MAD(E) = median(|E - median(E)|).
    Provides breakdown-resistant outlier detection that automatically adjusts
    to varying background environmental noise levels without parametric assumptions.
    Reference: Hampel, F. R. (1974). J. Amer. Statist. Assoc., 69(346), 383-393.

    Args:
        frame_energies: 1D array of frame energies.
        k_mad: Multiplier for MAD deviation (default 3.0).

    Returns:
        Scalar threshold value >= 0.0.
    """
    e = np.asarray(frame_energies, dtype=np.float64)
    if len(e) == 0:
        return 0.0

    med = float(np.median(e))
    mad = float(np.median(np.abs(e - med)))
    return med + k_mad * mad


def merge_and_filter_segments(
    raw_segments: List[Tuple[float, float]],
    merge_gap_s: float = 0.150,
    min_segment_s: float = 0.060,
) -> List[Tuple[float, float]]:
    """Merges adjacent event segments and removes sub-minimum transients.

    Algorithm:
    1. Discard isolated raw segments shorter than min_segment_s to eliminate
       sub-minimum noise glitches.
    2. If start[i+1] - end[i] < merge_gap_s, fuse into single continuous event.
    3. Discard any resulting segment where (end - start) < min_segment_s.
    Reference: Duda, Hart, & Stork (2001).

    Args:
        raw_segments: List of (start_time, end_time) in seconds.
        merge_gap_s: Maximum gap to merge in seconds (150 ms default).
        min_segment_s: Minimum allowable duration in seconds (60 ms default).

    Returns:
        List of merged (start_time, end_time) tuples.
    """
    if not raw_segments:
        return []

    # First filter out sub-minimum isolated transients (< min_segment_s)
    valid_raw = [
        seg for seg in raw_segments
        if (seg[1] - seg[0]) >= min_segment_s
    ]
    if not valid_raw:
        return []

    # Sort segments by start time
    sorted_segs = sorted(valid_raw, key=lambda s: s[0])

    merged: List[List[float]] = []
    for s_start, s_end in sorted_segs:
        if not merged:
            merged.append([s_start, s_end])
        else:
            prev_start, prev_end = merged[-1]
            if s_start - prev_end < merge_gap_s:
                # Extend previous segment
                merged[-1][1] = max(prev_end, s_end)
            else:
                merged.append([s_start, s_end])

    # Final verification of minimum duration
    filtered = [
        (seg[0], seg[1])
        for seg in merged
        if (seg[1] - seg[0]) >= min_segment_s
    ]
    return filtered


def segment_events(
    signal_arr: np.ndarray,
    sample_rate: int = 16000,
    frame_length: int = 400,
    hop_length: int = 160,
    onset_k_mad: float = 3.0,
    min_segment_ms: float = 60.0,
    merge_gap_ms: float = 150.0,
    extract_features_fn: Callable[[np.ndarray], Dict[str, Any]] | None = None,
) -> List[Tuple[float, float, Dict[str, Any]]]:
    """Segments an audio recording into discrete acoustic event windows.

    Emits (start_time, end_time, features) tuples for each detected event.

    Args:
        signal_arr: 1D audio waveform.
        sample_rate: Sampling frequency in Hz.
        frame_length: Analysis frame length in samples (400 for 25 ms @ 16 kHz).
        hop_length: Frame hop in samples (160 for 10 ms @ 16 kHz).
        onset_k_mad: Multiplier k for MAD adaptive threshold.
        min_segment_ms: Minimum segment duration in milliseconds (60 ms).
        merge_gap_ms: Gap below which adjacent segments are merged (150 ms).
        extract_features_fn: Optional custom callable to extract features from
            a 1D segment slice. Defaults to `extract_features`.

    Returns:
        List of (start_time_s, end_time_s, features_dict) tuples.
    """
    x = np.asarray(signal_arr, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(f"Expected 1D audio array, got shape {x.shape}")
    if len(x) == 0:
        return []

    energies, frame_indices = compute_frame_energies(
        x,
        frame_length=frame_length,
        hop_length=hop_length,
    )

    threshold = compute_adaptive_threshold(energies, k_mad=onset_k_mad)
    active = energies >= threshold

    # Group contiguous active frames
    raw_segments: List[Tuple[float, float]] = []
    in_event = False
    start_frame = 0

    for idx, is_act in enumerate(active):
        if is_act and not in_event:
            in_event = True
            start_frame = idx
        elif not is_act and in_event:
            in_event = False
            start_t = (start_frame * hop_length) / sample_rate
            end_t = (idx * hop_length + frame_length) / sample_rate
            raw_segments.append((start_t, min(end_t, len(x) / sample_rate)))

    if in_event:
        start_t = (start_frame * hop_length) / sample_rate
        end_t = len(x) / sample_rate
        raw_segments.append((start_t, end_t))

    merged_segments = merge_and_filter_segments(
        raw_segments,
        merge_gap_s=merge_gap_ms / 1000.0,
        min_segment_s=min_segment_ms / 1000.0,
    )

    # Compute features for each detected event segment
    if extract_features_fn is None:
        def default_extractor(chunk: np.ndarray) -> Dict[str, Any]:
            return extract_features(
                chunk,
                sample_rate=sample_rate,
                n_fft=frame_length,
                hop_length=hop_length,
            )
        feature_fn = default_extractor
    else:
        feature_fn = extract_features_fn

    results: List[Tuple[float, float, Dict[str, Any]]] = []
    for start_t, end_t in merged_segments:
        start_sample = int(round(start_t * sample_rate))
        end_sample = int(round(end_t * sample_rate))
        start_sample = max(0, min(start_sample, len(x)))
        end_sample = max(start_sample, min(end_sample, len(x)))

        segment_audio = x[start_sample:end_sample]
        feats = feature_fn(segment_audio)
        results.append((start_t, end_t, feats))

    return results
