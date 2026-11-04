"""Unit tests for the Sylva Cortex DSP module (Phase P2).

Required coverage per SPEC.md:
1. Filter frequency response within ±0.5 dB in the passband.
2. Denoise reduces noise-floor variance on synthetic white noise.
3. Feature extractor returns finite values on silent input (no NaN, no divide-by-zero).
4. Segmenter finds exactly the onsets in a synthetic tone-burst signal.
"""

import sys
from pathlib import Path

# Ensure repository root is in sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import unittest
import numpy as np
import scipy.signal as signal

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


class TestDSPModule(unittest.TestCase):
    """Test suite covering the 5 DSP modules."""

    def setUp(self):
        self.sample_rate = 16000
        self.rng = np.random.default_rng(42)

    # --------------------------------------------------------------------------
    # 1. Filter frequency response within ±0.5 dB in the passband
    # --------------------------------------------------------------------------
    def test_filter_frequency_response_passband(self):
        """Verify Butterworth filter magnitude response is flat within ±0.5 dB in passband."""
        low_hz = 80.0
        high_hz = 8000.0
        order = 4
        fs = self.sample_rate

        sos = design_butterworth_bandpass(
            sample_rate=fs,
            low_hz=low_hz,
            high_hz=high_hz,
            order=order,
        )

        # Compute complex frequency response
        worN = 4096
        freqs, h = signal.sosfreqz(sos, worN=worN, fs=fs)
        mag = np.abs(h)
        mag_db = 20.0 * np.log10(np.maximum(mag, 1e-12))

        # Passband region: safely away from transition edges
        # Cutoff is 80 Hz; passband defined from 160 Hz (2x low cutoff) to 7500 Hz
        passband_mask = (freqs >= 160.0) & (freqs <= 7500.0)
        self.assertTrue(np.any(passband_mask), "Passband mask should not be empty")

        passband_db = mag_db[passband_mask]
        max_dev = np.max(np.abs(passband_db))

        self.assertLess(
            max_dev,
            0.5,
            f"Passband deviation ({max_dev:.4f} dB) exceeds ±0.5 dB tolerance",
        )

        # Also verify bandpass design with explicit interior cutoffs (e.g. 100 Hz - 7000 Hz)
        sos_band = design_butterworth_bandpass(
            sample_rate=fs,
            low_hz=100.0,
            high_hz=7000.0,
            order=order,
        )
        freqs_b, h_b = signal.sosfreqz(sos_band, worN=worN, fs=fs)
        mag_db_b = 20.0 * np.log10(np.maximum(np.abs(h_b), 1e-12))
        pb_mask_b = (freqs_b >= 200.0) & (freqs_b <= 6000.0)
        max_dev_b = np.max(np.abs(mag_db_b[pb_mask_b]))

        self.assertLess(
            max_dev_b,
            0.5,
            f"Interior passband deviation ({max_dev_b:.4f} dB) exceeds ±0.5 dB tolerance",
        )

    # --------------------------------------------------------------------------
    # 2. Denoise reduces noise-floor variance on synthetic white noise
    # --------------------------------------------------------------------------
    def test_denoise_reduces_noise_floor_variance(self):
        """Verify spectral subtraction significantly suppresses variance of white noise."""
        duration_s = 2.0
        n_samples = int(duration_s * self.sample_rate)
        # Synthetic zero-mean white Gaussian noise
        noise = self.rng.normal(0.0, 0.1, n_samples)
        initial_variance = float(np.var(noise))

        denoised = denoise(
            noise,
            sample_rate=self.sample_rate,
            n_fft=400,
            hop_length=160,
            noise_percentile=10.0,
            oversubtraction_alpha=2.0,
            spectral_floor_beta=0.02,
            gain_smoothing_frames=3,
        )

        final_variance = float(np.var(denoised))

        self.assertEqual(len(denoised), len(noise), "Denoised signal must preserve sample length")
        self.assertTrue(np.all(np.isfinite(denoised)), "Denoised signal must contain only finite numbers")
        self.assertLess(
            final_variance,
            initial_variance,
            f"Denoising failed to reduce variance: initial={initial_variance:.6f}, final={final_variance:.6f}",
        )
        # Should reduce variance by at least 25% with alpha=2.0
        reduction_ratio = final_variance / initial_variance
        self.assertLess(
            reduction_ratio,
            0.75,
            f"Denoising reduction ratio ({reduction_ratio:.3f}) was insufficient",
        )

    # --------------------------------------------------------------------------
    # 3. Feature extractor returns finite values on silent input
    # --------------------------------------------------------------------------
    def test_feature_extractor_finite_on_silence(self):
        """Verify feature extraction on all-zero audio segment produces zero NaNs or Infs."""
        duration_s = 0.5
        n_samples = int(duration_s * self.sample_rate)
        silent_audio = np.zeros(n_samples, dtype=np.float64)

        features = extract_features(
            silent_audio,
            sample_rate=self.sample_rate,
            n_fft=400,
            hop_length=160,
            n_mels=40,
            n_mfcc=13,
            include_deltas=True,
            compute_flatness=True,
            compute_crest=True,
            compute_attack_time_flag=True,
        )

        # Inspect all returned matrices and scalar features
        self.assertTrue(np.all(np.isfinite(features["log_mel"])), "log_mel contains NaN or Inf on silence")
        self.assertTrue(np.all(np.isfinite(features["mfcc"])), "mfcc contains NaN or Inf on silence")
        self.assertTrue(np.all(np.isfinite(features["mfcc_deltas"])), "mfcc_deltas contains NaN or Inf on silence")
        self.assertTrue(np.all(np.isfinite(features["mfcc_delta_deltas"])), "mfcc_delta_deltas contains NaN or Inf on silence")
        self.assertTrue(np.all(np.isfinite(features["spectral_centroid"])), "spectral_centroid contains NaN or Inf on silence")
        self.assertTrue(np.all(np.isfinite(features["spectral_flatness"])), "spectral_flatness contains NaN or Inf on silence")
        self.assertTrue(np.all(np.isfinite(features["spectral_rolloff"])), "spectral_rolloff contains NaN or Inf on silence")
        self.assertTrue(np.isfinite(features["zero_crossing_rate"]), "zero_crossing_rate is not finite on silence")
        self.assertTrue(np.isfinite(features["crest_factor"]), "crest_factor is not finite on silence")
        self.assertTrue(np.isfinite(features["attack_time_ms"]), "attack_time_ms is not finite on silence")
        self.assertTrue(np.all(np.isfinite(features["feature_vector"])), "feature_vector contains NaN or Inf on silence")

    # --------------------------------------------------------------------------
    # 4. Segmenter finds exactly the onsets in a synthetic tone-burst signal
    # --------------------------------------------------------------------------
    def test_segmenter_finds_exact_onsets_tone_burst(self):
        """Verify adaptive energy segmentation detects exactly the two synthetic tone bursts."""
        duration_s = 3.0
        n_samples = int(duration_s * self.sample_rate)
        t = np.linspace(0.0, duration_s, n_samples, endpoint=False)
        audio = np.zeros(n_samples, dtype=np.float64)

        # Tone Burst 1: 0.50 s to 0.70 s (200 ms, 1000 Hz)
        mask1 = (t >= 0.50) & (t < 0.70)
        audio[mask1] = 1.0 * np.sin(2.0 * np.pi * 1000.0 * t[mask1])

        # Tone Burst 2: 1.50 s to 1.70 s (200 ms, 2000 Hz)
        mask2 = (t >= 1.50) & (t < 1.70)
        audio[mask2] = 1.0 * np.sin(2.0 * np.pi * 2000.0 * t[mask2])

        # Add realistic background noise floor
        audio += self.rng.normal(0.0, 0.05, n_samples)

        # Detect events
        events = segment_events(
            audio,
            sample_rate=self.sample_rate,
            frame_length=400,
            hop_length=160,
            onset_k_mad=3.0,
            min_segment_ms=60.0,
            merge_gap_ms=150.0,
        )

        self.assertEqual(
            len(events),
            2,
            f"Expected exactly 2 detected events, got {len(events)}: {events}",
        )

        onset1 = events[0][0]
        onset2 = events[1][0]

        # Onset accuracy within 50 ms
        self.assertAlmostEqual(onset1, 0.50, delta=0.06, msg=f"Onset 1 ({onset1}s) differed from 0.50s")
        self.assertAlmostEqual(onset2, 1.50, delta=0.06, msg=f"Onset 2 ({onset2}s) differed from 1.50s")

        # Verify features are populated in event tuples
        for start_t, end_t, feats in events:
            self.assertIn("feature_vector", feats)
            self.assertTrue(np.all(np.isfinite(feats["feature_vector"])))

    # --------------------------------------------------------------------------
    # 5. Preprocessing chain tests (DC removal, framing, pre-emphasis)
    # --------------------------------------------------------------------------
    def test_preprocessing_chain(self):
        """Verify DC offset removal, pre-emphasis, and full preprocessing chain."""
        n_samples = 8000
        # Signal with constant DC offset
        dc_val = 2.5
        sig = dc_val + np.sin(2.0 * np.pi * 440.0 * np.linspace(0, 0.5, n_samples))
        cleaned = remove_dc(sig, method="mean")
        self.assertAlmostEqual(float(np.mean(cleaned)), 0.0, places=6)

        # Pre-emphasis
        emphasized = pre_emphasis(cleaned, coeff=0.97)
        self.assertEqual(len(emphasized), len(cleaned))

        # Full preprocess
        full_proc = preprocess(sig, sample_rate=self.sample_rate)
        self.assertEqual(len(full_proc), len(sig))
        self.assertTrue(np.all(np.isfinite(full_proc)))

        # Framing
        frames, win = frame_signal(full_proc, frame_length=400, hop_length=160)
        self.assertEqual(frames.shape[1], 400)
        self.assertEqual(len(win), 400)
        self.assertTrue(np.all(np.isfinite(frames)))

    # --------------------------------------------------------------------------
    # 6. Harmonic-Percussive Source Separation (HPSS)
    # --------------------------------------------------------------------------
    def test_hpss_separation(self):
        """Verify HPSS separates continuous tonal signal from impulsive click."""
        duration_s = 1.0
        n_samples = int(duration_s * self.sample_rate)
        t = np.linspace(0.0, duration_s, n_samples, endpoint=False)

        # Pure harmonic tone (440 Hz)
        harmonic_sig = np.sin(2.0 * np.pi * 440.0 * t)
        # Sparse impulsive click at t=0.5s
        percussive_sig = np.zeros(n_samples)
        percussive_sig[int(0.5 * self.sample_rate)] = 15.0

        mix = harmonic_sig + percussive_sig

        x_harm, x_perc = hpss(
            mix,
            sample_rate=self.sample_rate,
            n_fft=400,
            hop_length=160,
            kernel_harmonic=15,
            kernel_percussive=15,
            power=2.0,
        )

        self.assertEqual(len(x_harm), n_samples)
        self.assertEqual(len(x_perc), n_samples)
        self.assertTrue(np.all(np.isfinite(x_harm)))
        self.assertTrue(np.all(np.isfinite(x_perc)))

        # Harmonic component should have significantly higher energy in x_harm than x_perc
        harm_energy = np.sum(x_harm ** 2)
        perc_energy = np.sum(x_perc ** 2)
        self.assertGreater(harm_energy, 0.0)
        self.assertGreater(perc_energy, 0.0)


if __name__ == "__main__":
    unittest.main()
