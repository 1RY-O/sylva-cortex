"""Synthetic SNR mixer with Domain Randomization for Sylva Cortex (Phase P4.6).

Applies realistic bioacoustic domain randomization:
1. Random time-stretching (+/- 10%) per clip to simulate engine throttle and call duration variance.
2. Random pitch-shifting (+/- 2 semitones) per clip to simulate engine speed and animal body size variation.
3. Random masking noise term (wind, rain, atmospheric turbulence) mixed onto threat signatures
   at 0 to +10 dB relative to threat signal.
4. Ambient background forest noise mixed across 7 SNR levels: {-10, -5, 0, 5, 10, 15, 20} dB.
5. Seeded RNG (seed=2026) for deterministic reproducibility.

Generates 846 balanced clips (840 mixtures + 6 pure samples) across 6 classes:
    chainsaw, gunshot, bird_call, rain, wind, forest_ambience.
"""
import csv
import os
from pathlib import Path
import sys
from typing import List, Dict, Any
import numpy as np
import scipy.signal as signal
import soundfile as sf
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent

# Ensure both scripts/ directory and REPO_ROOT are present on sys.path
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from download_data import generate_synthetic_acoustic_seeds
except ModuleNotFoundError:
    from scripts.download_data import generate_synthetic_acoustic_seeds


def load_config(config_path: Path = REPO_ROOT / "config.yaml") -> Dict[str, Any]:
    """Loads central configuration hyperparameters."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def time_stretch(audio: np.ndarray, rate: float = 1.0) -> np.ndarray:
    """Time-stretches audio array by linear interpolation (rate > 1: faster, rate < 1: slower)."""
    if abs(rate - 1.0) < 1e-4:
        return audio.copy()
    n_in = len(audio)
    n_out = max(16, int(round(n_in / rate)))
    t_in = np.linspace(0.0, 1.0, n_in, endpoint=True)
    t_out = np.linspace(0.0, 1.0, n_out, endpoint=True)
    return np.interp(t_out, t_in, audio)


def pitch_shift(audio: np.ndarray, n_semitones: float = 0.0) -> np.ndarray:
    """Pitch-shifts audio by n_semitones via resampling and duration restoration."""
    if abs(n_semitones) < 1e-4:
        return audio.copy()
    factor = 2.0 ** (n_semitones / 12.0)
    n_orig = len(audio)
    n_resampled = max(16, int(round(n_orig / factor)))

    t_orig = np.linspace(0.0, 1.0, n_orig, endpoint=True)
    t_resampled = np.linspace(0.0, 1.0, n_resampled, endpoint=True)

    # Change playback speed (shifts pitch)
    shifted = np.interp(t_resampled, t_orig, audio)
    # Restore duration to original length
    t_back = np.linspace(0.0, 1.0, n_orig, endpoint=True)
    t_shift = np.linspace(0.0, 1.0, len(shifted), endpoint=True)
    return np.interp(t_back, t_shift, shifted)


def calculate_noise_scaling(
    signal_arr: np.ndarray,
    noise_arr: np.ndarray,
    target_snr_db: float,
    eps: float = 1e-12,
) -> float:
    """Computes amplitude scaling factor for noise to achieve target SNR in dB."""
    p_sig = float(np.mean(signal_arr ** 2))
    p_noise = float(np.mean(noise_arr ** 2))
    if p_sig <= eps or p_noise <= eps:
        return 1.0

    target_noise_power = p_sig / (10.0 ** (target_snr_db / 10.0))
    scale = np.sqrt(target_noise_power / (p_noise + eps))
    return float(scale)


def create_mixed_dataset(
    config: Dict[str, Any],
    raw_dir: Path = REPO_ROOT / "data" / "raw",
    samples_dir: Path = REPO_ROOT / "data" / "samples",
    manifest_path: Path = REPO_ROOT / "data" / "manifest.csv",
    seed: int = 2026,
    regenerate_seeds: bool = True,
) -> List[Dict[str, Any]]:
    """Generates domain-randomized synthetic SNR sweep mixtures and writes manifest.csv."""
    samples_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    sr = int(config["audio"]["sample_rate"])
    snr_levels: List[float] = [float(x) for x in config["evaluation"]["snr_levels_db"]]
    mixtures_per_cell: int = int(config["evaluation"]["mixtures_per_cell"])

    # Ensure seed audio exists or force regenerate with cinematic physical models
    ambience_path = raw_dir / "forest_ambience_seed.wav"
    if regenerate_seeds or not ambience_path.exists():
        print(f"Regenerating cinematic acoustic seeds in {raw_dir} (f0=95Hz chainsaw, 60Hz gunshot, 1.5s echo)...")
        generate_synthetic_acoustic_seeds(raw_dir, sample_rate=sr, seed=seed)

    class_seed_files: Dict[str, Path] = {
        "chainsaw": raw_dir / "chainsaw_seed.wav",
        "gunshot": raw_dir / "gunshot_seed.wav",
        "bird_call": raw_dir / "bird_call_seed.wav",
        "rain": raw_dir / "rain_seed.wav",
        "wind": raw_dir / "wind_seed.wav",
        "forest_ambience": raw_dir / "forest_ambience_seed.wav",
    }

    class_signals: Dict[str, np.ndarray] = {}
    for label, seed_path in class_seed_files.items():
        data, _ = sf.read(str(seed_path))
        class_signals[label] = np.asarray(data, dtype=np.float64)

    ambience_full = class_signals["forest_ambience"]
    rain_full = class_signals["rain"]
    wind_full = class_signals["wind"]

    rng = np.random.default_rng(seed)
    clip_duration_s = 2.0
    clip_samples = int(clip_duration_s * sr)

    manifest_rows: List[Dict[str, Any]] = []

    # 1. Export pure baseline demo samples with cinematic volume
    for label, sig in class_signals.items():
        if len(sig) >= clip_samples:
            sig_clip = sig[:clip_samples].copy()
        else:
            sig_clip = np.pad(sig, (0, clip_samples - len(sig)), mode="wrap")

        # Peak normalize pure seeds: 1.0 for aggressive threats (chainsaw, gunshot), 0.95 for ambient/wildlife
        peak_pure = np.max(np.abs(sig_clip))
        if peak_pure > 1e-6:
            target_pure_peak = 1.0 if label in ["chainsaw", "gunshot"] else 0.95
            sig_clip = (sig_clip / peak_pure) * target_pure_peak

        sample_file_rel = f"data/samples/sample_pure_{label}.wav"
        out_full = REPO_ROOT / sample_file_rel
        sf.write(str(out_full), sig_clip.astype(np.float32), sr)

        manifest_rows.append({
            "file_path": sample_file_rel.replace("\\", "/"),
            "label": label,
            "source": "synthetic_physical_model_seed",
            "licence": "MIT / Synthetic Benchmark",
            "duration_sec": f"{clip_duration_s:.2f}",
            "sample_rate_hz": sr,
        })

    # 2. Generate Domain-Randomized SNR sweep mixtures
    target_classes = list(class_seed_files.keys())
    print(f"Generating domain-randomized synthetic SNR sweep mixtures (seed={seed})...")
    print(f"Target classes ({len(target_classes)}): {target_classes}")
    print(f"SNR levels (dB): {snr_levels}")
    print(f"Mixtures per cell: {mixtures_per_cell}")

    for label in target_classes:
        source_audio = class_signals[label]

        for snr_db in snr_levels:
            for mix_idx in range(mixtures_per_cell):
                # Sample random window from raw source (with extra margin for time-stretching)
                margin = int(0.4 * sr)
                slice_len = clip_samples + margin
                if len(source_audio) > slice_len:
                    max_start = len(source_audio) - slice_len
                    t_start = rng.integers(0, max_start)
                    raw_slice = source_audio[t_start : t_start + slice_len].copy()
                else:
                    raw_slice = np.pad(source_audio, (0, max(0, slice_len - len(source_audio))), mode="wrap")

                # Domain Randomization 1: Random Time-Stretching (+/- 10%)
                stretch_rate = float(rng.uniform(0.90, 1.10))
                stretched = time_stretch(raw_slice, rate=stretch_rate)

                # Ensure exact clip length
                if len(stretched) >= clip_samples:
                    s_clip = stretched[:clip_samples]
                else:
                    s_clip = np.pad(stretched, (0, clip_samples - len(stretched)), mode="wrap")

                # Domain Randomization 2: Random Pitch-Shifting (+/- 2 semitones)
                n_semitones = float(rng.uniform(-2.0, 2.0))
                s_clip = pitch_shift(s_clip, n_semitones=n_semitones)

                # Background primary ambience source
                if label == "forest_ambience":
                    # For forest ambience, background is random atmospheric turbulence
                    raw_noise = rng.normal(0.0, 0.08, clip_samples)
                    b_noise, a_noise = signal.butter(2, [100.0 / (sr / 2), 7000.0 / (sr / 2)], btype="bandpass")
                    b_clip = signal.lfilter(b_noise, a_noise, raw_noise)
                else:
                    if len(ambience_full) > clip_samples:
                        amb_start = rng.integers(0, len(ambience_full) - clip_samples)
                        b_clip = ambience_full[amb_start : amb_start + clip_samples].copy()
                    else:
                        b_clip = np.pad(ambience_full, (0, clip_samples - len(ambience_full)), mode="wrap")

                # Distance / orientation jitter (+/- 1.5 dB)
                jitter_db = rng.uniform(-1.5, 1.5)
                effective_snr = snr_db + jitter_db

                scale_factor = calculate_noise_scaling(s_clip, b_clip, target_snr_db=effective_snr)
                mixed = s_clip + scale_factor * b_clip

                # Domain Randomization 3: Masking noise term for threat classes
                # Mix a third random background noise (wind or rain) at 0 to +10 dB relative to threat signal
                if label in ["chainsaw", "gunshot"]:
                    masking_pool = [rain_full, wind_full]
                    chosen_mask = masking_pool[int(rng.integers(0, len(masking_pool)))]
                    if len(chosen_mask) > clip_samples:
                        mask_start = rng.integers(0, len(chosen_mask) - clip_samples)
                        m_clip = chosen_mask[mask_start : mask_start + clip_samples].copy()
                    else:
                        m_clip = np.pad(chosen_mask, (0, clip_samples - len(chosen_mask)), mode="wrap")

                    mask_snr_db = float(rng.uniform(0.0, 10.0))
                    mask_scale = calculate_noise_scaling(s_clip, m_clip, target_snr_db=mask_snr_db)
                    mixed += mask_scale * m_clip

                # Peak normalize clips: 1.0 for threats (chainsaw, gunshot) for cinematic impact, 0.95 for ambient/wildlife
                peak = np.max(np.abs(mixed))
                if peak > 1e-6:
                    target_mix_peak = 1.0 if label in ["chainsaw", "gunshot"] else 0.95
                    mixed = (mixed / peak) * target_mix_peak

                out_filename = f"data/samples/mix_{label}_snr{int(snr_db):+03d}db_idx{mix_idx:02d}.wav"
                out_path = REPO_ROOT / out_filename
                sf.write(str(out_path), mixed.astype(np.float32), sr)

                manifest_rows.append({
                    "file_path": out_filename.replace("\\", "/"),
                    "label": label,
                    "source": f"synthetic_mixture_snr_{snr_db:+03.0f}dB",
                    "licence": "MIT / Synthetic Benchmark",
                    "duration_sec": f"{clip_duration_s:.2f}",
                    "sample_rate_hz": sr,
                })

    # Write manifest.csv
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["file_path", "label", "source", "licence", "duration_sec", "sample_rate_hz"]
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\nManifest successfully written to {manifest_path} ({len(manifest_rows)} total clips)")
    return manifest_rows


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate synthetic mixed dataset for Sylva Cortex.")
    parser.add_argument("--regenerate-seeds", action="store_true", default=True, help="Force regenerate raw acoustic seeds with cinematic audio parameters")
    args = parser.parse_args()

    cfg = load_config()
    if args.regenerate_seeds:
        print("Regenerating acoustic seeds with 95Hz brutal chainsaw, 60Hz gunshot shockwave, and 1.5s canopy echo...")
        generate_synthetic_acoustic_seeds(REPO_ROOT / "data" / "raw", sample_rate=int(cfg["audio"]["sample_rate"]), seed=2026)

    rows = create_mixed_dataset(cfg, seed=2026, regenerate_seeds=args.regenerate_seeds)
    print(f"P4.6 dataset generation complete. Total samples registered: {len(rows)}")
