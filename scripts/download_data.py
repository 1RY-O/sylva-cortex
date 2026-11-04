"""Dataset download registry and metadata manager for Sylva Cortex.

Lists candidate external datasets with official URLs, licences, and extraction targets:
- ESC-50 (Environmental Sound Classification, 50 classes)
- UrbanSound8K (Urban sound excerpts, 10 classes)
- Xeno-canto (Avian and wildlife bioacoustics)
- FSD50K (Freesound Dataset 50K)

Under offline / network-isolated hackathon constraints, this script provides
offline metadata generation and acoustic physical model synthesis so that
downstream pipelines, evaluation sweeps, and test suites run reproducibly without
external network calls.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import scipy.signal as signal
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent.parent

CANDIDATE_DATASETS: Dict[str, Dict[str, Any]] = {
    "ESC-50": {
        "description": "2,000 labelled environmental audio recordings across 50 classes.",
        "licence": "CC-BY-NC 3.0",
        "url": "https://github.com/karolpiczak/ESC-50/archive/master.zip",
        "target_classes": ["chainsaw", "rain", "insects", "crow", "dog", "wind"],
        "citation": "Piczak, K. J. (2015). ESC: Dataset for Environmental Sound Classification. ACM Multimedia.",
    },
    "UrbanSound8K": {
        "description": "8,732 labelled excerpts (<4s) of urban sounds from 10 classes.",
        "licence": "CC-BY-NC 3.0",
        "url": "https://zenodo.org/record/1203745/files/UrbanSound8K.tar.gz",
        "target_classes": ["gun_shot", "engine_idling", "drilling", "siren"],
        "citation": "Salamon, J., Jacoby, C., & Bello, J. P. (2014). A Dataset and Taxonomy for Urban Sound Research. ACM MM.",
    },
    "Xeno-canto": {
        "description": "Community database of wildlife and bird vocalisations from around the globe.",
        "licence": "CC-BY-NC-SA 4.0",
        "url": "https://xeno-canto.org/api/2/recordings",
        "target_classes": ["bird_call", "mammal_call"],
        "citation": "Xeno-canto Foundation (https://xeno-canto.org).",
    },
    "FSD50K": {
        "description": "51,197 Freesound audio tracks under AudioSet ontology.",
        "licence": "CC-BY 4.0",
        "url": "https://zenodo.org/record/4060432",
        "target_classes": ["chainsaw", "rain", "wind", "speech", "tools"],
        "citation": "Fonseca, E. et al. (2021). FSD50K: an open dataset of everyday sounds. IEEE/ACM TASLP.",
    },
}


def print_dataset_info() -> None:
    """Prints catalogue of candidate datasets and licensing terms."""
    print("=" * 80)
    print("Sylva Cortex — Candidate Bioacoustic & Environmental Datasets")
    print("=" * 80)
    for name, meta in CANDIDATE_DATASETS.items():
        print(f"\n[{name}]")
        print(f"  Description:    {meta['description']}")
        print(f"  Licence:        {meta['licence']}")
        print(f"  Official URL:   {meta['url']}")
        print(f"  Target Classes: {', '.join(meta['target_classes'])}")
        print(f"  Citation:       {meta['citation']}")
    print("\n" + "=" * 80)


def generate_synthetic_acoustic_seeds(
    raw_dir: Path,
    sample_rate: int = 16000,
    duration_s: float = 10.0,
    seed: int = 1337,
) -> Dict[str, Path]:
    """Generates synthetic acoustic seed signals representing physical sound sources.

    Used strictly for offline testing when external network downloading is restricted.
    Models physical acoustic phenomena:
    - Forest ambience: low-passed continuous 1/f atmospheric fluctuation.
    - Chainsaw: high-order harmonic comb filter with engine mechanical buzz.
    - Gunshot: multiple gunshot shockwaves (<5ms) with exponential acoustic reverberation.
    - Rain: dense random Poisson droplet impacts.
    - Wind: low-frequency band-limited turbulence.
    - Bird call: rich sequence of frequency-modulated avian chirps and whistles.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    n_samples = int(duration_s * sample_rate)
    t = np.linspace(0.0, duration_s, n_samples, endpoint=False)
    seeds: Dict[str, Path] = {}

    # 1. Forest Ambience
    # Pink/brown noise approximation with gentle leaf rustle
    white = rng.normal(0.0, 0.1, n_samples)
    b, a = signal.butter(2, [150.0 / (sample_rate / 2), 2500.0 / (sample_rate / 2)], btype="bandpass")
    ambience = signal.lfilter(b, a, white)
    amb_path = raw_dir / "forest_ambience_seed.wav"
    sf.write(str(amb_path), ambience.astype(np.float32), sample_rate)
    seeds["forest_ambience"] = amb_path

    # 2. Chainsaw (Threat class) - Brutalized mechanical engine with 95 Hz sawtooth + aggressive rasp + violent cutting teeth chatter
    # Two-stroke combustion displacement: deepened fundamental f0 = 95.0 Hz with throttle modulation
    f0 = 95.0
    throttle_mod = 0.70 + 0.30 * np.sin(2.0 * np.pi * 7.5 * t)
    # Low-frequency sawtooth wave modeling reciprocating piston stroke and cylinder exhaust at 95 Hz
    saw_engine = signal.sawtooth(2.0 * np.pi * f0 * t + 0.15 * np.sin(2.0 * np.pi * 7.0 * t))

    # Engine harmonic combustion buzz (odd and even harmonics up to 3.5 kHz)
    combustion_harmonics = np.zeros(n_samples)
    for h_idx in range(1, 36):
        freq_h = f0 * h_idx + rng.uniform(-1.5, 1.5)
        if freq_h < sample_rate / 2:
            amp_h = (1.0 / (h_idx ** 0.75)) * (1.0 + 0.15 * rng.uniform(-1, 1))
            combustion_harmonics += amp_h * np.sin(2.0 * np.pi * freq_h * t + rng.uniform(0, 2 * np.pi))

    # Amplified high-frequency chain guide-bar and cutting teeth friction noise (2.0 kHz - 7.0 kHz) for violent presence
    b_teeth, a_teeth = signal.butter(3, [2000.0 / (sample_rate / 2), 7000.0 / (sample_rate / 2)], btype="bandpass")
    teeth_raw = signal.lfilter(b_teeth, a_teeth, rng.normal(0.0, 1.5, n_samples))
    # Teeth contact chatter modulated at tooth strike frequency (~480 Hz)
    teeth_mod = 0.50 + 0.50 * np.sin(2.0 * np.pi * 480.0 * t)
    cutting_teeth = teeth_raw * teeth_mod

    # Exhaust pipe resonance (bandpassed mechanical growl around 1.1 kHz - 2.5 kHz)
    b_exh, a_exh = signal.butter(2, [1100.0 / (sample_rate / 2), 2500.0 / (sample_rate / 2)], btype="bandpass")
    exhaust_growl = signal.lfilter(b_exh, a_exh, rng.normal(0.0, 0.9, n_samples))

    # Composite physical model with violent cutting teeth chatter weight (0.95)
    raw_chainsaw = (
        0.60 * saw_engine * throttle_mod
        + 0.40 * combustion_harmonics * throttle_mod
        + 0.95 * cutting_teeth
        + 0.35 * exhaust_growl
    )
    # Aggressive rasp distortion using np.tanh(x * 3.0) for authentic harsh 2-stroke cylinder saturation
    chainsaw_mechanical = np.tanh(raw_chainsaw * 3.0)

    # Scale peak amplitude to 1.0 (cinematic maximum volume)
    cs_peak = np.max(np.abs(chainsaw_mechanical))
    chainsaw = (chainsaw_mechanical / (cs_peak + 1e-9)) * 1.0
    cs_path = raw_dir / "chainsaw_seed.wav"
    sf.write(str(cs_path), chainsaw.astype(np.float32), sample_rate)
    seeds["chainsaw"] = cs_path

    # 3. Gunshot (Threat class) - Heavy Black-Powder Musket / Flintlock Rifle Model
    # Features: Sharp flint strike click, massive 45-60 Hz powder blast thud, 300ms black-powder hiss, and 2.0s cavernous canopy echo
    gunshot = np.zeros(n_samples)
    # Shots spaced by 2.6s so each heavy black-powder blast carries its full 2.0-second reverberation tail
    shot_times = [0.01, 2.6, 5.2, 7.8]

    # Deep bandpass filter for cavernous forest canopy acoustic reverberation (120 Hz - 2400 Hz)
    b_reverb, a_reverb = signal.butter(2, [120.0 / (sample_rate / 2), 2400.0 / (sample_rate / 2)], btype="bandpass")
    reverb_noise = signal.lfilter(b_reverb, a_reverb, rng.normal(0.0, 1.0, n_samples))

    for shot_time in shot_times:
        onset_idx = int(shot_time * sample_rate)
        # 2.2s window to comfortably capture the full 2.0-second cavernous canopy echo
        shot_dur = int(2.2 * sample_rate)
        if onset_idx + shot_dur > n_samples:
            shot_dur = n_samples - onset_idx
        t_shot = np.linspace(0.0, shot_dur / sample_rate, shot_dur, endpoint=False)

        # 1) Sharp "flint strike" crack (high frequency click/snap at t=0.01s as cock strikes the frizzen)
        b_flint, a_flint = signal.butter(3, [3200.0 / (sample_rate / 2), 7500.0 / (sample_rate / 2)], btype="bandpass")
        flint_noise = signal.lfilter(b_flint, a_flint, rng.normal(0.0, 1.3, shot_dur))
        flint_env = np.exp(-t_shot / 0.0035)  # ultra-fast 3.5ms spark click
        flint_strike = flint_noise * flint_env * 1.6

        # 2) Massive low-frequency "powder blast" thud (centered at 45 Hz - 60 Hz with 12-15ms attack and long rumbling decay)
        # Slower 12-15ms attack simulating deflagration of black powder charge followed by deep rumbling decay
        attack_env = 1.0 - np.exp(-t_shot / 0.012)
        decay_env = np.exp(-t_shot / 0.20)
        blast_envelope = attack_env * decay_env

        # Heavy sub-bass fundamentals at 48 Hz and 58 Hz + low-frequency combustion rumble
        blast_osc = (
            0.65 * np.sin(2.0 * np.pi * 48.0 * t_shot)
            + 0.45 * np.sin(2.0 * np.pi * 58.0 * t_shot)
            + 0.30 * np.sin(2.0 * np.pi * 34.0 * t_shot)
        )
        b_rumble, a_rumble = signal.butter(2, 110.0 / (sample_rate / 2), btype="lowpass")
        rumble_noise = signal.lfilter(b_rumble, a_rumble, rng.normal(0.0, 1.4, shot_dur))
        # Heavy non-linear soft-clipped powder blast thud
        powder_blast = np.tanh(2.6 * (blast_osc + 0.55 * rumble_noise)) * blast_envelope * 2.8

        # 3) "Black powder" hiss: bandpassed noise layer (1.5 kHz - 4.0 kHz) that decays over 300ms
        b_hiss, a_hiss = signal.butter(2, [1500.0 / (sample_rate / 2), 4000.0 / (sample_rate / 2)], btype="bandpass")
        hiss_noise = signal.lfilter(b_hiss, a_hiss, rng.normal(0.0, 1.1, shot_dur))
        # Smooth attack at blast, decaying to near zero within 300ms
        hiss_env = (1.0 - np.exp(-t_shot / 0.010)) * np.exp(-t_shot / 0.075) * np.where(t_shot <= 0.32, 1.0, np.exp(-(t_shot - 0.32) / 0.02))
        black_powder_hiss = hiss_noise * hiss_env * 1.8

        # 4) Extended canopy echo reverb to 2.0 seconds with deep cavernous delays simulating boom carrying through trees
        echo_envelope = np.exp(-t_shot / 0.55) * np.where(t_shot <= 2.0, 1.0, np.exp(-(t_shot - 2.0) / 0.06))
        reverb_tail = reverb_noise[onset_idx : onset_idx + shot_dur] * echo_envelope * 0.65

        # Cavernous multi-tap reflections off tree stands, valleys, and slopes across 2.0s
        cavern_taps = [
            (0.08, 0.55),  # near tree trunk reflection
            (0.24, 0.45),  # surrounding stand reverberation
            (0.48, 0.35),  # mid-canopy delay
            (0.85, 0.28),  # valley floor reflection
            (1.25, 0.20),  # distant ridge reflection
            (1.65, 0.15),  # far canopy reflection
            (1.95, 0.10),  # lingering cavernous atmospheric tail
        ]
        for delay_s, gain in cavern_taps:
            delay_idx = int(delay_s * sample_rate)
            win_len = int(0.16 * sample_rate)
            if delay_idx + win_len < shot_dur:
                reverb_tail[delay_idx : delay_idx + win_len] += (
                    gain * reverb_noise[onset_idx + delay_idx : onset_idx + delay_idx + win_len]
                )

        shot_composite = flint_strike + powder_blast + black_powder_hiss + reverb_tail
        gunshot[onset_idx : onset_idx + shot_dur] += shot_composite

    # Scale peak amplitude to 1.0 (cinematic maximum volume)
    gs_peak = np.max(np.abs(gunshot))
    gunshot = (gunshot / (gs_peak + 1e-9)) * 1.0
    gs_path = raw_dir / "gunshot_seed.wav"
    sf.write(str(gs_path), gunshot.astype(np.float32), sample_rate)
    seeds["gunshot"] = gs_path

    # 4. Rain (Ambient class)
    rain_noise = rng.normal(0.0, 0.18, n_samples)
    b_rain, a_rain = signal.butter(2, 750.0 / (sample_rate / 2), btype="highpass")
    rain = signal.lfilter(b_rain, a_rain, rain_noise)
    rain_path = raw_dir / "rain_seed.wav"
    sf.write(str(rain_path), (rain / np.max(np.abs(rain)) * 0.65).astype(np.float32), sample_rate)
    seeds["rain"] = rain_path

    # 5. Wind (Ambient class)
    wind_noise = rng.normal(0.0, 0.22, n_samples)
    b_wind, a_wind = signal.butter(2, 220.0 / (sample_rate / 2), btype="lowpass")
    wind = signal.lfilter(b_wind, a_wind, wind_noise)
    wind_mod = 0.5 + 0.5 * np.sin(2.0 * np.pi * 0.3 * t)
    wind = wind * wind_mod
    wind_path = raw_dir / "wind_seed.wav"
    sf.write(str(wind_path), (wind / np.max(np.abs(wind)) * 0.70).astype(np.float32), sample_rate)
    seeds["wind"] = wind_path

    # 6. Bird Call (Wildlife class)
    bird = np.zeros(n_samples)
    for c_idx in range(12):
        start_t = 0.4 + c_idx * 0.75
        dur_c = 0.22
        if start_t + dur_c > duration_s:
            break
        idx_c = (t >= start_t) & (t < start_t + dur_c)
        tc = t[idx_c] - start_t
        f_start = 2200.0 + (c_idx % 4) * 500.0
        f_end = f_start + 1800.0
        phase = 2.0 * np.pi * (f_start * tc + 0.5 * (f_end - f_start) * (tc ** 2) / dur_c)
        window = np.sin(np.pi * tc / dur_c)
        bird[idx_c] += window * np.sin(phase)
    bird_path = raw_dir / "bird_call_seed.wav"
    sf.write(str(bird_path), (bird / (np.max(np.abs(bird)) + 1e-6) * 0.85).astype(np.float32), sample_rate)
    seeds["bird_call"] = bird_path

    print(f"Generated {len(seeds)} synthetic acoustic seed waveforms (10.0s each, amplified presence) in {raw_dir}")
    return seeds


def main() -> None:
    parser = argparse.ArgumentParser(description="Sylva Cortex Data Registry & Fetcher")
    parser.add_argument("--info", action="store_true", help="Print candidate datasets and licences")
    parser.add_argument("--generate-seeds", action="store_true", help="Generate offline synthetic seed audio for tests")
    parser.add_argument("--raw-dir", type=str, default="data/raw", help="Target directory for raw data")
    args = parser.parse_args()

    if args.info or not args.generate_seeds:
        print_dataset_info()

    if args.generate_seeds:
        raw_path = REPO_ROOT / args.raw_dir
        generate_synthetic_acoustic_seeds(raw_path)


if __name__ == "__main__":
    main()

