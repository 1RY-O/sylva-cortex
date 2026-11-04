"""Generates a synthetic chirp waveform and runs it through the full DSP pipeline,

saving the output spectrogram to figures/synthetic_chirp_spectrogram.png.
"""

from pathlib import Path
import numpy as np
import scipy.signal as signal
from PIL import Image

from src.dsp.preprocess import preprocess
from src.dsp.denoise import denoise
from src.dsp.features import compute_log_mel_spectrogram


def generate_and_save_chirp_spectrogram(
    output_path: Path = Path("figures/synthetic_chirp_spectrogram.png"),
    sample_rate: int = 16000,
    duration: float = 1.5,
    f0: float = 200.0,
    f1: float = 6000.0,
) -> np.ndarray:
    """Generates synthetic chirp, processes through DSP chain, and saves spectrogram image."""
    t = np.linspace(0.0, duration, int(sample_rate * duration), endpoint=False)
    # Synthetic linear chirp
    chirp_raw = signal.chirp(t, f0=f0, t1=duration, f1=f1, method="linear")
    # Inject DC bias and background noise
    rng = np.random.default_rng(42)
    chirp_noisy = chirp_raw + 0.35 + rng.normal(0.0, 0.05, len(t))

    # Full preprocessing chain: DC removal -> pre-emphasis -> Butterworth bandpass
    preprocessed = preprocess(
        chirp_noisy,
        sample_rate=sample_rate,
        dc_removal=True,
        preemphasis_coeff=0.97,
        bandpass_low_hz=80.0,
        bandpass_high_hz=8000.0,
        bandpass_order=4,
    )

    # Denoise: spectral subtraction
    denoised = denoise(
        preprocessed,
        sample_rate=sample_rate,
        n_fft=400,
        hop_length=160,
        oversubtraction_alpha=2.0,
        spectral_floor_beta=0.02,
        gain_smoothing_frames=3,
    )

    # Feature representation: Log-Mel spectrogram (40 bands)
    log_mel = compute_log_mel_spectrogram(
        denoised,
        sample_rate=sample_rate,
        n_fft=400,
        hop_length=160,
        n_mels=40,
    )

    # Render image with frequency on Y-axis (low at bottom), time on X-axis
    spec_matrix = log_mel.T[::-1, :]
    val_min, val_max = np.min(spec_matrix), np.max(spec_matrix)
    norm = (spec_matrix - val_min) / (val_max - val_min + 1e-10)
    img_array = (norm * 255.0).astype(np.uint8)

    img = Image.fromarray(img_array)
    img_resized = img.resize((600, 300), Image.Resampling.BILINEAR)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img_resized.save(output_path)
    return log_mel


if __name__ == "__main__":
    out_file = Path("figures/synthetic_chirp_spectrogram.png")
    spec = generate_and_save_chirp_spectrogram(out_file)
    print(f"Spectrogram saved successfully to {out_file} (shape {spec.shape})")

