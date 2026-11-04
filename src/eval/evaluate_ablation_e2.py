"""Experiment E2: DSP Contribution Ablation (Raw Audio vs. Denoised Audio).

Evaluates the quantitative contribution of the classical DSP denoising stage
(10th-percentile stationary noise floor tracking, Berouti spectral subtraction,
and Wiener temporal smoothing).

Compares:
  Condition A (RAW): Preprocessing -> [Denoising BYPASSED] -> Feature Extraction -> Classifier
  Condition B (DENOISED): Preprocessing -> Spectral Subtraction Denoise -> Feature Extraction -> Classifier

Generates:
  1. results/ablation_e2.json: Granular SNR degradation comparisons.
  2. figures/e2_denoise_ablation.png: Comparative F1-score vs SNR plot.
"""

import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    from sklearn.ensemble import HistGradientBoostingClassifier

from src.dsp.preprocess import preprocess
from src.dsp.denoise import denoise
from src.dsp.features import extract_features


def load_config(config_path: Path = REPO_ROOT / "config.yaml") -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def extract_features_dual_condition(
    audio_path: Path,
    sample_rate: int = 16000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extracts feature vectors for both RAW (bypassing denoise) and DENOISED conditions."""
    audio, sr = sf.read(str(audio_path))
    if audio.ndim > 1:
        audio = np.mean(audio, axis=-1)
    audio = np.asarray(audio, dtype=np.float64)

    # 1. Preprocessing (DC removal, pre-emphasis, Butterworth bandpass)
    proc = preprocess(
        audio,
        sample_rate=sample_rate,
        dc_removal=True,
        preemphasis_coeff=0.97,
        bandpass_low_hz=80.0,
        bandpass_high_hz=8000.0,
        bandpass_order=4,
    )

    # Condition A: Raw preprocessed audio directly into feature extractor
    feats_raw = extract_features(
        proc,
        sample_rate=sample_rate,
        n_fft=400,
        hop_length=160,
        n_mels=40,
        n_mfcc=13,
        include_deltas=True,
        compute_flatness=True,
        compute_crest=True,
        compute_attack_time_flag=True,
    )

    # Condition B: Spectral subtraction denoising before feature extractor
    clean = denoise(
        proc,
        sample_rate=sample_rate,
        n_fft=400,
        hop_length=160,
        noise_percentile=10.0,
        oversubtraction_alpha=2.0,
        spectral_floor_beta=0.02,
        gain_smoothing_frames=3,
    )
    feats_denoised = extract_features(
        clean,
        sample_rate=sample_rate,
        n_fft=400,
        hop_length=160,
        n_mels=40,
        n_mfcc=13,
        include_deltas=True,
        compute_flatness=True,
        compute_crest=True,
        compute_attack_time_flag=True,
    )

    return feats_raw["feature_vector"], feats_denoised["feature_vector"]


def build_classifier(random_seed: int = 42):
    """Instantiates GBDT classifier."""
    if HAS_LIGHTGBM:
        return lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            random_state=random_seed,
            verbose=-1,
            force_col_wise=True,
        )
    return HistGradientBoostingClassifier(max_iter=100, random_state=random_seed)


def run_e2_ablation(
    manifest_path: Path = REPO_ROOT / "data" / "manifest.csv",
    out_json_path: Path = REPO_ROOT / "results" / "ablation_e2.json",
    out_plot_path: Path = REPO_ROOT / "figures" / "e2_denoise_ablation.png",
) -> Dict[str, Any]:
    """Executes Experiment E2 ablation study."""
    cfg = load_config()
    sr = int(cfg["audio"]["sample_rate"])
    seed = int(cfg["evaluation"].get("seed", 1337))

    with open(manifest_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"Loading {len(rows)} audio clips and extracting dual-condition features (Raw vs Denoised)...")

    raw_features: List[np.ndarray] = []
    denoised_features: List[np.ndarray] = []
    labels: List[str] = []
    groups: List[int] = []
    snr_levels: List[float | None] = []

    for idx, r in enumerate(rows):
        audio_file = REPO_ROOT / r["file_path"]
        f_raw, f_clean = extract_features_dual_condition(audio_file, sample_rate=sr)

        raw_features.append(f_raw)
        denoised_features.append(f_clean)
        labels.append(r["label"])

        # Mixture index grouping for leakage prevention
        m_idx = re.search(r"idx(\d+)", r["file_path"])
        groups.append(int(m_idx.group(1)) if m_idx else 1000 + idx)

        # SNR parsing
        m_snr = re.search(r"snr([+-]?\d+)db", r["file_path"], re.IGNORECASE)
        snr_levels.append(float(m_snr.group(1)) if m_snr else None)

    X_raw = np.array(raw_features, dtype=np.float32)
    X_denoised = np.array(denoised_features, dtype=np.float32)
    groups = np.array(groups, dtype=np.int32)

    le = LabelEncoder()
    y = le.fit_transform(labels)
    classes = list(le.classes_)

    print(f"Features extracted: X_raw {X_raw.shape}, X_denoised {X_denoised.shape}")

    # 1. Evaluate 5-fold CV on both conditions using Out-Of-Fold (OOF) validation
    gkf = GroupKFold(n_splits=5)
    cv_raw_acc, cv_raw_f1 = [], []
    cv_clean_acc, cv_clean_f1 = [], []
    oof_preds_raw = np.zeros(len(y), dtype=np.int32)
    oof_preds_clean = np.zeros(len(y), dtype=np.int32)

    print("\n--- Running 5-Fold Cross-Validation for Both Conditions ---")
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_raw, y, groups=groups), start=1):
        # Model A: Raw
        clf_raw = build_classifier(seed + fold)
        clf_raw.fit(X_raw[train_idx], y[train_idx])
        pred_raw = clf_raw.predict(X_raw[val_idx])
        oof_preds_raw[val_idx] = pred_raw
        cv_raw_acc.append(float(accuracy_score(y[val_idx], pred_raw)))
        cv_raw_f1.append(float(f1_score(y[val_idx], pred_raw, average="macro", zero_division=0)))

        # Model B: Denoised
        clf_clean = build_classifier(seed + fold)
        clf_clean.fit(X_denoised[train_idx], y[train_idx])
        pred_clean = clf_clean.predict(X_denoised[val_idx])
        oof_preds_clean[val_idx] = pred_clean
        cv_clean_acc.append(float(accuracy_score(y[val_idx], pred_clean)))
        cv_clean_f1.append(float(f1_score(y[val_idx], pred_clean, average="macro", zero_division=0)))

    # 2. Granular per-SNR breakdown using Out-Of-Fold predictions
    unique_snrs = sorted(list({s for s in snr_levels if s is not None}))
    snr_comparison: Dict[str, Any] = {}

    curve_raw_f1: List[float] = []
    curve_clean_f1: List[float] = []
    curve_delta_f1: List[float] = []

    print("\n--- SNR Sweep Performance Comparison (Out-of-Fold Validation: Raw vs Denoised) ---")
    for snr in unique_snrs:
        mask = [s == snr for s in snr_levels]
        idx_sub = np.where(mask)[0]

        y_sub = y[idx_sub]
        p_raw_sub = oof_preds_raw[idx_sub]
        p_clean_sub = oof_preds_clean[idx_sub]

        f1_r = float(f1_score(y_sub, p_raw_sub, average="macro", zero_division=0))
        acc_r = float(accuracy_score(y_sub, p_raw_sub))
        f1_c = float(f1_score(y_sub, p_clean_sub, average="macro", zero_division=0))
        acc_c = float(accuracy_score(y_sub, p_clean_sub))
        delta = f1_c - f1_r

        curve_raw_f1.append(f1_r)
        curve_clean_f1.append(f1_c)
        curve_delta_f1.append(delta)

        snr_comparison[str(int(snr))] = {
            "snr_db": snr,
            "sample_count": len(idx_sub),
            "raw": {"accuracy": acc_r, "f1_macro": f1_r},
            "denoised": {"accuracy": acc_c, "f1_macro": f1_c},
            "f1_improvement": delta,
        }
        print(f"SNR {snr:+03.0f} dB | Raw F1: {f1_r:.4f} | Denoised F1: {f1_c:.4f} | Gain: {delta:+.4f}")

    results_payload = {
        "experiment": "E2_dsp_denoising_ablation",
        "description": "Comparison of classifier performance with vs without classical spectral subtraction denoising",
        "cv_5fold": {
            "raw": {
                "mean_accuracy": float(np.mean(cv_raw_acc)),
                "std_accuracy": float(np.std(cv_raw_acc)),
                "mean_f1_macro": float(np.mean(cv_raw_f1)),
                "std_f1_macro": float(np.std(cv_raw_f1)),
            },
            "denoised": {
                "mean_accuracy": float(np.mean(cv_clean_acc)),
                "std_accuracy": float(np.std(cv_clean_acc)),
                "mean_f1_macro": float(np.mean(cv_clean_f1)),
                "std_f1_macro": float(np.std(cv_clean_f1)),
            },
            "overall_cv_f1_gain": float(np.mean(cv_clean_f1) - np.mean(cv_raw_f1)),
        },
        "by_snr": snr_comparison,
        "classes": classes,
    }

    # Save results to JSON
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)
    print(f"\nSaved E2 ablation results to: {out_json_path}")

    # Generate Plot
    out_plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 5.5))

    plt.plot(
        unique_snrs,
        curve_clean_f1,
        "o-",
        color="#1f77b4",
        linewidth=2.5,
        markersize=7,
        label="Denoised (Spectral Subtraction + Wiener Gain)",
    )
    plt.plot(
        unique_snrs,
        curve_raw_f1,
        "s--",
        color="#d62728",
        linewidth=2.0,
        markersize=6,
        label="Raw Audio (Denoising Bypassed)",
    )

    # Highlight improvement area
    plt.fill_between(
        unique_snrs,
        curve_raw_f1,
        curve_clean_f1,
        where=(np.array(curve_clean_f1) >= np.array(curve_raw_f1)),
        interpolate=True,
        color="#1f77b4",
        alpha=0.15,
        label="DSP Gain Region",
    )

    plt.title("Experiment E2: DSP Denoising Contribution vs. Acoustic SNR", fontsize=13, fontweight="bold")
    plt.xlabel("Signal-to-Noise Ratio (dB)", fontsize=11)
    plt.ylabel("Macro F1-Score", fontsize=11)
    plt.ylim(-0.05, 1.05)
    plt.xticks(unique_snrs)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="lower right", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(out_plot_path, dpi=200)
    plt.close()
    print(f"Saved E2 ablation plot to: {out_plot_path}")

    return results_payload


if __name__ == "__main__":
    run_e2_ablation()
