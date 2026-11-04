"""Baseline Classifier Training for Sylva Cortex (Phase P4).

Trains a LightGBM gradient-boosted decision tree classifier (with sklearn
HistGradientBoostingClassifier fallback) on engineered bioacoustic DSP features.

Pipeline:
1. Ingest audio from data/manifest.csv.
2. Signal preprocessing (DC removal, pre-emphasis, Butterworth bandpass).
3. Spectral subtraction denoising.
4. Feature extraction (Log-Mel, MFCC + deltas, spectral centroid, flatness,
   rolloff, zero-crossing rate, crest factor, attack time).
5. 5-Fold cross-validation using GroupKFold to prevent clip leakage.
6. Train final model and serialize to models/baseline_lgbm.joblib.
"""

import csv
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import joblib
import numpy as np
import soundfile as sf
import yaml
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    from sklearn.ensemble import HistGradientBoostingClassifier

from src.dsp.preprocess import preprocess
from src.dsp.denoise import denoise
from src.dsp.features import extract_features

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def load_config(config_path: Path = REPO_ROOT / "config.yaml") -> Dict[str, Any]:
    """Loads central configuration hyperparameters."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def extract_clip_features(
    audio_path: Path,
    sample_rate: int = 16000,
) -> np.ndarray:
    """Runs the full DSP pipeline on an audio clip and returns its feature vector."""
    audio, sr = sf.read(str(audio_path))
    if audio.ndim > 1:
        audio = np.mean(audio, axis=-1)
    audio = np.asarray(audio, dtype=np.float64)

    # 1. Preprocess
    proc = preprocess(
        audio,
        sample_rate=sample_rate,
        dc_removal=True,
        preemphasis_coeff=0.97,
        bandpass_low_hz=80.0,
        bandpass_high_hz=8000.0,
        bandpass_order=4,
    )

    # 2. Denoise
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

    # 3. Extract features
    feats = extract_features(
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
    return feats["feature_vector"]


def load_dataset_features(
    manifest_path: Path = REPO_ROOT / "data" / "manifest.csv",
    sample_rate: int = 16000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """Loads manifest, extracts features for all audio clips, and creates CV groups."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found at {manifest_path}")

    rows: List[Dict[str, Any]] = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Loaded manifest with {len(rows)} audio records.")

    features_list: List[np.ndarray] = []
    labels_list: List[str] = []
    groups_list: List[int] = []

    for idx, row in enumerate(rows):
        rel_path = row["file_path"]
        audio_full = REPO_ROOT / rel_path
        if not audio_full.exists():
            print(f"Warning: Audio file missing: {audio_full}")
            continue

        vec = extract_clip_features(audio_full, sample_rate=sample_rate)
        features_list.append(vec)
        labels_list.append(row["label"])

        # Determine group id from file name (e.g., idx05) to avoid leakage
        match = re.search(r"idx(\d+)", rel_path)
        if match:
            group_id = int(match.group(1))
        else:
            group_id = 1000 + idx
        groups_list.append(group_id)

    X = np.array(features_list, dtype=np.float32)
    y = np.array(labels_list)
    groups = np.array(groups_list, dtype=np.int32)
    return X, y, groups, rows


def train_baseline(
    config: Dict[str, Any] | None = None,
    output_model_path: Path = REPO_ROOT / "models" / "baseline_lgbm.joblib",
) -> Dict[str, Any]:
    """Executes 5-fold cross-validation and trains the final baseline classifier."""
    if config is None:
        config = load_config()

    sr = int(config["audio"]["sample_rate"])
    random_seed = int(config["model"].get("random_state", 42))

    X, y_raw, groups, rows = load_dataset_features(sample_rate=sr)

    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    class_names = list(label_encoder.classes_)
    print(f"Classes ({len(class_names)}): {class_names}")

    # Build 5-Fold GroupKFold
    n_splits = 5
    gkf = GroupKFold(n_splits=n_splits)

    fold_accuracies: List[float] = []
    fold_f1_macro: List[float] = []
    fold_f1_weighted: List[float] = []
    oof_predictions = np.zeros(len(y), dtype=np.int32)

    print("\n--- Starting 5-Fold Cross-Validation ---")
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups), start=1):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        if HAS_LIGHTGBM:
            clf = lgb.LGBMClassifier(
                n_estimators=100,
                learning_rate=0.05,
                random_state=random_seed + fold,
                verbose=-1,
                force_col_wise=True,
            )
        else:
            clf = HistGradientBoostingClassifier(
                max_iter=100,
                random_state=random_seed + fold,
            )

        clf.fit(X_train, y_train)
        preds = clf.predict(X_val)
        oof_predictions[val_idx] = preds

        acc = float(accuracy_score(y_val, preds))
        f1_m = float(f1_score(y_val, preds, average="macro", zero_division=0))
        f1_w = float(f1_score(y_val, preds, average="weighted", zero_division=0))

        fold_accuracies.append(acc)
        fold_f1_macro.append(f1_m)
        fold_f1_weighted.append(f1_w)
        print(f"Fold {fold}: Accuracy = {acc:.4f}, Macro-F1 = {f1_m:.4f}, Weighted-F1 = {f1_w:.4f}")

    mean_acc = float(np.mean(fold_accuracies))
    std_acc = float(np.std(fold_accuracies))
    mean_f1_m = float(np.mean(fold_f1_macro))
    std_f1_m = float(np.std(fold_f1_macro))
    mean_f1_w = float(np.mean(fold_f1_weighted))

    print(f"\nCV 5-Fold Results:")
    print(f"  Mean Accuracy:   {mean_acc:.4f} (+/- {std_acc:.4f})")
    print(f"  Mean Macro F1:   {mean_f1_m:.4f} (+/- {std_f1_m:.4f})")
    print(f"  Mean Weighted F1:{mean_f1_w:.4f}")

    # Train final model on entire dataset
    print("\nTraining final model on full dataset...")
    if HAS_LIGHTGBM:
        final_clf = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            random_state=random_seed,
            verbose=-1,
            force_col_wise=True,
        )
    else:
        final_clf = HistGradientBoostingClassifier(
            max_iter=100,
            random_state=random_seed,
        )

    final_clf.fit(X, y)

    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": final_clf,
        "label_encoder": label_encoder,
        "classes": class_names,
        "feature_dim": X.shape[1],
        "sample_rate": sr,
        "oof_predictions": oof_predictions,
        "cv_results": {
            "n_splits": n_splits,
            "fold_accuracies": fold_accuracies,
            "mean_accuracy": mean_acc,
            "std_accuracy": std_acc,
            "fold_f1_macro": fold_f1_macro,
            "mean_f1_macro": mean_f1_m,
            "std_f1_macro": std_f1_m,
            "mean_f1_weighted": mean_f1_w,
        },
    }
    joblib.dump(bundle, str(output_model_path))
    print(f"Trained model saved successfully to: {output_model_path}")
    return bundle


if __name__ == "__main__":
    train_baseline()
