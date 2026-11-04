"""Evaluation Harness for Experiment E1: Baseline Performance vs SNR (Phase P4).

Evaluates the trained gradient-boosted baseline classifier across controlled
acoustic SNR levels (-10 dB to +20 dB).

Produces:
1. results/metrics.json: Comprehensive evaluation metrics (accuracy, precision, recall, F1).
2. figures/e1_snr_curve.png: F1-score vs SNR degradation curve per class and macro average.
3. figures/e1_confusion_matrix.png: Full multi-class confusion matrix plot.
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from src.models.train import load_dataset_features, load_config

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def evaluate_experiment_e1(
    model_path: Path = REPO_ROOT / "models" / "baseline_lgbm.joblib",
    manifest_path: Path = REPO_ROOT / "data" / "manifest.csv",
    metrics_path: Path = REPO_ROOT / "results" / "metrics.json",
    snr_plot_path: Path = REPO_ROOT / "figures" / "e1_snr_curve.png",
    cm_plot_path: Path = REPO_ROOT / "figures" / "e1_confusion_matrix.png",
) -> Dict[str, Any]:
    """Runs Experiment E1: Evaluates classifier across SNR sweep and saves figures."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}. Run train.py first.")

    bundle = joblib.load(str(model_path))
    clf = bundle["model"]
    label_encoder = bundle["label_encoder"]
    classes = list(bundle["classes"])
    sr = int(bundle.get("sample_rate", 16000))

    # Load dataset and extract features
    X, y_raw, groups, rows = load_dataset_features(manifest_path, sample_rate=sr)
    y_true = label_encoder.transform(y_raw)

    # Use Out-Of-Fold (OOF) cross-validation predictions for true generalization across SNRs
    if "oof_predictions" in bundle and len(bundle["oof_predictions"]) == len(y_true):
        y_pred = np.array(bundle["oof_predictions"])
        eval_mode = "Out-Of-Fold Cross-Validation"
    else:
        y_pred = clf.predict(X)
        eval_mode = "Model Predictions"
    pred_labels = label_encoder.inverse_transform(y_pred)
    print(f"Evaluation Mode: {eval_mode}")

    # Overall metrics
    overall_acc = float(accuracy_score(y_true, y_pred))
    overall_f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    overall_f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    overall_prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    overall_rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    # Parse SNR levels from rows
    snr_extracted: List[float | None] = []
    for r in rows:
        match = re.search(r"snr([+-]?\d+)db", r["file_path"], re.IGNORECASE)
        if match:
            snr_extracted.append(float(match.group(1)))
        else:
            snr_extracted.append(None)

    # Group metrics by SNR level
    unique_snrs = sorted(list({s for s in snr_extracted if s is not None}))
    by_snr_metrics: Dict[str, Any] = {}
    f1_by_snr: Dict[str, List[float]] = {cls: [] for cls in classes}
    macro_f1_curve: List[float] = []

    print("\n--- Experiment E1: SNR Degradation Curve ---")
    for snr in unique_snrs:
        snr_mask = [s == snr for s in snr_extracted]
        idx_sub = np.where(snr_mask)[0]
        if len(idx_sub) == 0:
            continue

        y_true_sub = y_true[idx_sub]
        y_pred_sub = y_pred[idx_sub]

        sub_acc = float(accuracy_score(y_true_sub, y_pred_sub))
        sub_prec = float(precision_score(y_true_sub, y_pred_sub, average="macro", zero_division=0))
        sub_rec = float(recall_score(y_true_sub, y_pred_sub, average="macro", zero_division=0))
        sub_f1 = float(f1_score(y_true_sub, y_pred_sub, average="macro", zero_division=0))
        macro_f1_curve.append(sub_f1)

        # Per-class metrics at this SNR
        per_class_snr = {}
        for cls_idx, cls_name in enumerate(classes):
            bin_true = (y_true_sub == cls_idx).astype(int)
            bin_pred = (y_pred_sub == cls_idx).astype(int)
            c_f1 = float(f1_score(bin_true, bin_pred, zero_division=0))
            c_prec = float(precision_score(bin_true, bin_pred, zero_division=0))
            c_rec = float(recall_score(bin_true, bin_pred, zero_division=0))
            per_class_snr[cls_name] = {
                "f1": c_f1,
                "precision": c_prec,
                "recall": c_rec,
                "support": int(np.sum(bin_true)),
            }
            f1_by_snr[cls_name].append(c_f1)

        by_snr_metrics[str(int(snr))] = {
            "snr_db": snr,
            "sample_count": len(idx_sub),
            "accuracy": sub_acc,
            "precision_macro": sub_prec,
            "recall_macro": sub_rec,
            "f1_macro": sub_f1,
            "per_class": per_class_snr,
        }
        print(f"SNR {snr:+03.0f} dB | Acc: {sub_acc:.4f} | Prec: {sub_prec:.4f} | Rec: {sub_rec:.4f} | F1: {sub_f1:.4f}")

    # Per-class overall metrics
    class_report_dict = classification_report(y_true, y_pred, target_names=classes, output_dict=True, zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    metrics_payload = {
        "experiment": "E1_baseline_snr_sweep",
        "description": "Baseline GBDT classifier precision, recall, and F1 across SNR sweep (-10 to +20 dB)",
        "overall": {
            "total_samples": len(y_true),
            "accuracy": overall_acc,
            "f1_macro": overall_f1_macro,
            "f1_weighted": overall_f1_weighted,
            "precision_macro": overall_prec_macro,
            "recall_macro": overall_rec_macro,
        },
        "by_snr": by_snr_metrics,
        "by_class": {
            cls: {
                "precision": float(class_report_dict[cls]["precision"]),
                "recall": float(class_report_dict[cls]["recall"]),
                "f1": float(class_report_dict[cls]["f1-score"]),
                "support": int(class_report_dict[cls]["support"]),
            }
            for cls in classes
            if cls in class_report_dict
        },
        "confusion_matrix": {
            "classes": classes,
            "matrix": cm.tolist(),
        },
        "cv_5fold": bundle.get("cv_results", {}),
    }

    # Save metrics.json
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"\nSaved metrics to: {metrics_path}")

    # Plot 1: F1-score vs SNR Curve
    snr_plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 5.5))
    plt.plot(
        unique_snrs,
        macro_f1_curve,
        "o-",
        color="#1f77b4",
        linewidth=2.5,
        markersize=7,
        label="Macro F1 (Average)",
    )

    colors = ["#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]
    for idx, (cls_name, f1_vals) in enumerate(f1_by_snr.items()):
        if any(v > 0 for v in f1_vals):
            c = colors[idx % len(colors)]
            plt.plot(
                unique_snrs,
                f1_vals,
                "s--",
                color=c,
                linewidth=1.8,
                markersize=6,
                label=f"Class: {cls_name}",
            )

    plt.title("Experiment E1: Threat Detection F1-Score vs. Acoustic SNR", fontsize=13, fontweight="bold")
    plt.xlabel("Signal-to-Noise Ratio (dB)", fontsize=11)
    plt.ylabel("F1 Score", fontsize=11)
    plt.ylim(-0.05, 1.05)
    plt.xticks(unique_snrs)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="lower right", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(snr_plot_path, dpi=200)
    plt.close()
    print(f"Saved SNR curve plot to: {snr_plot_path}")

    # Plot 2: Confusion Matrix
    cm_plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        title="Experiment E1: Confusion Matrix",
        ylabel="True Label",
        xlabel="Predicted Label",
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            ax.text(
                j,
                i,
                f"{val}",
                ha="center",
                va="center",
                color="white" if val > thresh else "black",
                fontweight="bold",
            )
    fig.tight_layout()
    plt.savefig(cm_plot_path, dpi=200)
    plt.close()
    print(f"Saved confusion matrix plot to: {cm_plot_path}")

    return metrics_payload


if __name__ == "__main__":
    evaluate_experiment_e1()
