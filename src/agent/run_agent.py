"""Agent Simulation Runner for Sylva Cortex (Phase P5).

Runs an end-to-end simulation of the bioacoustic agent loop:
1. Loads configuration from config.yaml and trained model from models/baseline_lgbm.joblib.
2. Ingests audio clips from data/samples/ (e.g. chainsaw and gunshot clips at -5 dB SNR).
3. Executes the full DSP preprocessing, denoising, and feature extraction pipeline.
4. Updates belief state with temporal decay and spatio-temporal context multipliers.
5. Emits decisions (IGNORE, LOG, ALERT, ESCALATE) and records traces to results/traces/agent_trace.jsonl.
6. Simulates operator false-positive feedback and demonstrates proportional controller adaptation.

Reference:
- SPEC.md Section 5.2 (The agent loop)
- SPEC.md Section 5.3 (Structured trace logging)
"""

import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import soundfile as sf
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.agent.policy import Action, adapt_threshold, decide_action
from src.agent.state import BeliefState
from src.agent.tracer import AgentTracer
from src.dsp.denoise import denoise
from src.dsp.features import extract_features
from src.dsp.preprocess import preprocess


def load_config(config_path: Path = REPO_ROOT / "config.yaml") -> Dict[str, Any]:
    """Loads central configuration hyperparameters."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model(model_path: Path = REPO_ROOT / "models" / "baseline_lgbm.joblib") -> Dict[str, Any]:
    """Loads trained model bundle."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model bundle not found at {model_path}. Train baseline model first.")
    return joblib.load(str(model_path))


def process_audio_clip(
    file_path: Path,
    sample_rate: int = 16000,
) -> np.ndarray:
    """Runs full DSP pipeline (preprocessing, denoising, feature extraction) on audio."""
    audio, sr = sf.read(str(file_path))
    if audio.ndim > 1:
        audio = np.mean(audio, axis=-1)
    audio = np.asarray(audio, dtype=np.float64)

    # 1. Preprocess: DC removal, pre-emphasis, Butterworth bandpass (80 Hz - 8000 Hz)
    proc = preprocess(
        audio,
        sample_rate=sample_rate,
        dc_removal=True,
        preemphasis_coeff=0.97,
        bandpass_low_hz=80.0,
        bandpass_high_hz=8000.0,
        bandpass_order=4,
    )

    # 2. Spectral subtraction denoising
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

    # 3. Bioacoustic feature extraction
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


def run_agent_simulation(
    trace_path: Path = REPO_ROOT / "results" / "traces" / "agent_trace.jsonl",
) -> Dict[str, Any]:
    """Executes the autonomous agent simulation over sample clips.

    Returns:
        Summary statistics dictionary of the simulation run.
    """
    config = load_config()
    model_bundle = load_model()

    clf = model_bundle["model"]
    label_encoder = model_bundle["label_encoder"]
    classes = list(label_encoder.classes_)
    sample_rate = config["audio"]["sample_rate"]
    agent_cfg = config.get("agent", {})
    context_multipliers = agent_cfg.get("context_multipliers", {
        "night_core_zone": 1.30,
        "day_concession": 0.80,
        "storm": 0.90,
    })

    decay_rate = agent_cfg.get("belief_decay", 0.95)
    reinforce_window_s = agent_cfg.get("reinforce_window_s", 5.0)
    initial_threshold = agent_cfg.get("initial_threshold", 0.70)
    threshold_step = agent_cfg.get("threshold_step", 0.02)
    threshold_min = agent_cfg.get("threshold_min", 0.40)
    threshold_max = agent_cfg.get("threshold_max", 0.95)
    target_precision = agent_cfg.get("target_precision", 0.85)

    # Initialize state and tracer
    state = BeliefState()
    for c in classes:
        state.thresholds[c] = initial_threshold
        state.precision_track[c] = target_precision

    tracer = AgentTracer(trace_path)
    tracer.clear()

    initial_thresholds = dict(state.thresholds)

    # Define simulation scenario
    # Focus on chainsaw and gunshot clips at -5 dB SNR in different contexts, plus environmental baselines
    scenario_events = [
        {
            "t": 10.0,
            "file": "data/samples/mix_bird_call_snr+00db_idx00.wav",
            "context": {"time_of_day": "day", "zone": "buffer", "weather": "clear"},
            "simulate_feedback": None,
            "note": "Ambient wildlife detection during daytime buffer patrol",
        },
        {
            "t": 35.0,
            "file": "data/samples/mix_chainsaw_snr-05db_idx00.wav",
            "context": {"time_of_day": "day", "zone": "concession", "weather": "clear"},
            "simulate_feedback": None,
            "note": "Chainsaw activity near legal logging concession during work hours",
        },
        {
            "t": 60.0,
            "file": "data/samples/mix_chainsaw_snr-05db_idx01.wav",
            "context": {"time_of_day": "night", "zone": "core_zone", "weather": "clear"},
            "simulate_feedback": "false_positive",
            "note": "Chainsaw detection at night in protected core zone -> Ranger flags False Positive (wind/fallen bough)",
        },
        {
            "t": 80.0,
            "file": "data/samples/mix_chainsaw_snr-05db_idx02.wav",
            "context": {"time_of_day": "night", "zone": "core_zone", "weather": "clear"},
            "simulate_feedback": "false_positive",
            "note": "Second chainsaw alert at night -> Ranger flags second False Positive (equipment vibration)",
        },
        {
            "t": 105.0,
            "file": "data/samples/mix_gunshot_snr-05db_idx00.wav",
            "context": {"time_of_day": "night", "zone": "core_zone", "weather": "clear"},
            "simulate_feedback": "true_positive",
            "note": "Impulsive acoustic event: Gunshot in protected core zone -> Ranger confirms True Positive poaching attempt",
        },
        {
            "t": 110.0,
            "file": "data/samples/mix_gunshot_snr-05db_idx01.wav",
            "context": {"time_of_day": "night", "zone": "core_zone", "weather": "clear"},
            "simulate_feedback": None,
            "note": "Rapid second gunshot within 5s reinforcement window -> Temporal reinforcement into ESCALATE",
        },
        {
            "t": 140.0,
            "file": "data/samples/mix_chainsaw_snr-05db_idx03.wav",
            "context": {"time_of_day": "night", "zone": "core_zone", "weather": "storm"},
            "simulate_feedback": None,
            "note": "Chainsaw sound during storm after threshold has adapted upwards",
        },
        {
            "t": 190.0,
            "file": "data/samples/sample_pure_forest_ambience.wav",
            "context": {"time_of_day": "night", "zone": "core_zone", "weather": "clear"},
            "simulate_feedback": None,
            "note": "Calm ambient recording demonstrating belief decay over silent intervals",
        },
    ]

    print("=" * 80)
    print("SYLVA CORTEX — AGENT SIMULATION & ADAPTATION RUNNER (PHASE P5)")
    print("=" * 80)
    print(f"Loaded {len(classes)} classes from classifier bundle.")
    print(f"Context Multipliers: {context_multipliers}")
    print(f"Initial Threshold: {initial_threshold} | Step: {threshold_step} | Bounds: [{threshold_min}, {threshold_max}]")
    print(f"Target Precision:  {target_precision} | Belief Decay: {decay_rate}")
    print("-" * 80)

    trace_records = []
    adaptation_log = []

    for event_id, item in enumerate(scenario_events, start=1):
        t = item["t"]
        file_rel = item["file"]
        file_path = REPO_ROOT / file_rel
        context = item["context"]
        feedback = item["simulate_feedback"]
        scenario_note = item["note"]

        # 1. Update State Context
        state.context = dict(context)

        # 2. Ingest Audio and Run DSP Pipeline
        if not file_path.exists():
            print(f"[WARN] File not found: {file_path}, skipping.")
            continue

        feat_vec = process_audio_clip(file_path, sample_rate=sample_rate)

        # 3. Model Inference (Probabilities)
        probs = clf.predict_proba([feat_vec])[0]
        obs_dict = {cls_name: float(probs[i]) for i, cls_name in enumerate(classes)}
        top_idx = int(np.argmax(probs))
        top_class = classes[top_idx]
        top_p = float(probs[top_idx])

        # 4. Agent State: Update Belief (Bayesian update + temporal decay)
        state.update_belief(
            observation=obs_dict,
            timestamp=t,
            reinforce_window_s=reinforce_window_s,
            decay_rate=decay_rate,
        )

        # 5. Agent Policy: Contextualise & Decide Action
        action, thresh_at_decision, reason = decide_action(
            state=state,
            observed_class=top_class,
            observed_prob=top_p,
            context_multipliers=context_multipliers,
        )

        # 6. Record Event in State Memory
        state.record_event({
            "t": t,
            "event_id": event_id,
            "class": top_class,
            "p": top_p,
            "action": action.value,
        })

        # 7. Structured Trace Logging
        trace_record = tracer.log_decision(
            t=t,
            event_id=event_id,
            observed_class=top_class,
            observed_p=top_p,
            belief=state.belief,
            context_note=reason.split("; ")[1] if "; " in reason else state.context.get("zone", "buffer"),
            threshold_at_decision=thresh_at_decision,
            action=action.value,
            reason=reason,
            extra={"scenario_note": scenario_note, "audio_clip": file_rel},
        )
        trace_records.append(trace_record)

        print(f"\n[Event {event_id:02d} | t={t:5.1f}s] Clip: {Path(file_rel).name}")
        print(f"  Context:   {context['time_of_day']} in {context['zone']} (Weather: {context['weather']})")
        print(f"  Observed:  {top_class} (p={top_p:.4f})")
        print(f"  Belief:    {top_class} = {state.belief.get(top_class, 0.0):.4f}")
        print(f"  Threshold: {thresh_at_decision:.4f} (Base: {state.thresholds.get(top_class, 0.70):.4f})")
        print(f"  Decision:  ==> [{action.value}] <==")
        print(f"  Reason:    {reason}")

        # 8. Simulate Operator Feedback and Adapt Threshold
        if feedback:
            adapt_res = adapt_threshold(
                state=state,
                class_name=top_class,
                feedback=feedback,
                threshold_step=threshold_step,
                threshold_min=threshold_min,
                threshold_max=threshold_max,
                target_precision=target_precision,
            )
            adaptation_log.append(adapt_res)
            print(f"  >> Operator Feedback: [{feedback.upper()}] <<")
            print(
                f"  >> Adaptation ({adapt_res['adaptation_type']}): "
                f"Threshold for '{top_class}' adjusted from {adapt_res['old_threshold']:.4f} "
                f"-> {adapt_res['new_threshold']:.4f} (Delta: {adapt_res['delta']:+.4f}) | "
                f"Precision: {adapt_res['running_precision']:.2f}"
            )

    print("\n" + "=" * 80)
    print("SIMULATION SUMMARY & THRESHOLD ADAPTATION RESULTS")
    print("=" * 80)
    print(f"Total Events Processed: {len(trace_records)}")
    print(f"Trace File Generated:   {trace_path}")
    print("\nThreshold Evolution:")
    print(f"{'Class':<18} | {'Initial Threshold':<18} | {'Final Adapted':<18} | {'Delta':<10}")
    print("-" * 72)
    for c in classes:
        init_t = initial_thresholds.get(c, 0.70)
        final_t = state.thresholds.get(c, 0.70)
        delta_t = final_t - init_t
        print(f"{c:<18} | {init_t:<18.4f} | {final_t:<18.4f} | {delta_t:+10.4f}")

    return {
        "total_events": len(trace_records),
        "initial_thresholds": initial_thresholds,
        "final_thresholds": state.thresholds,
        "adaptation_log": adaptation_log,
        "trace_file": str(trace_path),
    }


if __name__ == "__main__":
    run_agent_simulation()

