"""Agent Decision Policy and Adaptation Logic for Sylva Cortex (Phase P5 & P7).

Implements:
1. Context-aware threshold modulation (using context_multipliers from config.yaml).
2. Four-tier decision mapping: IGNORE, LOG, ALERT, ESCALATE.
3. Physical Action Execution layer:
   - ESCALATE & chainsaw: "Dispatched Drone Node 4 to Sector 7G."
   - ESCALATE & gunshot: "Alerted Local Ranger Station & Authorities."
   - ALERT: "Logged alert and pinged on-duty ranger."
   - LOG: "No physical action required."
4. Proportional controller adaptation responding to operator feedback (FP/TP).

Reference:
- SPEC.md Section 5.2 (The loop: DECIDE, ACT, ADAPT)
- config.yaml: agent hyperparameters
"""

from enum import Enum
from typing import Any, Dict, Optional, Set, Tuple, Union

import numpy as np

from src.agent.state import BeliefState


class Action(str, Enum):
    """The four action tiers available to the autonomous agent."""
    IGNORE = "IGNORE"
    LOG = "LOG"
    ALERT = "ALERT"
    ESCALATE = "ESCALATE"


DEFAULT_THREAT_CLASSES: Set[str] = {
    "chainsaw",
    "gunshot",
    "vehicle",
}


def execute_action(
    class_name: str,
    context: Optional[Dict[str, Any]] = None,
    action: Union[Action, str] = Action.LOG,
) -> str:
    """Executes the autonomous physical response corresponding to decision tier and threat.

    Response Rules:
    - ESCALATE & chainsaw: "Dispatched Drone Node 4 to Sector 7G."
    - ESCALATE & gunshot: "Alerted Local Ranger Station & Authorities."
    - ESCALATE & other: "Dispatched Fast-Response Interceptor Unit to Sector 4B."
    - ALERT: "Logged alert and pinged on-duty ranger."
    - LOG: "No physical action required."
    - IGNORE: "Signal suppressed (below noise gate)."

    Args:
        class_name: Detected bioacoustic class label.
        context: Operational context dictionary.
        action: Selected action tier.

    Returns:
        Action taken string describing autonomous physical deployment.
    """
    act_str = action.value if isinstance(action, Action) else str(action).upper()
    c_lower = str(class_name).lower()

    if act_str == "ESCALATE":
        if "chainsaw" in c_lower:
            return "Dispatched Drone Node 4 to Sector 7G."
        elif "gunshot" in c_lower:
            return "Alerted Local Ranger Station & Authorities."
        else:
            return "Dispatched Fast-Response Interceptor Unit to Sector 4B."
    elif act_str == "ALERT":
        return "Logged alert and pinged on-duty ranger."
    elif act_str == "LOG":
        return "No physical action required."
    else:
        return "Signal suppressed (below noise gate)."


def resolve_context_multiplier(
    context: Dict[str, Any],
    context_multipliers: Optional[Dict[str, float]] = None,
) -> Tuple[float, str]:
    """Resolves operational context into a threat risk multiplier and explanation note.

    Context Multipliers (from config.yaml):
    - night_core_zone (1.30): Nighttime in protected core zone -> lowers effective threshold (higher sensitivity).
    - day_concession (0.80): Daytime in logging concession -> raises effective threshold (accommodates legal forestry).
    - storm (0.90): Stormy weather raises false-positive risk from acoustic clutter -> raises effective threshold.

    Args:
        context: Operational context dictionary (time_of_day, zone, weather, battery).
        context_multipliers: Optional dictionary from config.yaml.

    Returns:
        Tuple of (effective_risk_multiplier, context_explanation_note).
    """
    if context_multipliers is None:
        context_multipliers = {
            "night_core_zone": 1.30,
            "day_concession": 0.80,
            "storm": 0.90,
        }

    time_of_day = str(context.get("time_of_day", "day")).lower()
    zone = str(context.get("zone", "buffer")).lower()
    weather = str(context.get("weather", "clear")).lower()

    multiplier = 1.0
    notes = []

    # 1. Spatio-temporal risk modulation
    if time_of_day == "night" and "core" in zone:
        m = context_multipliers.get("night_core_zone", 1.30)
        multiplier *= m
        notes.append(f"night + core zone multiplier {m:.2f}")
    elif time_of_day == "day" and "concession" in zone:
        m = context_multipliers.get("day_concession", 0.80)
        multiplier *= m
        notes.append(f"day + concession multiplier {m:.2f}")
    else:
        notes.append(f"{time_of_day}, {zone}")

    # 2. Weather modulation (acoustic masking risk)
    if weather == "storm":
        m = context_multipliers.get("storm", 0.90)
        multiplier *= m
        notes.append(f"storm clutter factor {m:.2f}")

    note_str = "; ".join(notes)
    return float(multiplier), note_str


class DecisionResult(tuple):
    """3-tuple subclass containing (action, threshold_at_decision, reason) with action_taken attribute."""
    def __new__(cls, action: Action, threshold: float, reason: str, action_taken: str = "No physical action required."):
        return super().__new__(cls, (action, threshold, reason))

    def __init__(self, action: Action, threshold: float, reason: str, action_taken: str = "No physical action required."):
        self.action = action
        self.threshold = threshold
        self.reason = reason
        self.action_taken = action_taken


def decide_action(
    state: BeliefState,
    observed_class: str,
    observed_prob: float,
    context_multipliers: Optional[Dict[str, float]] = None,
    threat_classes: Optional[Set[str]] = None,
    return_action_taken: bool = False,
) -> Union[DecisionResult, Tuple[Action, float, str, str]]:
    """Selects an action from {IGNORE, LOG, ALERT, ESCALATE} given belief and context.

    Decision Rule:
    1. Base threshold theta_base = state.thresholds[class].
    2. Effective threshold theta_eff = theta_base / context_multiplier.
       (In high-risk context M=1.30, theta_eff drops from 0.70 to ~0.538).
    3. Action Mapping:
       - Non-threat classes: LOG if p >= 0.25, else IGNORE.
       - Threat classes:
         * posterior < 0.35 * theta_eff -> IGNORE (sub-noise gate)
         * posterior < theta_eff -> LOG (suspicious but unconfirmed)
         * posterior >= theta_eff:
           - If gunshot, or posterior >= 0.85 in high-risk zone, or repeated recent alerts -> ESCALATE
           - Otherwise -> ALERT
    4. Action Execution:
       - Maps action and threat class to physical deployment (e.g., Drone Node 4 dispatch).

    Args:
        state: Current BeliefState.
        observed_class: Top bioacoustic class predicted by classifier.
        observed_prob: Classifier confidence probability for observed_class.
        context_multipliers: Optional dictionary from config.yaml.
        threat_classes: Optional set of classes treated as threats.
        return_action_taken: If True, returns 4-tuple including action_taken string.

    Returns:
        DecisionResult (3-tuple unpackable with .action_taken) or 4-tuple.
    """
    if threat_classes is None:
        threat_classes = DEFAULT_THREAT_CLASSES

    # 1. Determine situational context multiplier
    multiplier, context_note = resolve_context_multiplier(state.context, context_multipliers)

    # 2. Compute effective decision threshold
    base_thresh = state.thresholds.get(observed_class, 0.70)
    effective_thresh = float(np.clip(base_thresh / multiplier, 0.40, 0.95))

    # Current posterior belief (after Bayesian/decay updates)
    posterior = state.belief.get(observed_class, observed_prob)

    # Case A: Benign environmental or wildlife classes (bird, rain, wind, forest_ambience)
    if observed_class not in threat_classes:
        if posterior >= 0.25:
            action = Action.LOG
            reason = f"benign bioacoustic signal logged ({observed_class}, p={posterior:.2f})"
        else:
            action = Action.IGNORE
            reason = f"sub-threshold environmental signal ({observed_class}, p={posterior:.2f})"
        # Reset escalation level to 0 (Calm) when benign non-threat signals are detected
        state.escalation_level = 0
        action_taken = execute_action(observed_class, state.context, action)
        if return_action_taken:
            return action, round(effective_thresh, 4), reason, action_taken
        return DecisionResult(action, round(effective_thresh, 4), reason, action_taken)

    # Case B: Threat classes (chainsaw, gunshot, vehicle)
    noise_gate = 0.35 * effective_thresh
    if posterior < noise_gate:
        action = Action.IGNORE
        reason = f"posterior {posterior:.2f} below acoustic noise gate ({noise_gate:.2f})"
        # Signals below acoustic noise gate de-escalate towards calm
        state.escalation_level = max(0, state.escalation_level - 1)
    elif posterior < effective_thresh:
        action = Action.LOG
        reason = (
            f"suspicious acoustic signature detected, below adaptive alert threshold "
            f"({posterior:.2f} < {effective_thresh:.2f}; {context_note})"
        )
    else:
        # Alert threshold met. Evaluate escalation criteria:
        is_gunshot = (observed_class == "gunshot")
        high_risk_burst = (posterior >= 0.85 and multiplier >= 1.20)
        recent_threats = state.get_recent_events(window_s=30.0, class_name=observed_class)
        repeated_threat = len(recent_threats) >= 1
        escalated_state = (state.escalation_level >= 2)

        if is_gunshot or high_risk_burst or repeated_threat or escalated_state:
            action = Action.ESCALATE
            escalate_reasons = []
            if is_gunshot:
                escalate_reasons.append("gunshot discharge detected")
            if high_risk_burst:
                escalate_reasons.append(f"high posterior {posterior:.2f} in protected zone")
            if repeated_threat:
                escalate_reasons.append(f"repeated detection within 30s ({len(recent_threats)+1} events)")
            if escalated_state:
                escalate_reasons.append(f"escalation level {state.escalation_level}")

            reason = (
                f"posterior {posterior:.2f} exceeded adaptive threshold {effective_thresh:.2f}; "
                f"{context_note}; " + ", ".join(escalate_reasons)
            )
            state.escalation_level = min(3, state.escalation_level + 1)
        else:
            action = Action.ALERT
            reason = (
                f"posterior {posterior:.2f} exceeded adaptive threshold {effective_thresh:.2f}; "
                f"{context_note}"
            )
            state.escalation_level = max(1, state.escalation_level)

    action_taken = execute_action(observed_class, state.context, action)
    if return_action_taken:
        return action, round(effective_thresh, 4), reason, action_taken
    return DecisionResult(action, round(effective_thresh, 4), reason, action_taken)


def adapt_threshold(
    state: BeliefState,
    class_name: str,
    feedback: str,
    threshold_step: float = 0.02,
    threshold_min: float = 0.40,
    threshold_max: float = 0.95,
    target_precision: float = 0.85,
) -> Dict[str, Any]:
    """Proportional controller adapting decision thresholds based on operator feedback."""
    feedback_norm = feedback.strip().lower()
    old_threshold = state.thresholds.get(class_name, 0.70)

    if class_name not in state.feedback_history:
        state.feedback_history[class_name] = {"tp": 0, "fp": 0}

    is_fp = feedback_norm in {"false_positive", "fp", "dismiss", "reject"}
    is_tp = feedback_norm in {"true_positive", "tp", "confirm", "accept"}

    if is_fp:
        state.feedback_history[class_name]["fp"] += 1
    elif is_tp:
        state.feedback_history[class_name]["tp"] += 1

    tp = state.feedback_history[class_name]["tp"]
    fp = state.feedback_history[class_name]["fp"]
    total = tp + fp
    running_precision = tp / total if total > 0 else target_precision
    state.precision_track[class_name] = round(running_precision, 4)

    if is_fp:
        precision_error = max(0.0, target_precision - running_precision)
        kp_gain = 1.0 + 2.0 * precision_error
        step = threshold_step * kp_gain
        new_threshold = min(old_threshold + step, threshold_max)
        adaptation_type = "raised_threshold_on_fp"
    elif is_tp:
        if running_precision >= target_precision:
            step = 0.5 * threshold_step
            new_threshold = max(old_threshold - step, threshold_min)
            adaptation_type = "lowered_threshold_on_tp"
        else:
            new_threshold = old_threshold
            adaptation_type = "threshold_unchanged_below_target_precision"
    else:
        new_threshold = old_threshold
        adaptation_type = "unknown_feedback"

    new_threshold = round(float(new_threshold), 4)
    state.thresholds[class_name] = new_threshold

    return {
        "class": class_name,
        "feedback": feedback_norm,
        "adaptation_type": adaptation_type,
        "old_threshold": round(old_threshold, 4),
        "new_threshold": new_threshold,
        "delta": round(new_threshold - old_threshold, 4),
        "running_precision": round(running_precision, 4),
        "target_precision": target_precision,
        "tp_count": tp,
        "fp_count": fp,
    }
