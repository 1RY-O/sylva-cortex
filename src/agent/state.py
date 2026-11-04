"""Agent State and Belief Representation for Sylva Cortex (Phase P5).

Defines the BeliefState (and AgentState) dataclass managing class posteriors,
temporal decay, adaptive per-class decision thresholds, rolling event memory,
and situational context.

Reference:
- SPEC.md Section 5.1 (State definition)
- SPEC.md Section 5.2 (The loop: OBSERVE -> UPDATE BELIEF -> CONTEXTUALISE -> DECIDE -> ACT -> ADAPT)
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


DEFAULT_CLASSES: List[str] = [
    "bird_call",
    "mammal_call",
    "insect",
    "rain",
    "wind",
    "chainsaw",
    "gunshot",
    "vehicle",
    "human_speech",
    "unknown",
]


@dataclass
class BeliefState:
    """Stateful agent representation tracking bioacoustic beliefs and operational context.
    
    Attributes:
        belief: Mapping of class name to current posterior probability (decays over time).
        thresholds: Mapping of class name to current adaptive decision threshold.
        event_memory: Rolling window of recent bioacoustic detections and decisions.
        precision_track: Running precision metric per class used by the adaptation controller.
        escalation_level: Escalation indicator from 0 (calm) to 3 (emergency).
        context: Operational metadata (time_of_day, zone, weather, battery level).
        feedback_history: Running true-positive (tp) and false-positive (fp) feedback counts.
        current_time: Simulated wall-clock timestamp in seconds.
    """

    belief: Dict[str, float] = field(default_factory=dict)
    thresholds: Dict[str, float] = field(default_factory=dict)
    event_memory: deque = field(default_factory=lambda: deque(maxlen=100))
    precision_track: Dict[str, float] = field(default_factory=dict)
    escalation_level: int = 0
    context: Dict[str, Any] = field(default_factory=dict)
    feedback_history: Dict[str, Dict[str, int]] = field(default_factory=dict)
    current_time: float = 0.0
    classes: Optional[List[str]] = None

    def __post_init__(self) -> None:
        """Initialize missing state dictionaries with default values if empty."""
        class_list = self.classes or (list(self.belief.keys()) if self.belief else DEFAULT_CLASSES)

        for c in class_list:
            if c not in self.belief:
                self.belief[c] = 0.0
            if c not in self.thresholds:
                self.thresholds[c] = 0.70
            if c not in self.precision_track:
                self.precision_track[c] = 0.85
            if c not in self.feedback_history:
                self.feedback_history[c] = {"tp": 0, "fp": 0}

        if not self.context:
            self.context = {
                "time_of_day": "day",
                "zone": "buffer",
                "weather": "clear",
                "battery": 1.0,
            }

    def decay(self, decay_rate: float = 0.95) -> Dict[str, float]:
        """Applies multiplicative temporal decay to all class beliefs for this tick.

        Belief at tick t is computed as:
            B_t(c) = B_{t-1}(c) * decay_rate

        Args:
            decay_rate: Decay factor in (0, 1], default 0.95 from config.yaml.

        Returns:
            Updated belief dictionary.
        """
        for c in self.belief:
            self.belief[c] = float(np_round := round(self.belief[c] * decay_rate, 6))
        return self.belief

    def update_belief(
        self,
        observation: Dict[str, float],
        timestamp: float,
        reinforce_window_s: float = 5.0,
        decay_rate: float = 0.95,
    ) -> Dict[str, float]:
        """Updates class beliefs given new classifier observation and temporal proximity.

        If a detection of the same class occurred within reinforce_window_s,
        evidence is reinforced using independent evidence fusion:
            B_t(c) = 1.0 - (1.0 - B_{t-1}(c)) * (1.0 - P_obs(c))
        Otherwise, an exponential belief update is applied:
            B_t(c) = alpha * P_obs(c) + (1 - alpha) * (B_{t-1}(c) * decay_rate)

        Unobserved classes are decayed by decay_rate.

        Args:
            observation: Dictionary mapping class name to observed classifier probability.
            timestamp: Event arrival timestamp in seconds.
            reinforce_window_s: Time window in seconds within which detections reinforce each other.
            decay_rate: Multiplicative temporal decay factor.

        Returns:
            Updated belief dictionary.
        """
        # Auto-register any new classes present in the observation
        for c in observation:
            if c not in self.belief:
                self.belief[c] = 0.0
                if c not in self.thresholds:
                    self.thresholds[c] = 0.70
                if c not in self.precision_track:
                    self.precision_track[c] = 0.85
                if c not in self.feedback_history:
                    self.feedback_history[c] = {"tp": 0, "fp": 0}

        # Calculate temporal decay factor based on elapsed time between events
        if self.current_time > 0.0 and timestamp > self.current_time:
            dt = timestamp - self.current_time
            steps = max(1, int(round(dt / 2.0)))
            step_decay = float(decay_rate ** steps)
        else:
            step_decay = float(decay_rate)

        self.current_time = timestamp

        # Check for recent detections within the temporal reinforcement window
        recent_classes = set()
        for ev in self.event_memory:
            if timestamp - ev.get("t", 0.0) <= reinforce_window_s:
                recent_classes.add(ev.get("class"))

        for c in list(self.belief.keys()):
            obs_p = observation.get(c, 0.0)
            prior_b = self.belief.get(c, 0.0)

            if obs_p > 0.05:
                if c in recent_classes:
                    # Temporal proximity reinforcement (probabilistic union / independent evidence)
                    new_b = 1.0 - (1.0 - prior_b) * (1.0 - obs_p)
                else:
                    # Exponential mixture update combining prior and observation
                    alpha = 0.70
                    new_b = alpha * obs_p + (1.0 - alpha) * (prior_b * step_decay)
            else:
                # Decaying prior in absence of evidence
                new_b = prior_b * step_decay

            self.belief[c] = round(float(min(max(new_b, 0.0), 1.0)), 6)

        return self.belief

    def record_event(self, event: Dict[str, Any]) -> None:
        """Records an event into rolling memory and advances simulation clock.

        Args:
            event: Event metadata containing 't', 'event_id', 'class', 'p', 'action', etc.
        """
        if "t" in event:
            self.current_time = float(event["t"])
        self.event_memory.append(dict(event))

    def get_recent_events(
        self,
        window_s: float = 30.0,
        class_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieves rolling window detections within the specified duration.

        Args:
            window_s: Time horizon looking back from current_time in seconds.
            class_name: Optional filter for a specific bioacoustic class.

        Returns:
            List of matching events recorded within the window.
        """
        results = []
        for ev in self.event_memory:
            dt = self.current_time - ev.get("t", 0.0)
            if 0.0 <= dt <= window_s:
                if class_name is None or ev.get("class") == class_name:
                    results.append(ev)
        return results

    def to_dict(self) -> Dict[str, Any]:
        """Serializes agent state for logging and UI inspection."""
        return {
            "current_time": round(self.current_time, 2),
            "escalation_level": self.escalation_level,
            "belief": {k: round(v, 4) for k, v in self.belief.items() if v > 0.01},
            "thresholds": {k: round(v, 4) for k, v in self.thresholds.items()},
            "precision_track": {k: round(v, 4) for k, v in self.precision_track.items()},
            "context": dict(self.context),
            "memory_size": len(self.event_memory),
        }


# Alias AgentState to BeliefState per SPEC.md section 5.1
AgentState = BeliefState
