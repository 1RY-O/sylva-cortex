"""Agent Trace Logging Utility for Sylva Cortex (Phase P5 & P7).

Provides structured JSON Lines (JSONL) tracing for every decision cycle,
recording evidence observation, belief state, operational context, effective
decision thresholds, actions, explanatory rationales, and the physical action taken.

Reference:
- SPEC.md Section 5.3 (Trace format)
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.agent.policy import execute_action

DEFAULT_TRACE_PATH = Path("results/traces/agent_trace.jsonl")


class AgentTracer:
    """Structured logger recording agent decision cycles to JSONL."""

    def __init__(self, trace_file: Union[str, Path] = DEFAULT_TRACE_PATH) -> None:
        """Initializes the agent trace logger.

        Args:
            trace_file: Path to destination .jsonl file.
        """
        self.trace_file = Path(trace_file)
        self.trace_file.parent.mkdir(parents=True, exist_ok=True)
        self._records: List[Dict[str, Any]] = []

    def clear(self) -> None:
        """Truncates the trace file and clears in-memory buffer."""
        self._records.clear()
        with open(self.trace_file, "w", encoding="utf-8") as f:
            pass

    def log_decision(
        self,
        t: float,
        event_id: int,
        observed_class: str,
        observed_p: float,
        belief: Dict[str, float],
        context_note: str,
        threshold_at_decision: float,
        action: str,
        reason: str,
        action_taken: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Logs a single decision cycle conforming to SPEC.md section 5.3 + action_taken.

        Args:
            t: Simulation timestamp in seconds.
            event_id: Monotonically increasing event index.
            observed_class: Top class label emitted by classifier.
            observed_p: Classifier confidence probability for observed_class.
            belief: Dictionary of active class beliefs.
            context_note: Human-readable description of active context.
            threshold_at_decision: Effective decision threshold applied.
            action: Action selected (IGNORE, LOG, ALERT, ESCALATE).
            reason: Causal explanation for the action.
            action_taken: Physical deployment description (e.g. "Dispatched Drone Node 4").
            extra: Optional additional metadata.

        Returns:
            The structured record dictionary written to disk.
        """
        # Auto-compute action_taken if not explicitly provided
        if action_taken is None:
            action_taken = execute_action(observed_class, {}, action)

        # Filter belief to non-trivial entries for compact readability
        filtered_belief = {
            k: round(v, 4) for k, v in belief.items() if v >= 0.01
        }
        if not filtered_belief and observed_class in belief:
            filtered_belief[observed_class] = round(belief[observed_class], 4)

        record: Dict[str, Any] = {
            "t": round(float(t), 2),
            "event_id": int(event_id),
            "observed": {
                "class": str(observed_class),
                "p": round(float(observed_p), 4),
            },
            "belief": filtered_belief,
            "context_note": str(context_note),
            "threshold_at_decision": round(float(threshold_at_decision), 4),
            "action": str(action),
            "reason": str(reason),
            "action_taken": str(action_taken),
        }

        if extra:
            record["extra"] = extra

        self._records.append(record)

        # Write immediately to JSONL
        with open(self.trace_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        return record

    def get_traces(self) -> List[Dict[str, Any]]:
        """Reads all trace records stored in the trace file.

        Returns:
            List of parsed trace dictionaries.
        """
        if not self.trace_file.exists():
            return []
        records = []
        with open(self.trace_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return records
