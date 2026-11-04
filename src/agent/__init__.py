"""Sylva Cortex Agent Module.

Provides stateful bioacoustic event contextualization, adaptive decision policies,
temporal belief decay, proportional controller adaptation, and structured trace logging.
"""

from src.agent.state import BeliefState, AgentState
from src.agent.policy import (
    decide_action,
    resolve_context_multiplier,
    adapt_threshold,
    Action,
)
from src.agent.tracer import AgentTracer

__all__ = [
    "BeliefState",
    "AgentState",
    "decide_action",
    "resolve_context_multiplier",
    "adapt_threshold",
    "Action",
    "AgentTracer",
]

