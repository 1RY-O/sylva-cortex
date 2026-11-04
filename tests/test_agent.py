"""Unit tests for the Sylva Cortex Agent Module (Phase P5).

Tests coverage:
1. BeliefState initialization, temporal decay, and rolling event memory.
2. Temporal proximity reinforcement within reinforce_window_s.
3. Context multiplier resolution and effective threshold modulation.
4. Four-tier decision mapping (IGNORE, LOG, ALERT, ESCALATE).
5. Proportional controller adaptation on operator feedback (FP raises threshold).
6. AgentTracer JSONL schema compliance (SPEC.md Section 5.3).
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.agent.policy import (
    Action,
    adapt_threshold,
    decide_action,
    resolve_context_multiplier,
)
from src.agent.state import BeliefState
from src.agent.tracer import AgentTracer


class TestBeliefState(unittest.TestCase):
    """Verifies state tracking, temporal decay, and event memory."""

    def test_initialization(self) -> None:
        state = BeliefState()
        self.assertIn("chainsaw", state.belief)
        self.assertIn("gunshot", state.belief)
        self.assertAlmostEqual(state.thresholds["chainsaw"], 0.70)
        self.assertEqual(state.escalation_level, 0)
        self.assertEqual(len(state.event_memory), 0)

    def test_temporal_decay(self) -> None:
        state = BeliefState()
        state.belief["chainsaw"] = 0.80
        state.belief["gunshot"] = 0.50

        state.decay(decay_rate=0.95)
        self.assertAlmostEqual(state.belief["chainsaw"], 0.76, places=3)
        self.assertAlmostEqual(state.belief["gunshot"], 0.475, places=3)

        # Decay over multiple ticks
        for _ in range(10):
            state.decay(decay_rate=0.95)
        self.assertLess(state.belief["chainsaw"], 0.50)

    def test_temporal_reinforcement(self) -> None:
        state = BeliefState()
        # Event 1 at t=10.0s
        state.update_belief({"chainsaw": 0.60}, timestamp=10.0, reinforce_window_s=5.0)
        state.record_event({"t": 10.0, "class": "chainsaw", "p": 0.60, "action": "ALERT"})
        b1 = state.belief["chainsaw"]

        # Event 2 arriving 2 seconds later (within 5s window) -> reinforces
        state.update_belief({"chainsaw": 0.60}, timestamp=12.0, reinforce_window_s=5.0)
        b2 = state.belief["chainsaw"]
        self.assertGreater(b2, b1, "Temporal proximity within window should reinforce belief.")


class TestContextAndPolicy(unittest.TestCase):
    """Verifies context multiplier application and decision tiers."""

    def test_context_multipliers(self) -> None:
        # 1. Night in core zone (high risk) -> multiplier 1.30
        ctx_night_core = {"time_of_day": "night", "zone": "core_zone", "weather": "clear"}
        m1, note1 = resolve_context_multiplier(ctx_night_core)
        self.assertAlmostEqual(m1, 1.30)
        self.assertIn("night + core zone", note1)

        # 2. Day in logging concession (low risk) -> multiplier 0.80
        ctx_day_concession = {"time_of_day": "day", "zone": "concession", "weather": "clear"}
        m2, note2 = resolve_context_multiplier(ctx_day_concession)
        self.assertAlmostEqual(m2, 0.80)
        self.assertIn("day + concession", note2)

        # 3. Storm clutter factor -> multiplier 0.90
        ctx_storm = {"time_of_day": "day", "zone": "buffer", "weather": "storm"}
        m3, note3 = resolve_context_multiplier(ctx_storm)
        self.assertAlmostEqual(m3, 0.90)

    def test_decision_tiers(self) -> None:
        state = BeliefState()
        state.thresholds["chainsaw"] = 0.70

        # Sub-noise gate -> IGNORE
        state.belief["chainsaw"] = 0.10
        action, thresh, reason = decide_action(state, "chainsaw", 0.10)
        self.assertEqual(action, Action.IGNORE)

        # Between noise gate and threshold -> LOG
        state.belief["chainsaw"] = 0.50
        action, thresh, reason = decide_action(state, "chainsaw", 0.50)
        self.assertEqual(action, Action.LOG)

        # High-risk context (night core zone) lowers threshold (0.70 / 1.30 = 0.538)
        state.context = {"time_of_day": "night", "zone": "core_zone", "weather": "clear"}
        state.belief["chainsaw"] = 0.60
        action, thresh, reason = decide_action(state, "chainsaw", 0.60)
        self.assertIn(action, [Action.ALERT, Action.ESCALATE])
        self.assertAlmostEqual(thresh, 0.70 / 1.30, places=2)

        # Gunshot triggers ESCALATE
        state.thresholds["gunshot"] = 0.70
        state.belief["gunshot"] = 0.75
        action, thresh, reason = decide_action(state, "gunshot", 0.75)
        self.assertEqual(action, Action.ESCALATE)


class TestAdaptationController(unittest.TestCase):
    """Verifies proportional controller adaptation on operator feedback."""

    def test_false_positive_raises_threshold(self) -> None:
        state = BeliefState()
        initial_thresh = state.thresholds["chainsaw"]  # 0.70

        # Operator flags false positive
        res = adapt_threshold(
            state,
            class_name="chainsaw",
            feedback="false_positive",
            threshold_step=0.02,
            threshold_max=0.95,
        )

        self.assertGreater(state.thresholds["chainsaw"], initial_thresh)
        self.assertEqual(res["adaptation_type"], "raised_threshold_on_fp")
        self.assertGreaterEqual(res["delta"], 0.02)
        self.assertEqual(res["fp_count"], 1)

    def test_threshold_bounds(self) -> None:
        state = BeliefState()
        state.thresholds["chainsaw"] = 0.94

        # Cannot exceed threshold_max
        adapt_threshold(
            state,
            class_name="chainsaw",
            feedback="false_positive",
            threshold_step=0.05,
            threshold_max=0.95,
        )
        self.assertLessEqual(state.thresholds["chainsaw"], 0.95)


class TestAgentTracer(unittest.TestCase):
    """Verifies JSONL trace format compliance with SPEC.md section 5.3."""

    def test_trace_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_path = Path(tmpdir) / "test_trace.jsonl"
            tracer = AgentTracer(trace_path)

            tracer.log_decision(
                t=41.2,
                event_id=17,
                observed_class="chainsaw",
                observed_p=0.81,
                belief={"chainsaw": 0.88},
                context_note="02:14, core zone",
                threshold_at_decision=0.72,
                action="ESCALATE",
                reason="posterior exceeded adaptive threshold; night + core zone multiplier 1.3",
            )

            records = tracer.get_traces()
            self.assertEqual(len(records), 1)
            rec = records[0]

            # Verify exact SPEC.md 5.3 keys
            required_keys = {
                "t",
                "event_id",
                "observed",
                "belief",
                "context_note",
                "threshold_at_decision",
                "action",
                "reason",
            }
            self.assertTrue(required_keys.issubset(rec.keys()))
            self.assertEqual(rec["observed"]["class"], "chainsaw")
            self.assertEqual(rec["action"], "ESCALATE")
            self.assertAlmostEqual(rec["threshold_at_decision"], 0.72)


if __name__ == "__main__":
    unittest.main()

