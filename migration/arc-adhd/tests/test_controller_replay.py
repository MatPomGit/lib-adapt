"""Test deterministic ARC-ADHD reference-controller replay."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from scripts.replay_controller import (
    DEFAULT_RULES,
    evaluate_cases,
    load_json,
    replay_decision,
    validate_rules,
)

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "examples" / "controller_replay_cases.json"


class ControllerReplayTests(unittest.TestCase):
    """Verify priority, bounded adaptation, and replay determinism."""

    @classmethod
    def setUpClass(cls) -> None:
        """Load reference controller rules and replay cases."""
        cls.rules = load_json(DEFAULT_RULES)
        cls.cases = load_json(CASES)
        validate_rules(cls.rules)

    def test_all_reference_cases_match(self) -> None:
        """Match every committed expected replay output."""
        results = evaluate_cases(self.cases, self.rules)
        mismatches = [
            result["case_id"]
            for result in results
            if not result["match"]
        ]
        self.assertEqual(mismatches, [])

    def test_same_input_produces_same_decision(self) -> None:
        """Return identical decisions for identical deterministic input."""
        inputs = self.cases[4]["input"]
        first = replay_decision(inputs, self.rules)
        second = replay_decision(inputs, self.rules)
        self.assertEqual(first, second)

    def test_safety_override_beats_child_pause(self) -> None:
        """Keep safety override above lower-priority controls."""
        inputs = {
            "safety_override": True,
            "child_stop": False,
            "child_pause": True,
            "sensory_overload": True,
            "performance_state": "below",
            "recovery_state": "not_observed",
            "cognitive_level": 2,
            "sensory_level": 2,
        }
        result = replay_decision(inputs, self.rules)
        self.assertEqual(result["reason_code"], "safety_override")
        self.assertEqual(result["proposed_action"], "safe_stop")

    def test_low_performance_reduces_sensory_before_cognitive(self) -> None:
        """Preserve the sensory-first adaptation rule."""
        inputs = {
            "safety_override": False,
            "child_stop": False,
            "child_pause": False,
            "sensory_overload": False,
            "performance_state": "below",
            "recovery_state": "none",
            "cognitive_level": 3,
            "sensory_level": 2,
        }
        result = replay_decision(inputs, self.rules)
        self.assertEqual(
            result["reason_code"],
            "performance_below_with_sensory_load",
        )
        self.assertEqual(result["cognitive_level"], 3)
        self.assertEqual(result["sensory_level"], 1)

    def test_unrecovered_state_blocks_upward_adaptation(self) -> None:
        """Hold difficulty after non-recovery despite strong performance."""
        inputs = {
            "safety_override": False,
            "child_stop": False,
            "child_pause": False,
            "sensory_overload": False,
            "performance_state": "above",
            "recovery_state": "not_observed",
            "cognitive_level": 2,
            "sensory_level": 0,
        }
        result = replay_decision(inputs, self.rules)
        self.assertEqual(result["reason_code"], "recovery_not_observed")
        self.assertEqual(result["cognitive_level"], 2)

    def test_replay_case_file_is_valid_json_array(self) -> None:
        """Keep committed replay fixtures machine-readable."""
        raw = json.loads(CASES.read_text(encoding="utf-8"))
        self.assertIsInstance(raw, list)
        self.assertGreater(len(raw), 0)


if __name__ == "__main__":
    unittest.main()
