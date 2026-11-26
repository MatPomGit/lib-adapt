"""Test the ARC-ADHD recovery feature candidate space."""

from __future__ import annotations

import copy
import unittest

from scripts.evaluate_recovery_feature_candidates import (
    DEFAULT_CONFIG,
    CandidateFeatureError,
    compute_feature_deficits,
    evaluate_case,
    load_json,
    score_models,
    validate_config,
)


class RecoveryFeatureCandidateTests(unittest.TestCase):
    """Verify the small, falsifiable pre-Phase 0 feature space."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_json(DEFAULT_CONFIG)
        validate_config(cls.config)

    def _base_case(self) -> dict:
        return {
            "case_id": "unit",
            "timing_qualified": True,
            "qualified_device_timing_floor_ms": 30,
            "prechallenge_reference": {
                "rule_id": "A",
                "go_successes": 5,
                "go_total": 6,
                "nogo_successes": 5,
                "nogo_total": 6,
            },
            "target_rule_recovery_reference": {
                "rule_id": "B",
                "go_stimulus_id": "orange",
                "go_successes": 5,
                "go_total": 6,
                "nogo_successes": 5,
                "nogo_total": 6,
                "correct_go_rt_ms": [600, 620, 640, 660],
            },
            "window": {
                "rule_id": "B",
                "go_stimulus_id": "orange",
                "go_successes": 4,
                "go_total": 4,
                "nogo_successes": 4,
                "nogo_total": 4,
                "correct_go_rt_ms": [600, 620, 640, 660],
            },
        }

    def test_candidate_space_contains_exactly_four_models(self) -> None:
        self.assertEqual(
            {item["model_id"] for item in self.config["candidate_models"]},
            {"B", "B+S", "B+V", "B+S+V"},
        )

    def test_behavior_is_mandatory_in_every_model(self) -> None:
        for model in self.config["candidate_models"]:
            self.assertIn("balanced_rule_success", model["features"])

    def test_above_reference_behavior_cannot_create_positive_q(self) -> None:
        result = evaluate_case(self.config, self._base_case())
        for model in result["models"].values():
            if model["computable"]:
                self.assertLessEqual(model["q"], 0.0)

    def test_immediate_prechallenge_performance_does_not_define_q_reference(self) -> None:
        """Keep current-rule readiness separate from target-rule q baseline."""
        first = self._base_case()
        second = self._base_case()
        second["prechallenge_reference"]["go_successes"] = 2
        second["prechallenge_reference"]["nogo_successes"] = 2

        first_result = evaluate_case(self.config, first)
        second_result = evaluate_case(self.config, second)
        self.assertEqual(first_result["deficits"], second_result["deficits"])

    def test_behavioral_drop_affects_every_candidate(self) -> None:
        case = self._base_case()
        case["window"]["go_successes"] = 2
        case["window"]["nogo_successes"] = 2
        result = evaluate_case(self.config, case)
        for model in result["models"].values():
            self.assertTrue(model["computable"])
            self.assertLess(model["q"], 0.0)

    def test_slowing_does_not_penalize_behavior_only_model(self) -> None:
        case = self._base_case()
        case["window"]["correct_go_rt_ms"] = [800, 820, 840, 860]
        result = evaluate_case(self.config, case)
        self.assertAlmostEqual(result["models"]["B"]["q"], 0.0)
        self.assertLess(result["models"]["B+S"]["q"], 0.0)
        self.assertAlmostEqual(result["models"]["B+V"]["q"], 0.0)

    def test_increased_rt_variability_targets_variability_models(self) -> None:
        case = self._base_case()
        case["window"]["correct_go_rt_ms"] = [560, 610, 650, 700]
        result = evaluate_case(self.config, case)
        self.assertAlmostEqual(result["models"]["B"]["q"], 0.0)
        self.assertAlmostEqual(result["models"]["B+S"]["q"], 0.0)
        self.assertLess(result["models"]["B+V"]["q"], 0.0)

    def test_timing_models_are_unavailable_without_qualification(self) -> None:
        case = self._base_case()
        case["timing_qualified"] = False
        result = evaluate_case(self.config, case)
        self.assertTrue(result["models"]["B"]["computable"])
        self.assertFalse(result["models"]["B+S"]["computable"])
        self.assertFalse(result["models"]["B+V"]["computable"])
        self.assertFalse(result["models"]["B+S+V"]["computable"])

    def test_behavior_reference_must_match_target_rule(self) -> None:
        """Reject current-rule behavior as the recovery-score baseline."""
        case = self._base_case()
        case["target_rule_recovery_reference"]["rule_id"] = "A"
        with self.assertRaises(CandidateFeatureError):
            evaluate_case(self.config, case)

    def test_recovery_reference_must_match_go_stimulus(self) -> None:
        """Reject a recovery baseline for a different GO stimulus identity."""
        case = self._base_case()
        case["target_rule_recovery_reference"]["go_stimulus_id"] = "blue"
        with self.assertRaises(CandidateFeatureError):
            evaluate_case(self.config, case)

    def test_positive_compensation_is_structurally_disallowed(self) -> None:
        deficits = {
            "balanced_rule_success": 2.0,
            "correct_go_rt_slowing": 0.0,
            "correct_go_rt_variability_increase": 0.0,
        }
        scores = score_models(self.config, deficits)
        self.assertLess(scores["B+S+V"]["q"], 0.0)

    def test_model_without_behavior_is_rejected(self) -> None:
        config = copy.deepcopy(self.config)
        config["candidate_models"][0]["features"] = [
            "correct_go_rt_slowing"
        ]
        config["candidate_models"][0]["weights"] = {
            "correct_go_rt_slowing": 1.0
        }
        with self.assertRaises(CandidateFeatureError):
            validate_config(config)

    def test_robustness_grid_is_frozen(self) -> None:
        """Reject silent threshold-grid expansion after Phase 0 begins."""
        config = copy.deepcopy(self.config)
        config["phase0_feature_selection"]["robustness_grid"]["delta"].append(
            1.0
        )
        with self.assertRaises(CandidateFeatureError):
            validate_config(config)

    def test_robustness_rule_uses_worst_grid_result(self) -> None:
        """Reject cherry-picking of the most favorable threshold pair."""
        config = copy.deepcopy(self.config)
        config["phase0_feature_selection"]["robustness_grid"]["rule"] = (
            "Use the best threshold combination."
        )
        with self.assertRaises(CandidateFeatureError):
            validate_config(config)


if __name__ == "__main__":
    unittest.main()
