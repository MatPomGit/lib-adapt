"""Test deterministic ARC-ADHD deficit-based recovery simulation."""

from __future__ import annotations

import copy
import unittest

from scripts.simulate_recovery_metric import (
    DEFAULT_CONFIG,
    DEFAULT_FEATURE_CONFIG,
    ConfigurationError,
    load_config,
    recovery_score,
    simulate_scenario,
    validate_config,
)


class RecoveryMetricSimulationTests(unittest.TestCase):
    """Verify synthetic recovery-score and censoring behavior."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config(DEFAULT_CONFIG)
        cls.feature_config = load_config(DEFAULT_FEATURE_CONFIG)
        validate_config(cls.config, cls.feature_config)
        cls.model = next(
            item
            for item in cls.feature_config["candidate_models"]
            if item["model_id"] == cls.config["synthetic_model_id"]
        )

    def test_zero_deficits_produce_zero_recovery_score(self) -> None:
        deficits = {feature: 0.0 for feature in self.model["features"]}
        self.assertAlmostEqual(
            recovery_score(deficits, self.model),
            0.0,
            places=10,
        )

    def test_positive_deficit_never_produces_positive_q(self) -> None:
        deficits = {feature: 0.5 for feature in self.model["features"]}
        self.assertLess(recovery_score(deficits, self.model), 0.0)

    def test_immediate_scenario_recovers(self) -> None:
        result = simulate_scenario(
            self.config,
            self.feature_config,
            "immediate",
            seed=11,
        )
        self.assertTrue(result["recovery_observed"])
        self.assertFalse(result["right_censored"])
        self.assertIsNotNone(result["recovery_latency_ms"])

    def test_delayed_scenario_recovers_later_than_immediate(self) -> None:
        immediate = simulate_scenario(
            self.config,
            self.feature_config,
            "immediate",
            seed=21,
        )
        delayed = simulate_scenario(
            self.config,
            self.feature_config,
            "delayed",
            seed=21,
        )
        self.assertTrue(delayed["recovery_observed"])
        self.assertGreater(
            delayed["recovery_latency_ms"],
            immediate["recovery_latency_ms"],
        )

    def test_censored_scenario_does_not_recover(self) -> None:
        result = simulate_scenario(
            self.config,
            self.feature_config,
            "censored",
            seed=31,
        )
        self.assertFalse(result["recovery_observed"])
        self.assertTrue(result["right_censored"])
        self.assertIsNone(result["recovery_latency_ms"])

    def test_simulation_is_deterministic_for_fixed_seed(self) -> None:
        first = simulate_scenario(
            self.config,
            self.feature_config,
            "delayed",
            seed=41,
        )
        second = simulate_scenario(
            self.config,
            self.feature_config,
            "delayed",
            seed=41,
        )
        self.assertEqual(first, second)

    def test_total_return_time_decomposes_exactly(self) -> None:
        result = simulate_scenario(
            self.config,
            self.feature_config,
            "delayed",
            seed=51,
        )
        self.assertEqual(
            result["total_return_latency_ms"],
            result["pre_monitoring_latency_ms"]
            + result["recovery_latency_ms"],
        )

    def test_non_monitoring_latency_origin_is_rejected(self) -> None:
        invalid = copy.deepcopy(self.config)
        invalid["recovery"]["latency_origin"] = "challenge_onset"
        with self.assertRaises(ConfigurationError):
            validate_config(invalid, self.feature_config)

    def test_unknown_candidate_model_is_rejected(self) -> None:
        invalid = copy.deepcopy(self.config)
        invalid["synthetic_model_id"] = "TIMING_ONLY"
        with self.assertRaises(ConfigurationError):
            validate_config(invalid, self.feature_config)


if __name__ == "__main__":
    unittest.main()
