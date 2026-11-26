#!/usr/bin/env python3
"""Simulate the ARC-ADHD recovery score using non-negative feature deficits.

The script validates mathematical and implementation behavior only. It uses
one model from the prespecified B/S/V candidate space for synthetic trajectory
coverage. The selected synthetic model is not a scientific model-selection
decision.

Example:
    python scripts/simulate_recovery_metric.py --scenario all
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

try:
    from scripts.evaluate_recovery_feature_candidates import (
        validate_config as validate_feature_config,
    )
except ModuleNotFoundError:
    from evaluate_recovery_feature_candidates import (
        validate_config as validate_feature_config,
    )

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "phase0_recovery.example.json"
DEFAULT_FEATURE_CONFIG = (
    ROOT / "config" / "recovery_feature_candidates.phase0.json"
)

SCENARIOS = {
    "immediate": {
        "severity_deficit": 0.35,
        "recovery_rate": 0.80,
        "noise_sd": 0.01,
    },
    "delayed": {
        "severity_deficit": 2.50,
        "recovery_rate": 0.35,
        "noise_sd": 0.02,
    },
    "censored": {
        "severity_deficit": 3.00,
        "recovery_rate": 0.01,
        "noise_sd": 0.01,
    },
}


class ConfigurationError(ValueError):
    """Report an invalid synthetic recovery configuration."""


def load_config(path: Path) -> dict[str, Any]:
    """Load a UTF-8 JSON recovery configuration."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigurationError(
            f"{path}: invalid JSON at line {exc.lineno}, column {exc.colno}"
        ) from exc
    if not isinstance(data, dict):
        raise ConfigurationError("Recovery configuration must be a JSON object.")
    return data


def _model_by_id(
    feature_config: dict[str, Any],
    model_id: str,
) -> dict[str, Any]:
    for model in feature_config["candidate_models"]:
        if model["model_id"] == model_id:
            return model
    raise ConfigurationError(
        f"synthetic_model_id {model_id!r} is not in the candidate space."
    )


def validate_config(
    config: dict[str, Any],
    feature_config: dict[str, Any],
) -> None:
    """Validate the deficit-based synthetic simulation configuration."""
    try:
        validate_feature_config(feature_config)
    except ValueError as exc:
        raise ConfigurationError(str(exc)) from exc

    if config.get("config_schema") != "0.3":
        raise ConfigurationError("Synthetic recovery config_schema must be 0.3.")

    if config.get("feature_candidate_space") != (
        "config/recovery_feature_candidates.phase0.json"
    ):
        raise ConfigurationError(
            "Synthetic recovery config must reference the committed "
            "feature candidate space."
        )

    model_id = config.get("synthetic_model_id")
    if not isinstance(model_id, str):
        raise ConfigurationError("synthetic_model_id must be a string.")
    _model_by_id(feature_config, model_id)

    recovery = config.get("recovery")
    if not isinstance(recovery, dict):
        raise ConfigurationError("Missing recovery configuration object.")

    interval = config.get("observation_interval_ms")
    if not isinstance(interval, int) or isinstance(interval, bool) or interval <= 0:
        raise ConfigurationError(
            "observation_interval_ms must be a positive integer."
        )

    delta = recovery.get("delta")
    if (
        not isinstance(delta, (int, float))
        or isinstance(delta, bool)
        or delta < 0
    ):
        raise ConfigurationError("recovery.delta must be non-negative.")

    stability_n = recovery.get("stability_window_updates")
    censor_n = recovery.get("censor_limit_observations")
    if (
        not isinstance(stability_n, int)
        or isinstance(stability_n, bool)
        or stability_n <= 0
    ):
        raise ConfigurationError(
            "stability_window_updates must be a positive integer."
        )
    if (
        not isinstance(censor_n, int)
        or isinstance(censor_n, bool)
        or censor_n <= 0
    ):
        raise ConfigurationError(
            "censor_limit_observations must be a positive integer."
        )
    if stability_n > censor_n:
        raise ConfigurationError(
            "Stability window cannot exceed the censoring window."
        )

    if recovery.get("latency_origin") != "recovery_monitoring_onset":
        raise ConfigurationError(
            "Synthetic recovery latency must be monitoring-anchored."
        )
    pre_monitoring = config.get("synthetic_pre_monitoring_latency_ms")
    if (
        not isinstance(pre_monitoring, int)
        or isinstance(pre_monitoring, bool)
        or pre_monitoring < 0
    ):
        raise ConfigurationError(
            "synthetic_pre_monitoring_latency_ms must be non-negative."
        )
    if recovery.get("censor_limit_ms") != censor_n * interval:
        raise ConfigurationError(
            "censor_limit_ms must match observations times interval."
        )


def recovery_score(
    deficits: dict[str, float],
    model: dict[str, Any],
) -> float:
    """Compute q as the negative weighted composite of non-negative deficits."""
    q_value = 0.0
    for feature in model["features"]:
        if feature not in deficits:
            raise ConfigurationError(
                f"Missing synthetic deficit for feature {feature}."
            )
        deficit = deficits[feature]
        if (
            not isinstance(deficit, (int, float))
            or isinstance(deficit, bool)
            or deficit < 0
        ):
            raise ConfigurationError(
                f"Deficit for {feature} must be a non-negative number."
            )
        q_value -= float(model["weights"][feature]) * float(deficit)

    if q_value > 1e-12:
        raise ConfigurationError("Recovery score cannot exceed zero.")
    return q_value


def simulate_scenario(
    config: dict[str, Any],
    feature_config: dict[str, Any],
    scenario_name: str,
    seed: int,
) -> dict[str, Any]:
    """Simulate one deterministic synthetic challenge-recovery trajectory."""
    validate_config(config, feature_config)
    try:
        scenario = SCENARIOS[scenario_name]
    except KeyError as exc:
        raise ConfigurationError(
            f"Unknown scenario {scenario_name!r}."
        ) from exc

    recovery = config["recovery"]
    model = _model_by_id(
        feature_config,
        str(config["synthetic_model_id"]),
    )
    interval_ms = int(config["observation_interval_ms"])
    delta = float(recovery["delta"])
    stable_n = int(recovery["stability_window_updates"])
    censor_n = int(recovery["censor_limit_observations"])
    random_generator = random.Random(seed)

    trajectory: list[dict[str, Any]] = []
    stable_run = 0
    recovery_latency_ms: int | None = None

    for observation_index in range(1, censor_n + 1):
        base_deficit = float(scenario["severity_deficit"]) * math.exp(
            -float(scenario["recovery_rate"]) * (observation_index - 1)
        )
        deficits: dict[str, float] = {}

        for feature_index, feature in enumerate(model["features"]):
            feature_scale = 1.0 + 0.05 * feature_index
            noise = random_generator.gauss(
                0.0,
                float(scenario["noise_sd"]),
            )
            deficits[str(feature)] = max(
                0.0,
                base_deficit * feature_scale + noise,
            )

        score = recovery_score(deficits, model)
        above_threshold = score >= -delta
        stable_run = stable_run + 1 if above_threshold else 0

        trajectory.append(
            {
                "observation_index": observation_index,
                "time_ms": observation_index * interval_ms,
                "deficits": {
                    name: round(value, 6)
                    for name, value in deficits.items()
                },
                "q": round(score, 6),
                "above_threshold": above_threshold,
            }
        )

        if stable_run >= stable_n:
            start_index = observation_index - stable_n + 1
            recovery_latency_ms = start_index * interval_ms
            break

    observed = recovery_latency_ms is not None
    pre_monitoring_latency_ms = int(config["synthetic_pre_monitoring_latency_ms"])
    total_return_latency_ms = (
        pre_monitoring_latency_ms + recovery_latency_ms
        if recovery_latency_ms is not None
        else None
    )
    return {
        "scenario": scenario_name,
        "seed": seed,
        "definition_version": config["definition_version"],
        "synthetic_model_id": model["model_id"],
        "recovery_observed": observed,
        "pre_monitoring_latency_ms": pre_monitoring_latency_ms,
        "recovery_latency_ms": recovery_latency_ms,
        "total_return_latency_ms": total_return_latency_ms,
        "right_censored": not observed,
        "censor_limit_ms": recovery["censor_limit_ms"],
        "trajectory": trajectory,
    }


def summarize(result: dict[str, Any], include_trajectory: bool) -> dict[str, Any]:
    if include_trajectory:
        return result
    return {
        key: value
        for key, value in result.items()
        if key != "trajectory"
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Simulate ARC-ADHD deficit-based recovery behavior."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Synthetic recovery configuration JSON.",
    )
    parser.add_argument(
        "--feature-config",
        type=Path,
        default=DEFAULT_FEATURE_CONFIG,
        help="B/S/V candidate feature configuration JSON.",
    )
    parser.add_argument(
        "--scenario",
        choices=[*SCENARIOS, "all"],
        default="all",
        help="Synthetic recovery scenario.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260928,
        help="Deterministic random seed.",
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="Include per-observation deficit/q trajectories.",
    )
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        feature_config = load_config(args.feature_config)
        validate_config(config, feature_config)
        names = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
        results = [
            summarize(
                simulate_scenario(
                    config,
                    feature_config,
                    name,
                    args.seed + index,
                ),
                args.details,
            )
            for index, name in enumerate(names)
        ]
    except (OSError, ConfigurationError) as exc:
        print(f"INVALID: {exc}")
        return 1

    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
