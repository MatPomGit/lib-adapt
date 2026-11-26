#!/usr/bin/env python3
"""Replay deterministic ARC-ADHD reference-controller decisions.

This module validates implementation and audit behavior only. The reference
rules are not a frozen intervention policy unless a future study explicitly
versions and freezes them.

Examples:
    python scripts/replay_controller.py examples/controller_replay_cases.json
    python scripts/replay_controller.py examples/controller_replay_cases.json --strict
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RULES = ROOT / "config" / "controller_rules.example.json"


class ControllerReplayError(ValueError):
    """Report an invalid controller configuration or replay case."""


def load_json(path: Path) -> Any:
    """Load UTF-8 JSON from a path."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ControllerReplayError(
            f"{path}: invalid JSON at line {exc.lineno}, column {exc.colno}"
        ) from exc


def _validate_level(
    name: str,
    value: Any,
    limits: dict[str, Any],
) -> int:
    """Validate one integer controller level against configured limits."""
    if not isinstance(value, int) or isinstance(value, bool):
        raise ControllerReplayError(f"{name} must be an integer.")

    minimum = limits.get("minimum")
    maximum = limits.get("maximum")
    if not isinstance(minimum, int) or not isinstance(maximum, int):
        raise ControllerReplayError(
            f"{name} limits must contain integer minimum and maximum."
        )
    if minimum > maximum:
        raise ControllerReplayError(
            f"{name} minimum cannot exceed maximum."
        )
    if not minimum <= value <= maximum:
        raise ControllerReplayError(
            f"{name}={value} is outside [{minimum}, {maximum}]."
        )
    return value


def validate_rules(rules: dict[str, Any]) -> None:
    """Validate the deterministic reference-controller configuration."""
    priority = rules.get("priority")
    actions = rules.get("actions")
    if not isinstance(priority, list) or not priority:
        raise ControllerReplayError("priority must be a non-empty list.")
    if not isinstance(actions, dict):
        raise ControllerReplayError("actions must be an object.")

    required_priority = [
        "safety_override",
        "child_stop",
        "child_pause",
        "sensory_overload",
        "performance_below_with_sensory_load",
        "performance_below",
        "recovery_not_observed",
        "performance_above",
        "hold",
    ]
    if priority != required_priority:
        raise ControllerReplayError(
            "Reference priority order differs from the audited specification."
        )

    allowed_actions = {
        "safe_stop",
        "pause",
        "decrease_sensory",
        "decrease_cognitive",
        "increase_cognitive",
        "hold",
    }
    for reason in priority:
        if reason not in actions:
            raise ControllerReplayError(
                f"Missing action mapping for reason {reason!r}."
            )
        if actions[reason] not in allowed_actions:
            raise ControllerReplayError(
                f"Unsupported action {actions[reason]!r} for {reason!r}."
            )

    for level_name in ("cognitive_level", "sensory_level"):
        limits = rules.get(level_name)
        if not isinstance(limits, dict):
            raise ControllerReplayError(
                f"{level_name} configuration must be an object."
            )
        step = limits.get("step")
        if not isinstance(step, int) or isinstance(step, bool) or step <= 0:
            raise ControllerReplayError(
                f"{level_name}.step must be a positive integer."
            )
        _validate_level(
            level_name,
            limits["minimum"],
            limits,
        )
        _validate_level(
            level_name,
            limits["maximum"],
            limits,
        )


def _select_reason(
    inputs: dict[str, Any],
    rules: dict[str, Any],
) -> str:
    """Select the highest-priority controller reason for one input state."""
    for field in (
        "safety_override",
        "child_stop",
        "child_pause",
        "sensory_overload",
    ):
        if not isinstance(inputs.get(field), bool):
            raise ControllerReplayError(f"{field} must be boolean.")

    performance = inputs.get("performance_state")
    if performance not in {"above", "within", "below"}:
        raise ControllerReplayError(
            "performance_state must be above, within, or below."
        )

    recovery = inputs.get("recovery_state")
    if recovery not in {"none", "stable", "not_observed"}:
        raise ControllerReplayError(
            "recovery_state must be none, stable, or not_observed."
        )

    sensory = _validate_level(
        "sensory_level",
        inputs.get("sensory_level"),
        rules["sensory_level"],
    )

    if inputs["safety_override"]:
        return "safety_override"
    if inputs["child_stop"]:
        return "child_stop"
    if inputs["child_pause"]:
        return "child_pause"
    if inputs["sensory_overload"]:
        return "sensory_overload"
    if (
        performance == "below"
        and sensory > rules["sensory_level"]["minimum"]
    ):
        return "performance_below_with_sensory_load"
    if performance == "below":
        return "performance_below"
    if recovery == "not_observed":
        return "recovery_not_observed"
    if performance == "above":
        return "performance_above"
    return "hold"


def _bounded_change(
    value: int,
    direction: int,
    limits: dict[str, Any],
) -> tuple[int, bool]:
    """Apply a bounded integer step and report whether clipping occurred."""
    step = int(limits["step"])
    proposed = value + direction * step
    bounded = max(
        int(limits["minimum"]),
        min(int(limits["maximum"]), proposed),
    )
    return bounded, bounded != proposed


def replay_decision(
    inputs: dict[str, Any],
    rules: dict[str, Any],
) -> dict[str, Any]:
    """Return the deterministic reference-controller decision."""
    validate_rules(rules)

    cognitive = _validate_level(
        "cognitive_level",
        inputs.get("cognitive_level"),
        rules["cognitive_level"],
    )
    sensory = _validate_level(
        "sensory_level",
        inputs.get("sensory_level"),
        rules["sensory_level"],
    )

    reason = _select_reason(inputs, rules)
    action = rules["actions"][reason]
    next_cognitive = cognitive
    next_sensory = sensory
    clipped = False

    if action == "decrease_sensory":
        next_sensory, clipped = _bounded_change(
            sensory,
            -1,
            rules["sensory_level"],
        )
    elif action == "decrease_cognitive":
        next_cognitive, clipped = _bounded_change(
            cognitive,
            -1,
            rules["cognitive_level"],
        )
    elif action == "increase_cognitive":
        next_cognitive, clipped = _bounded_change(
            cognitive,
            1,
            rules["cognitive_level"],
        )

    return {
        "controller_version": rules.get("controller_version"),
        "reason_code": reason,
        "proposed_action": action,
        "previous_cognitive_level": cognitive,
        "previous_sensory_level": sensory,
        "cognitive_level": next_cognitive,
        "sensory_level": next_sensory,
        "clipped": clipped,
    }


def evaluate_cases(
    cases: list[dict[str, Any]],
    rules: dict[str, Any],
) -> list[dict[str, Any]]:
    """Replay cases and return results with expected-value comparisons."""
    results: list[dict[str, Any]] = []

    for case in cases:
        if not isinstance(case, dict):
            raise ControllerReplayError("Each replay case must be an object.")

        case_id = case.get("case_id")
        inputs = case.get("input")
        expected = case.get("expected")
        if not isinstance(case_id, str) or not case_id:
            raise ControllerReplayError("Each replay case requires case_id.")
        if not isinstance(inputs, dict):
            raise ControllerReplayError(
                f"{case_id}: input must be an object."
            )
        if not isinstance(expected, dict):
            raise ControllerReplayError(
                f"{case_id}: expected must be an object."
            )

        actual = replay_decision(inputs, rules)
        compared_actual = {
            key: actual.get(key)
            for key in expected
        }
        match = compared_actual == expected

        results.append(
            {
                "case_id": case_id,
                "match": match,
                "expected": expected,
                "actual": compared_actual,
                "full_actual": actual,
            }
        )

    return results


def main() -> int:
    """Run deterministic controller replay from the command line."""
    parser = argparse.ArgumentParser(
        description="Replay ARC-ADHD reference-controller test cases."
    )
    parser.add_argument(
        "cases",
        type=Path,
        help="JSON file containing controller replay cases.",
    )
    parser.add_argument(
        "--rules",
        type=Path,
        default=DEFAULT_RULES,
        help="Reference controller-rules JSON.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero when any expected output mismatches.",
    )
    args = parser.parse_args()

    try:
        rules = load_json(args.rules)
        cases = load_json(args.cases)
        if not isinstance(rules, dict):
            raise ControllerReplayError("Rules JSON must be an object.")
        if not isinstance(cases, list):
            raise ControllerReplayError("Cases JSON must be an array.")
        results = evaluate_cases(cases, rules)
    except (OSError, ControllerReplayError) as exc:
        print(f"INVALID replay input: {exc}")
        return 1

    print(json.dumps(results, indent=2, sort_keys=True))
    mismatches = sum(not item["match"] for item in results)

    if args.strict and mismatches:
        print(f"{mismatches} replay case(s) did not match expected output.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
