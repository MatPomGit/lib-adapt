#!/usr/bin/env python3
"""Validate and evaluate the ARC-ADHD Phase 0 recovery feature candidates.

The candidate space is intentionally small. It contains one mandatory
goal-directed behavior feature (B) and two optional timing features:
correct-GO slowing (S) and correct-GO variability increase (V).

This tool is for implementation and measurement validation only. It must not
be used to choose a feature model from clinical outcomes.

Example:
    python scripts/evaluate_recovery_feature_candidates.py \
      examples/recovery_feature_candidate_cases.json --strict
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "recovery_feature_candidates.phase0.json"
DEFAULT_CASES = ROOT / "examples" / "recovery_feature_candidate_cases.json"

FEATURES = {
    "balanced_rule_success",
    "correct_go_rt_slowing",
    "correct_go_rt_variability_increase",
}
MODEL_IDS = {"B", "B+S", "B+V", "B+S+V"}


class CandidateFeatureError(ValueError):
    """Report an invalid recovery-feature candidate specification or case."""


def load_json(path: Path) -> Any:
    """Load UTF-8 JSON."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CandidateFeatureError(
            f"{path}: invalid JSON at line {exc.lineno}, column {exc.colno}"
        ) from exc


def _fraction(value: Any, field: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not 0 <= value <= 1
    ):
        raise CandidateFeatureError(f"{field} must be in [0, 1].")
    return float(value)


def validate_config(config: dict[str, Any]) -> None:
    """Validate invariants of the deliberately small candidate space."""
    if config.get("schema_version") != "0.2":
        raise CandidateFeatureError("schema_version must be 0.2.")
    if config.get("task_id") != "signal-gate":
        raise CandidateFeatureError("task_id must be signal-gate.")

    challenge_types = config.get("eligible_challenge_types")
    if challenge_types != ["rule_switch"]:
        raise CandidateFeatureError(
            "The pre-Phase 0 candidate space must currently be restricted "
            "to the Signal Gate rule_switch challenge."
        )

    primitive = config.get("primitive_features")
    if not isinstance(primitive, dict) or set(primitive) != FEATURES:
        raise CandidateFeatureError(
            f"primitive_features must contain exactly {sorted(FEATURES)}."
        )

    if primitive["balanced_rule_success"].get("required") is not True:
        raise CandidateFeatureError("balanced_rule_success must be mandatory.")

    eligibility_reference = config.get("eligibility_reference_window")
    recovery_reference = config.get("target_rule_recovery_reference")
    post = config.get("postchallenge_window")
    if (
        not isinstance(eligibility_reference, dict)
        or not isinstance(recovery_reference, dict)
        or not isinstance(post, dict)
    ):
        raise CandidateFeatureError(
            "eligibility_reference_window, target_rule_recovery_reference "
            "and postchallenge_window must be objects."
        )

    for name, block, fields in (
        (
            "eligibility_reference_window",
            eligibility_reference,
            ("valid_trials", "minimum_go_trials", "minimum_nogo_trials"),
        ),
        (
            "target_rule_recovery_reference",
            recovery_reference,
            (
                "valid_trials",
                "minimum_go_trials",
                "minimum_nogo_trials",
                "minimum_correct_go_rt_trials",
            ),
        ),
        (
            "postchallenge_window",
            post,
            (
                "valid_trials",
                "update_stride_valid_trials",
                "minimum_go_trials",
                "minimum_nogo_trials",
                "minimum_correct_go_rt_trials_for_slowing",
                "minimum_correct_go_rt_trials_for_variability",
            ),
        ),
    ):
        for field in fields:
            value = block.get(field)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise CandidateFeatureError(
                    f"{name}.{field} must be a positive integer."
                )

    if recovery_reference.get("source") != "RULE_CALIBRATION":
        raise CandidateFeatureError(
            "target_rule_recovery_reference.source must be RULE_CALIBRATION."
        )
    if recovery_reference.get("immutable_within_session") is not True:
        raise CandidateFeatureError(
            "Target-rule recovery reference must be immutable within session."
        )
    if recovery_reference.get("require_same_rule_id_as_recovery") is not True:
        raise CandidateFeatureError(
            "Recovery reference must require the same target rule as recovery."
        )
    if (
        recovery_reference.get("require_same_go_stimulus_id_as_recovery")
        is not True
    ):
        raise CandidateFeatureError(
            "Recovery reference must require the same GO stimulus identity."
        )

    numerical = config.get("numerical_rules")
    if not isinstance(numerical, dict):
        raise CandidateFeatureError("numerical_rules must be an object.")
    for field in ("beta_prior_alpha", "beta_prior_beta"):
        value = numerical.get(field)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or value <= 0
        ):
            raise CandidateFeatureError(f"{field} must be positive.")
    if numerical.get("beta_prior_alpha") != 0.5 or numerical.get(
        "beta_prior_beta"
    ) != 0.5:
        raise CandidateFeatureError(
            "The candidate behavioral feature must use the frozen Jeffreys "
            "Beta(0.5, 0.5) smoothing rule."
        )

    models = config.get("candidate_models")
    if not isinstance(models, list) or len(models) != 4:
        raise CandidateFeatureError("candidate_models must contain four models.")
    if {model.get("model_id") for model in models} != MODEL_IDS:
        raise CandidateFeatureError(
            f"candidate model IDs must be exactly {sorted(MODEL_IDS)}."
        )

    feature_sets: set[tuple[str, ...]] = set()
    for model in models:
        features = model.get("features")
        weights = model.get("weights")
        if not isinstance(features, list) or not features:
            raise CandidateFeatureError("Every model needs a non-empty feature list.")
        if "balanced_rule_success" not in features:
            raise CandidateFeatureError(
                f"{model.get('model_id')}: behavior feature B is mandatory."
            )
        if not set(features) <= FEATURES:
            raise CandidateFeatureError(
                f"{model.get('model_id')}: unknown candidate feature."
            )
        feature_key = tuple(sorted(features))
        if feature_key in feature_sets:
            raise CandidateFeatureError("Candidate feature sets must be unique.")
        feature_sets.add(feature_key)

        if not isinstance(weights, dict) or set(weights) != set(features):
            raise CandidateFeatureError(
                f"{model.get('model_id')}: weights must match features exactly."
            )
        total = 0.0
        for feature, weight in weights.items():
            if (
                not isinstance(weight, (int, float))
                or isinstance(weight, bool)
                or weight <= 0
            ):
                raise CandidateFeatureError(
                    f"{model.get('model_id')}: weight for {feature} must be positive."
                )
            total += float(weight)
        if not math.isclose(total, 1.0, abs_tol=1e-12):
            raise CandidateFeatureError(
                f"{model.get('model_id')}: weights must sum to 1."
            )

        timing_features = set(features) - {"balanced_rule_success"}
        if timing_features:
            behavior_weight = float(weights["balanced_rule_success"])
            timing_weight = sum(float(weights[name]) for name in timing_features)
            if not math.isclose(behavior_weight, 0.5, abs_tol=1e-12):
                raise CandidateFeatureError(
                    f"{model.get('model_id')}: enriched models must reserve "
                    "0.5 total weight for behavior."
                )
            if not math.isclose(timing_weight, 0.5, abs_tol=1e-12):
                raise CandidateFeatureError(
                    f"{model.get('model_id')}: enriched models must reserve "
                    "0.5 total weight for timing."
                )

    selection = config.get("phase0_feature_selection")
    if not isinstance(selection, dict):
        raise CandidateFeatureError("phase0_feature_selection must be an object.")
    gates = selection.get("hard_gates")
    if not isinstance(gates, dict):
        raise CandidateFeatureError("hard_gates must be an object.")
    for field, value in gates.items():
        _fraction(value, f"hard_gates.{field}")

    robustness = selection.get("robustness_grid")
    if not isinstance(robustness, dict):
        raise CandidateFeatureError("robustness_grid must be an object.")
    if robustness.get("delta") != [0.25, 0.5, 0.75]:
        raise CandidateFeatureError(
            "robustness_grid.delta must be exactly [0.25, 0.5, 0.75]."
        )
    if robustness.get("stability_window_score_updates") != [2, 3]:
        raise CandidateFeatureError(
            "robustness_grid.stability_window_score_updates must be exactly [2, 3]."
        )
    rule = robustness.get("rule")
    if not isinstance(rule, str) or "worst" not in rule.lower():
        raise CandidateFeatureError(
            "robustness_grid.rule must require the worst grid result."
        )

    forbidden = selection.get("forbidden_selection_signals")
    if not isinstance(forbidden, list) or not forbidden:
        raise CandidateFeatureError(
            "forbidden_selection_signals must be a non-empty list."
        )
    normalized = " ".join(str(item).lower() for item in forbidden)
    for required_term in ("clinical", "treatment-group", "edi"):
        if required_term not in normalized:
            raise CandidateFeatureError(
                f"forbidden_selection_signals must explicitly cover {required_term!r}."
            )


def _jeffreys_mean_variance(
    successes: int,
    total: int,
    alpha: float,
    beta: float,
) -> tuple[float, float]:
    if not isinstance(successes, int) or isinstance(successes, bool):
        raise CandidateFeatureError("success counts must be integers.")
    if not isinstance(total, int) or isinstance(total, bool) or total <= 0:
        raise CandidateFeatureError("trial totals must be positive integers.")
    if not 0 <= successes <= total:
        raise CandidateFeatureError("success counts must lie in [0, total].")
    a = successes + alpha
    b = total - successes + beta
    denominator = a + b
    mean = a / denominator
    variance = (a * b) / (
        denominator * denominator * (denominator + 1.0)
    )
    return mean, variance


def _median(values: list[float]) -> float:
    if not values:
        raise CandidateFeatureError("Reaction-time list cannot be empty.")
    return float(statistics.median(values))


def _robust_scale(values: list[float], multiplier: float) -> float:
    center = _median(values)
    mad = statistics.median(abs(float(value) - center) for value in values)
    return float(multiplier) * float(mad)


def _validate_trial_summary(
    summary: dict[str, Any],
    *,
    minimum_go: int,
    minimum_nogo: int,
    require_rt_list: bool = True,
) -> None:
    for field in ("go_successes", "go_total", "nogo_successes", "nogo_total"):
        if field not in summary:
            raise CandidateFeatureError(f"Missing trial summary field {field}.")
    if summary["go_total"] < minimum_go:
        raise CandidateFeatureError("Insufficient GO trials.")
    if summary["nogo_total"] < minimum_nogo:
        raise CandidateFeatureError("Insufficient NO-GO trials.")
    if not 0 <= summary["go_successes"] <= summary["go_total"]:
        raise CandidateFeatureError("Invalid GO success count.")
    if not 0 <= summary["nogo_successes"] <= summary["nogo_total"]:
        raise CandidateFeatureError("Invalid NO-GO success count.")
    if require_rt_list:
        rts = summary.get("correct_go_rt_ms")
        if not isinstance(rts, list):
            raise CandidateFeatureError("correct_go_rt_ms must be a list.")
        for value in rts:
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or value <= 0
            ):
                raise CandidateFeatureError(
                    "correct_go_rt_ms values must be positive numbers."
                )


def compute_feature_deficits(
    config: dict[str, Any],
    case: dict[str, Any],
) -> dict[str, float | None]:
    """Compute B, S and V against one target-rule recovery reference."""
    validate_config(config)

    prechallenge = case.get("prechallenge_reference")
    recovery_reference = case.get("target_rule_recovery_reference")
    window = case.get("window")
    if (
        not isinstance(prechallenge, dict)
        or not isinstance(recovery_reference, dict)
        or not isinstance(window, dict)
    ):
        raise CandidateFeatureError(
            "Case requires prechallenge_reference, "
            "target_rule_recovery_reference and window objects."
        )

    eligibility_cfg = config["eligibility_reference_window"]
    recovery_ref_cfg = config["target_rule_recovery_reference"]
    win_cfg = config["postchallenge_window"]

    _validate_trial_summary(
        prechallenge,
        minimum_go=int(eligibility_cfg["minimum_go_trials"]),
        minimum_nogo=int(eligibility_cfg["minimum_nogo_trials"]),
        require_rt_list=False,
    )
    _validate_trial_summary(
        recovery_reference,
        minimum_go=int(recovery_ref_cfg["minimum_go_trials"]),
        minimum_nogo=int(recovery_ref_cfg["minimum_nogo_trials"]),
        require_rt_list=False,
    )
    _validate_trial_summary(
        window,
        minimum_go=int(win_cfg["minimum_go_trials"]),
        minimum_nogo=int(win_cfg["minimum_nogo_trials"]),
    )

    prechallenge_rule = prechallenge.get("rule_id")
    recovery_rule = window.get("rule_id")
    reference_rule = recovery_reference.get("rule_id")
    if not all(
        isinstance(value, str)
        for value in (prechallenge_rule, recovery_rule, reference_rule)
    ):
        raise CandidateFeatureError(
            "Prechallenge reference, recovery reference and window require rule_id."
        )
    if prechallenge_rule == recovery_rule:
        raise CandidateFeatureError(
            "A rule-switch case must use different prechallenge and "
            "post-switch rule IDs."
        )
    if reference_rule != recovery_rule:
        raise CandidateFeatureError(
            "Target-rule recovery reference must match the post-switch rule."
        )

    reference_go_stimulus = recovery_reference.get("go_stimulus_id")
    recovery_go_stimulus = window.get("go_stimulus_id")
    if (
        not isinstance(reference_go_stimulus, str)
        or not isinstance(recovery_go_stimulus, str)
        or reference_go_stimulus != recovery_go_stimulus
    ):
        raise CandidateFeatureError(
            "Target-rule recovery reference GO stimulus must match recovery."
        )

    numerical = config["numerical_rules"]
    alpha = float(numerical["beta_prior_alpha"])
    beta = float(numerical["beta_prior_beta"])
    minimum_behavior_scale = float(numerical["minimum_behavior_scale"])

    ref_go, ref_go_var = _jeffreys_mean_variance(
        int(recovery_reference["go_successes"]),
        int(recovery_reference["go_total"]),
        alpha,
        beta,
    )
    ref_nogo, ref_nogo_var = _jeffreys_mean_variance(
        int(recovery_reference["nogo_successes"]),
        int(recovery_reference["nogo_total"]),
        alpha,
        beta,
    )
    win_go, _ = _jeffreys_mean_variance(
        int(window["go_successes"]),
        int(window["go_total"]),
        alpha,
        beta,
    )
    win_nogo, _ = _jeffreys_mean_variance(
        int(window["nogo_successes"]),
        int(window["nogo_total"]),
        alpha,
        beta,
    )

    balanced_ref = 0.5 * (ref_go + ref_nogo)
    balanced_window = 0.5 * (win_go + win_nogo)

    predictive_go_var = (
        ref_go * (1.0 - ref_go) / int(window["go_total"])
    )
    predictive_nogo_var = (
        ref_nogo * (1.0 - ref_nogo) / int(window["nogo_total"])
    )
    behavior_comparison_scale = 0.5 * math.sqrt(
        ref_go_var
        + ref_nogo_var
        + predictive_go_var
        + predictive_nogo_var
    )
    behavior_scale = max(
        behavior_comparison_scale,
        minimum_behavior_scale,
    )
    deficit_b = max(
        0.0,
        (balanced_ref - balanced_window) / behavior_scale,
    )

    deficits: dict[str, float | None] = {
        "balanced_rule_success": deficit_b,
        "correct_go_rt_slowing": None,
        "correct_go_rt_variability_increase": None,
    }

    if case.get("timing_qualified") is not True:
        return deficits

    timing_floor = case.get("qualified_device_timing_floor_ms")
    if (
        not isinstance(timing_floor, (int, float))
        or isinstance(timing_floor, bool)
        or timing_floor <= 0
    ):
        raise CandidateFeatureError(
            "A timing-qualified case requires positive "
            "qualified_device_timing_floor_ms."
        )

    ref_rts_raw = recovery_reference.get("correct_go_rt_ms")
    if not isinstance(ref_rts_raw, list):
        raise CandidateFeatureError(
            "target_rule_recovery_reference.correct_go_rt_ms must be a list."
        )
    ref_rts = [float(value) for value in ref_rts_raw]
    win_rts = [float(value) for value in window["correct_go_rt_ms"]]
    if len(ref_rts) < int(
        recovery_ref_cfg["minimum_correct_go_rt_trials"]
    ):
        return deficits
    if any(value <= 0 for value in ref_rts):
        raise CandidateFeatureError(
            "Target-rule reference reaction times must be positive."
        )

    multiplier = float(numerical["robust_scale_mad_multiplier"])
    ref_center = _median(ref_rts)
    ref_scale = max(
        _robust_scale(ref_rts, multiplier),
        float(timing_floor),
    )

    if len(win_rts) >= int(
        win_cfg["minimum_correct_go_rt_trials_for_slowing"]
    ):
        win_center = _median(win_rts)
        deficits["correct_go_rt_slowing"] = max(
            0.0,
            (win_center - ref_center) / ref_scale,
        )

    if len(win_rts) >= int(
        win_cfg["minimum_correct_go_rt_trials_for_variability"]
    ):
        win_scale = max(
            _robust_scale(win_rts, multiplier),
            float(timing_floor),
        )
        deficits["correct_go_rt_variability_increase"] = max(
            0.0,
            (win_scale / ref_scale) - 1.0,
        )

    return deficits

def score_models(
    config: dict[str, Any],
    deficits: dict[str, float | None],
) -> dict[str, dict[str, Any]]:
    """Compute q for every candidate whose required components are available."""
    output: dict[str, dict[str, Any]] = {}
    for model in config["candidate_models"]:
        model_id = str(model["model_id"])
        missing = [
            feature
            for feature in model["features"]
            if deficits.get(feature) is None
        ]
        if missing:
            output[model_id] = {
                "computable": False,
                "missing_features": missing,
                "q": None,
            }
            continue

        q_value = -sum(
            float(model["weights"][feature]) * float(deficits[feature])
            for feature in model["features"]
        )
        output[model_id] = {
            "computable": True,
            "missing_features": [],
            "q": q_value,
        }
    return output


def evaluate_case(
    config: dict[str, Any],
    case: dict[str, Any],
) -> dict[str, Any]:
    deficits = compute_feature_deficits(config, case)
    models = score_models(config, deficits)
    return {
        "case_id": case.get("case_id"),
        "deficits": deficits,
        "models": models,
    }


def _strict_check(case: dict[str, Any], result: dict[str, Any]) -> None:
    expected = case.get("expected")
    if not isinstance(expected, dict):
        raise CandidateFeatureError(
            f"{case.get('case_id')}: --strict requires expected results."
        )

    for model_id, rule in expected.items():
        observed = result["models"].get(model_id)
        if observed is None:
            raise CandidateFeatureError(
                f"{case.get('case_id')}: unknown expected model {model_id}."
            )
        if "computable" in rule and observed["computable"] != rule["computable"]:
            raise CandidateFeatureError(
                f"{case.get('case_id')} {model_id}: computability mismatch."
            )
        relation = rule.get("q_relation")
        q_value = observed["q"]
        if relation == "zero":
            if q_value is None or not math.isclose(
                float(q_value), 0.0, abs_tol=1e-12
            ):
                raise CandidateFeatureError(
                    f"{case.get('case_id')} {model_id}: expected q == 0."
                )
        elif relation == "negative":
            if q_value is None or not float(q_value) < 0:
                raise CandidateFeatureError(
                    f"{case.get('case_id')} {model_id}: expected q < 0."
                )
        elif relation is not None:
            raise CandidateFeatureError(
                f"{case.get('case_id')} {model_id}: unknown q_relation."
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate ARC-ADHD recovery feature candidate cases."
    )
    parser.add_argument(
        "cases",
        nargs="?",
        type=Path,
        default=DEFAULT_CASES,
        help="JSON file containing deterministic candidate cases.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Candidate feature configuration.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Verify expected computability and q sign/zero conditions.",
    )
    args = parser.parse_args()

    try:
        config = load_json(args.config)
        if not isinstance(config, dict):
            raise CandidateFeatureError("Candidate config must be an object.")
        validate_config(config)

        cases = load_json(args.cases)
        if not isinstance(cases, list) or not cases:
            raise CandidateFeatureError("Cases file must contain a non-empty list.")

        results = []
        for case in cases:
            if not isinstance(case, dict):
                raise CandidateFeatureError("Every case must be an object.")
            result = evaluate_case(config, case)
            if args.strict:
                _strict_check(case, result)
            results.append(result)
    except (OSError, CandidateFeatureError) as exc:
        print(f"INVALID: {exc}")
        return 1

    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
