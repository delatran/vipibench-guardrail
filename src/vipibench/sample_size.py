from __future__ import annotations

from pathlib import Path

from vipibench.modeling import load_yaml

LOCKED_EPISODES = 2400
LOCKED_TRAINING_CONTEXT_LIMIT = 10000
LOCKED_SEEDS = [17, 29, 43]
LOCKED_ARMS = [
    ("exec_only", 0),
    ("synth_1k", 1000),
    ("synth_5k", 5000),
    ("synth_10k", 10000),
]


def _mapping(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def validate_sample_size_protocol(config_path: Path) -> dict[str, object]:
    """Validate the revised executable-benchmark and training-only scaling contract."""

    config = load_yaml(config_path)
    benchmark = _mapping(config.get("executable_benchmark"))
    synthetic = _mapping(config.get("training_only_synthetic"))
    scaling = _mapping(config.get("scaling_study"))
    stop_rule = _mapping(config.get("stop_rule"))
    errors: list[str] = []

    if config.get("schema_version") != "2.0.0":
        errors.append("schema_version_must_equal_2_0_0")
    if config.get("decision_status") != "locked_research_design":
        errors.append("decision_status_must_be_locked_research_design")
    if int(benchmark.get("accepted_episodes", -1)) != LOCKED_EPISODES:
        errors.append("accepted_episodes_must_equal_2400")
    if benchmark.get("labels_by_construction") is not True:
        errors.append("benchmark_labels_must_be_by_construction")
    if benchmark.get("deterministic_oracle_required") is not True:
        errors.append("deterministic_oracle_must_be_required")
    if benchmark.get("family_split") != [48, 16, 16]:
        errors.append("family_split_must_equal_48_16_16")

    if int(synthetic.get("accepted_context_limit", -1)) != LOCKED_TRAINING_CONTEXT_LIMIT:
        errors.append("training_context_limit_must_equal_10000")
    if synthetic.get("training_only") is not True:
        errors.append("synthetic_contexts_must_be_training_only")
    if synthetic.get("forbidden_splits") != ["dev", "test", "ood"]:
        errors.append("synthetic_forbidden_splits_must_equal_dev_test_ood")

    if scaling.get("seeds") != LOCKED_SEEDS:
        errors.append("scaling_seeds_must_equal_17_29_43")
    if scaling.get("fixed_exec_dev_test") is not True:
        errors.append("scaling_must_fix_exec_dev_test")
    if scaling.get("threshold_source") != "exec_dev_only":
        errors.append("threshold_source_must_equal_exec_dev_only")
    observed_arms = []
    for arm in scaling.get("arms", []):
        value = _mapping(arm)
        observed_arms.append(
            (str(value.get("arm_id", "")), int(value.get("synthetic_contexts", -1)))
        )
    if observed_arms != LOCKED_ARMS:
        errors.append("scaling_arms_must_equal_0_1k_5k_10k")

    if float(stop_rule.get("practical_equivalence_margin_abs", -1)) != 0.01:
        errors.append("practical_equivalence_margin_must_equal_0_01")
    if stop_rule.get("comparison") != "synth_10k_vs_synth_5k":
        errors.append("stop_rule_comparison_must_equal_10k_vs_5k")
    if stop_rule.get("stop_if_interval_contains_zero") is not True:
        errors.append("stop_rule_must_use_interval_containing_zero")
    if float(stop_rule.get("max_hard_negative_fpr_regression_abs", -1)) != 0.01:
        errors.append("hard_negative_regression_limit_must_equal_0_01")

    return {
        "schema_version": "2.0.0",
        "status": "PASS" if not errors else "FAIL",
        "config_path": str(config_path),
        "errors": errors,
        "accepted_episode_count": LOCKED_EPISODES,
        "training_context_limit": LOCKED_TRAINING_CONTEXT_LIMIT,
        "arms": observed_arms,
    }
