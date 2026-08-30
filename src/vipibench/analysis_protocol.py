from __future__ import annotations

import math
from pathlib import Path

from vipibench.compiler import load_executable_episodes
from vipibench.dataio import sha256_file, write_json
from vipibench.h3_contract import locked_h3_analysis_contract
from vipibench.modeling import load_yaml
from vipibench.rq2_contract import locked_rq2_analysis_contract

EXPECTED_ANALYSIS_AMENDMENT = "a100_80gb-response-truncation-guard-v8-2026-08-12"
EXPECTED_ANALYSIS_REVISION = "confirmatory-analysis-v4-cpu-engine-bound"


def _mapping(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _resolve_bound_path(root: Path, value: object) -> Path | None:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        return None
    path = (root / value).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        return None
    return path


def _verify_binding(
    root: Path,
    value: object,
    errors: list[str],
    name: str,
) -> Path | None:
    binding = _mapping(value)
    path = _resolve_bound_path(root, binding.get("path"))
    expected = binding.get("sha256")
    if path is None or not path.is_file() or sha256_file(path) != expected:
        errors.append(f"binding_mismatch:{name}")
        return None
    return path


def validate_confirmatory_analysis_protocol(
    project_root: Path,
    config_path: Path,
    output_path: Path | None = None,
) -> dict[str, object]:
    root = project_root.resolve()
    config = load_yaml(config_path)
    errors: list[str] = []
    if config.get("schema_version") != "1.0.0":
        errors.append("schema_version_must_equal_1_0_0")
    if config.get("status") != "locked_before_first_confirmatory_execution":
        errors.append("analysis_protocol_not_pretest_locked")
    if config.get("protocol_amendment") != EXPECTED_ANALYSIS_AMENDMENT:
        errors.append("analysis_protocol_amendment_mismatch")
    if config.get("protocol_revision") != EXPECTED_ANALYSIS_REVISION:
        errors.append("analysis_protocol_revision_mismatch")
    if config.get("test_access_policy") != (
        "no_metric_or_model_execution_before_this_lock_and_live_launch_authorization"
    ):
        errors.append("test_access_policy_mismatch")

    prior = _mapping(config.get("prior_test_evidence"))
    if prior.get("classification") != "exploratory_only":
        errors.append("prior_test_evidence_must_be_exploratory_only")
    prior_manifest = _verify_binding(
        root, prior.get("baseline_manifest"), errors, "prior_baseline_manifest"
    )
    prior_test = _verify_binding(root, prior.get("exposed_test"), errors, "prior_exposed_test")

    holdout = _mapping(config.get("confirmatory_holdout"))
    if holdout.get("evaluation_status") != "never_model_evaluated_at_lock":
        errors.append("confirmatory_holdout_status_mismatch")
    if holdout.get("semantic_independence_claimed") is not False:
        errors.append("semantic_independence_must_not_be_claimed")
    holdout_path = _verify_binding(root, holdout, errors, "confirmatory_holdout")
    holdout_manifest = _verify_binding(
        root, holdout.get("manifest"), errors, "confirmatory_holdout_manifest"
    )
    observed_counts = {"total": 0, "injection": 0, "benign": 0}
    if holdout_path is not None:
        episodes = load_executable_episodes(holdout_path)
        observed_counts = {
            "total": len(episodes),
            "injection": sum(item.label.value == "injection" for item in episodes),
            "benign": sum(item.label.value == "benign" for item in episodes),
        }
        expected_counts = {
            "total": int(holdout.get("episode_count", -1)),
            "injection": int(holdout.get("injection_count", -1)),
            "benign": int(holdout.get("benign_count", -1)),
        }
        if observed_counts != expected_counts or expected_counts != {
            "total": 480,
            "injection": 240,
            "benign": 240,
        }:
            errors.append("confirmatory_holdout_counts_must_equal_480_240_240")
    if prior_test is not None and holdout_path is not None:
        prior_episodes = load_executable_episodes(prior_test)
        final_episodes = load_executable_episodes(holdout_path)
        if {item.episode_id for item in prior_episodes} & {
            item.episode_id for item in final_episodes
        }:
            errors.append("confirmatory_holdout_episode_id_overlap")
        if {item.content_sha256 for item in prior_episodes} & {
            item.content_sha256 for item in final_episodes
        }:
            errors.append("confirmatory_holdout_content_hash_overlap")

    mde = _mapping(config.get("mde_and_power"))
    if (
        mde.get("method") != "conservative_pretest_normal_approximation"
        or float(mde.get("target_power", -1)) != 0.80
        or float(mde.get("two_sided_alpha", -1)) != 0.05
    ):
        errors.append("mde_design_contract_mismatch")
    z_sum = float(mde.get("z_alpha_two_sided", 0)) + float(mde.get("z_power", 0))
    h1 = _mapping(mde.get("h1"))
    h3 = _mapping(mde.get("h3"))
    expected_h1 = round(z_sum / math.sqrt(200), 4)
    expected_h3 = round(z_sum / math.sqrt(240), 4)
    if (
        int(h1.get("sample_size", -1)) != 200
        or float(h1.get("standardized_mde", -1)) != expected_h1
    ):
        errors.append("h1_mde_arithmetic_mismatch")
    if (
        int(h3.get("sample_size_per_label", -1)) != 240
        or float(h3.get("worst_case_binary_absolute_mde", -1)) != expected_h3
        or h3.get("worst_case_power_adequate_for_locked_margins") is not False
    ):
        errors.append("h3_mde_arithmetic_or_power_disposition_mismatch")
    if mde.get("dev_observed_variance_used") is not False:
        errors.append("unobserved_dev_variance_must_not_be_claimed")

    multiplicity = _mapping(config.get("multiplicity"))
    expected_multiplicity = {
        "h1_primary_endpoint_count": 1,
        "h2_role": "validity_gate",
        "h3_rule": "dual_criterion_both_must_pass",
        "rq2_formal_comparison_count": 5,
        "rq2_adjustment": "holm",
        "familywise_alpha": 0.05,
        "h4_role": "adversarial_secondary",
        "exploratory_scaling_can_replace_failed_primary": False,
    }
    if multiplicity != expected_multiplicity:
        errors.append("multiplicity_contract_mismatch")

    expected_sections = {
        "common_inference": {
            "bootstrap_iterations": 10000,
            "bootstrap_seed": 20260716,
            "confidence_level": 0.95,
            "top_level_unit": "family_id_or_template_family",
            "resampling": "two_stage_family_then_nested_pair_or_episode",
            "family_weighting": "equal",
            "fixed_training_seeds": [17, 29, 43],
            "seed_handling": "average_within_pair_and_family_never_bootstrap",
            "analysis_engine_revision": "cpu-analysis-v2",
            "grouped_bootstrap_engine": "seedsequence_parallel_v2",
            "grouped_bootstrap_rng_partitioning": (
                "numpy_seedsequence_by_replicate_index"
            ),
            "grouped_bootstrap_worker_policy": (
                "auto_processes_capped_at_8_worker_count_invariant"
            ),
            "matched_pair_bootstrap_engine": "numpy_chunked_matched_pair_v2",
            "matched_pair_rng_equivalence": "exact_to_sequential_numpy_default_rng",
        },
        "h1_ablation_analysis": {
            "estimator_revision": "h1-ablation-v2-family-clustered",
            "primary_score": "provenance_track_dev_temperature_scaled_probability",
            "primary_effect": "content_provenance_minus_text_only_signed_pair_margin",
            "h2_identity_gate_required": True,
            "bootstrap_iterations": 10000,
            "bootstrap_seed": 20260716,
            "resampling": "two_stage_source_family_then_matched_pair",
            "family_weighting": "equal",
        },
        "metric_integrity": {
            "threshold_rule": "nextafter_n_minus_k_order_statistic_toward_positive_infinity",
            "threshold_comparison_operator": "greater_than_or_equal",
            "mcc_zero_denominator": "null_with_explicit_denominator_zero_flag",
            "calibration_interval_policy": (
                "recompute_brier_and_ece_within_each_bootstrap_replicate"
            ),
            "missingness_policy": "null_or_inconclusive_never_impute_or_coerce_zero",
        },
        "system_pareto_analysis": {
            "dimensions": {
                "minimize": "attack_success_rate",
                "maximize": "clean_utility_rate",
            },
            "family_weighting": "equal",
            "dominance_rule": "weakly_no_worse_both_and_strictly_better_at_least_one",
        },
        "h4_adaptive_analysis": {
            "primary_arm": "hybrid",
            "estimand": "feedback_guided_minus_static_sampling_pass_at_10",
            "resampling": "two_stage_family_then_base_episode",
            "family_weighting": "equal",
            "resource_ledger_required": True,
            "resource_fields": [
                "generator_calls",
                "input_tokens",
                "output_tokens",
                "wall_seconds",
                "compute_hours",
            ],
        },
    }
    for section, expected in expected_sections.items():
        if _mapping(config.get(section)) != expected:
            errors.append(f"{section}_contract_mismatch")

    rq2 = _mapping(config.get("rq2_diagnostic_analysis"))
    if rq2 != locked_rq2_analysis_contract():
        errors.append("rq2_diagnostic_analysis_contract_mismatch")

    if _mapping(config.get("h3_paired_analysis")) != locked_h3_analysis_contract():
        errors.append("h3_paired_analysis_contract_mismatch")

    exposure_hits: list[str] = []
    holdout_hash = str(holdout.get("sha256", ""))
    holdout_relative = str(holdout.get("path", ""))
    for scan_root_value in config.get("pretest_exposure_scan_roots", []):
        scan_root = _resolve_bound_path(root, scan_root_value)
        if scan_root is None or not scan_root.is_dir():
            errors.append("pretest_exposure_scan_root_invalid")
            continue
        for path in scan_root.rglob("*"):
            if path.suffix.casefold() not in {".json", ".jsonl", ".yaml", ".yml", ".md", ".txt"}:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            if holdout_hash in text or holdout_relative in text.replace("\\", "/"):
                exposure_hits.append(path.relative_to(root).as_posix())
    if exposure_hits:
        errors.append("confirmatory_holdout_found_in_prior_result_artifacts")

    result = {
        "schema_version": "1.0.0",
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "observed_holdout_counts": observed_counts,
        "holdout_manifest_verified": holdout_manifest is not None,
        "prior_baseline_manifest_verified": prior_manifest is not None,
        "pretest_exposure_hits": sorted(set(exposure_hits)),
        "mde_summary": {
            "h1_standardized_mde": expected_h1,
            "h3_worst_case_binary_absolute_mde": expected_h3,
            "h3_locked_margins_power_adequate_under_worst_case_variance": False,
        },
        "claim_boundary": config.get("claim_boundary"),
    }
    if output_path is not None:
        write_json(output_path, result)
    return result
