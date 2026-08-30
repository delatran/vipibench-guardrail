"""The pre-execution RQ2 contract shared by protocol validation and analysis."""

from __future__ import annotations

from vipibench.provenance_contrast import DIAGNOSTIC_CONDITIONS


def locked_rq2_analysis_contract() -> dict[str, object]:
    """Return a fresh exact contract so callers cannot mutate a shared singleton."""

    return {
        "estimator_revision": "rq2-diagnostic-v2-family-clustered",
        "formal_condition_order": list(DIAGNOSTIC_CONDITIONS),
        "canonical_reference": {
            "benchmark_track": "provenance_contrast",
            "diagnostic_condition": "canonical",
            "expected_pair_count": 200,
        },
        "diagnostic_expected_pair_count": 40,
        "source_family_paired_reference": {
            "expected_source_family_count": 16,
            "unit": "source_family_aggregated_matched_pairs",
        },
        "core_template_disjoint_reference": {
            "benchmark_track": "core_stress",
            "disposition": "report_separately_never_substitute_for_canonical_provenance_reference",
        },
        "calibration_and_threshold": {
            "calibration": "temperature_scaling",
            "calibration_fit_split": "dev",
            "profile": "normal",
            "target_fpr": 0.05,
        },
        "uncertainty": {
            "confidence_level": 0.95,
            "bootstrap_unit": "two_stage_source_family_then_matched_pair",
            "family_weighting": "equal",
            "locked_seed_handling": "averaged_within_pair_and_family_not_bootstrapped",
            "calibration_metrics_recomputed_within_each_replicate": True,
            "bootstrap_iterations": 10000,
            "bootstrap_seed": 20260716,
        },
        "hypothesis_tests": {
            "method": "exhaustive_one_sided_family_sign_flip_sensitivity",
            "inference_status": "assumption_based_not_randomization_exact",
            "null_assumption": "family_effect_sign_symmetry_under_null",
            "sign_assignments": 65536,
            "familywise_adjustment": "holm",
            "familywise_alpha": 0.05,
        },
        "generalization_gate": {
            "h2_counterfactual_identity_required": True,
            "h2_max_abs_margin": 1.0e-6,
            "missing_or_renamed_formal_condition_blocks_generalization": True,
            "stale_or_non_dev_calibration_blocks_generalization": True,
        },
    }
