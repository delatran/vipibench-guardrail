from dataclasses import dataclass

import pytest

from vipibench.episode import EpisodeLabel
from vipibench.h3_analysis import (
    H3PairedEpisode,
    _h3_disposition,
    _stratified_bootstrap,
    analyze_h3_units,
)
from vipibench.h3_contract import locked_h3_analysis_contract


@dataclass(frozen=True)
class _Outcome:
    attack_success: bool | None
    clean_utility_pass: bool | None


def _unit(
    index: int, label: EpisodeLabel, *, security_gain: bool, utility_gain: bool
) -> H3PairedEpisode:
    if label == EpisodeLabel.INJECTION:
        detector = _Outcome(attack_success=True, clean_utility_pass=None)
        hybrid = _Outcome(attack_success=not security_gain, clean_utility_pass=None)
    else:
        detector = _Outcome(attack_success=None, clean_utility_pass=True)
        hybrid = _Outcome(attack_success=None, clean_utility_pass=utility_gain)
    return H3PairedEpisode(
        f"episode-{label.value}-{index:03d}",
        label,
        detector,  # type: ignore[arg-type]
        hybrid,  # type: ignore[arg-type]
        f"family-{index % 16:02d}",
    )


def _units(
    *, security_gain: bool = True, utility_gain: bool = True, size: int = 240
) -> list[H3PairedEpisode]:
    return [
        *[
            _unit(
                index,
                EpisodeLabel.INJECTION,
                security_gain=security_gain,
                utility_gain=utility_gain,
            )
            for index in range(size)
        ],
        *[
            _unit(
                index, EpisodeLabel.BENIGN, security_gain=security_gain, utility_gain=utility_gain
            )
            for index in range(size)
        ],
    ]


def _hashes() -> dict[str, str]:
    return {
        "four_arm_report_sha256": "A" * 64,
        "static_analysis_sha256": "B" * 64,
        "analysis_config_sha256": "C" * 64,
    }


def test_h3_reports_both_components_joint_bounds_and_locked_underpowered_warning() -> None:
    result = analyze_h3_units(_units(), source_hashes=_hashes(), bootstrap_iterations=100)

    assert result["status"] == "PASS", result["errors"]
    assert result["security"]["point_effect"] == 1.0
    assert result["utility"]["point_effect"] == 0.0
    assert result["security"]["bound_passes_locked_margin"] is True
    assert result["utility"]["bound_passes_locked_margin"] is True
    assert result["security"]["discordant_pair_counts"]["candidate_improves"] == 240
    assert result["joint_decision"]["theoretical_joint_pass"] is True
    assert result["joint_decision"]["disposition"] == "SUPPORTED"
    assert result["research_claim_eligible"] is True
    assert result["simultaneous_inference"]["security_family_count"] == 16
    assert result["simultaneous_inference"][
        "same_sampled_families_for_both_components"
    ] is True
    assert result["security"]["small_family_sensitivity"]["family_only_bootstrap"][
        "valid_iterations"
    ] == 100
    assert result["security"]["small_family_sensitivity"][
        "family_level_t_lower_bound"
    ]["family_count"] == 16
    assert result["simultaneous_inference"]["procedure"] == (
        "two_one_sided_95_percent_marginal_lower_bounds"
    )


def test_h3_component_failures_are_distinguished_before_the_locked_power_disposition() -> None:
    security_only = analyze_h3_units(
        _units(security_gain=True, utility_gain=False),
        source_hashes=_hashes(),
        bootstrap_iterations=50,
    )
    utility_only = analyze_h3_units(
        _units(security_gain=False, utility_gain=True),
        source_hashes=_hashes(),
        bootstrap_iterations=50,
    )
    neither = analyze_h3_units(
        _units(security_gain=False, utility_gain=False),
        source_hashes=_hashes(),
        bootstrap_iterations=50,
    )

    assert security_only["joint_decision"]["security_component_pass"] is True
    assert security_only["joint_decision"]["utility_component_pass"] is False
    assert utility_only["joint_decision"]["security_component_pass"] is False
    assert utility_only["joint_decision"]["utility_component_pass"] is True
    assert neither["joint_decision"]["theoretical_joint_pass"] is False


def test_h3_decision_branches_cover_pass_fail_and_exact_margin_boundary() -> None:
    contract = locked_h3_analysis_contract()
    contract["underpowered_policy"]["prelocked_margins_adequately_powered"] = True
    contract["underpowered_policy"]["worst_case_binary_absolute_mde"] = 0.01
    kwargs = {
        "errors": [],
        "sample_sizes": {"injection": 240, "benign": 240},
        "contract": contract,
    }
    assert (
        _h3_disposition(security_pass=True, utility_pass=True, **kwargs)["disposition"]
        == "SUPPORTED"
    )
    assert (
        _h3_disposition(security_pass=True, utility_pass=False, **kwargs)["disposition"]
        == "FAIL_UTILITY_NONINFERIORITY"
    )
    assert (
        _h3_disposition(security_pass=False, utility_pass=True, **kwargs)["disposition"]
        == "FAIL_SECURITY_SUPERIORITY"
    )
    assert (
        _h3_disposition(security_pass=False, utility_pass=False, **kwargs)["disposition"]
        == "FAIL_BOTH_COMPONENTS"
    )
    bounds = _stratified_bootstrap([0.05] * 240, [-0.02] * 240, iterations=20, seed=1)
    assert bounds["security_lower_bound"] == pytest.approx(0.05)
    assert bounds["utility_lower_bound"] == pytest.approx(-0.02)


def test_h3_missing_label_stratum_and_fallback_are_inconclusive_not_imputed() -> None:
    injection_only = [
        _unit(index, EpisodeLabel.INJECTION, security_gain=True, utility_gain=True)
        for index in range(20)
    ]
    missing = analyze_h3_units(injection_only, source_hashes=_hashes(), bootstrap_iterations=10)
    fallback = analyze_h3_units(
        _units(size=20),
        source_hashes=_hashes(),
        input_errors=["format_fallback_present"],
        bootstrap_iterations=10,
    )

    assert missing["status"] == "FAIL"
    assert "h3_benign_denominator_zero" in missing["errors"]
    assert missing["utility"]["point_effect"] is None
    assert missing["joint_decision"]["disposition"] == "INCONCLUSIVE_INVALID_OR_MISSING_INPUT"
    assert fallback["joint_decision"]["disposition"] == "INCONCLUSIVE_INVALID_OR_MISSING_INPUT"


def test_h3_bootstrap_is_deterministic_for_same_fixture_and_seed() -> None:
    first = analyze_h3_units(
        _units(), source_hashes=_hashes(), bootstrap_iterations=50, bootstrap_seed=7
    )
    second = analyze_h3_units(
        _units(), source_hashes=_hashes(), bootstrap_iterations=50, bootstrap_seed=7
    )

    assert first == second


def test_h3_family_set_mismatch_returns_fail_closed_artifact() -> None:
    units = _units()
    original = units[-1]
    units[-1] = H3PairedEpisode(
        original.episode_id,
        original.label,
        original.detector_only,
        original.hybrid,
        "utility-only-family",
    )

    result = analyze_h3_units(
        units,
        source_hashes=_hashes(),
        bootstrap_iterations=20,
    )

    assert result["status"] == "FAIL"
    assert "h3_security_utility_family_sets_mismatch" in result["errors"]
    assert result["simultaneous_inference"]["security_valid_iterations"] == 0
