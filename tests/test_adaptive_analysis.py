import hashlib

from vipibench.adaptive_analysis import (
    ARMS,
    STRATEGIES,
    _validate_generation_bindings,
    analyze_adaptive_report,
)
from vipibench.dataio import canonical_json


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest().upper()


def _source_hashes() -> dict[str, str | None]:
    return {
        "adaptive_report_sha256": "A" * 64,
        "candidate_dataset_sha256": "B" * 64,
        "candidate_validity_sha256": "C" * 64,
        "candidate_manifest_sha256": "D" * 64,
        "generator_config_sha256": "E" * 64,
        "runtime_telemetry_sha256": None,
        "strict_capacity_receipt_sha256": None,
    }


def _resource_ledger() -> dict[str, object]:
    return {
        "status": "PASS",
        "errors": [],
        "accounting_unit": "unique_generator_call_id",
        "totals": {
            "generator_calls": 4,
            "candidate_proposals": 40,
            "input_tokens": 400,
            "output_tokens": 200,
            "wall_seconds": 4.0,
        },
        "by_strategy": {
            strategy: {
                "generator_calls": 2,
                "candidate_proposals": 20,
                "input_tokens": 200,
                "output_tokens": 100,
                "wall_seconds": 2.0,
            }
            for strategy in STRATEGIES
        },
    }


def _fixture(effect: str = "positive") -> tuple[dict[str, object], dict[str, object]]:
    validity_records: list[dict[str, object]] = []
    records: list[dict[str, object]] = []
    for base_index in range(2):
        base_id = f"base-{base_index}"
        for strategy in STRATEGIES:
            for candidate_index in range(10):
                candidate_id = f"{base_id}-{strategy}-{candidate_index}"
                candidate_hash = _sha(f"candidate:{candidate_id}")
                text_hash = _sha(f"text:{candidate_id}")
                validity_records.append(
                    {
                        "episode_id": candidate_id,
                        "episode_sha256": candidate_hash,
                        "base_episode_id": base_id,
                        "strategy": strategy,
                        "candidate_index": candidate_index,
                        "raw_text_sha256": text_hash,
                        "semantic_preservation": "UNVERIFIED",
                    }
                )
                security_failure = _security_failure(effect, base_index, strategy, candidate_index)
                for arm in ARMS:
                    records.append(
                        {
                            "base_episode_id": base_id,
                            "family_id": f"family-{base_index}",
                            "candidate_episode_id": candidate_id,
                            "candidate_episode_sha256": candidate_hash,
                            "candidate_text_sha256": text_hash,
                            "candidate_index": candidate_index,
                            "strategy": strategy,
                            "arm": arm,
                            "detector_score": 0.1,
                            "security_failure": security_failure,
                            "attack_success": False,
                            "format_status": "strict_json",
                            "fallback_status": "not_used",
                            "format_fallback": False,
                            "parse_error_class": None,
                            "trajectory_synthetic_fixture": False,
                            "outcome_sha256": _sha(f"outcome:{candidate_id}:{arm}"),
                            "violation_codes": ["UNKNOWN_TOOL"] if security_failure else [],
                        }
                    )
    report: dict[str, object] = {
        "status": "PASS",
        "errors": [],
        "execution_plan": {"equal_query_budget": True},
        "records": records,
        "records_sha256": _sha(canonical_json(records)),
    }
    validity: dict[str, object] = {
        "status": "PASS",
        "errors": [],
        "raw_proposal_count": 40,
        "accepted_candidate_count": 40,
        "rejected_candidate_count": 0,
        "semantic_preservation_verified": False,
        "research_claim_eligible": False,
        "records": validity_records,
    }
    return report, validity


def _security_failure(effect: str, base_index: int, strategy: str, candidate_index: int) -> bool:
    if candidate_index != 0 or base_index != 0:
        return False
    if effect == "positive":
        return strategy == "feedback_guided"
    if effect == "negative":
        return strategy == "static_sampling"
    return False


def _analyze(effect: str = "positive") -> dict[str, object]:
    report, validity = _fixture(effect)
    return analyze_adaptive_report(
        report=report,
        candidate_validity=validity,
        telemetry_ledger=None,
        strict_capacity_receipt=None,
        generation_resource_ledger=_resource_ledger(),
        source_hashes=_source_hashes(),
        expected_base_count=2,
        bootstrap_iterations=100,
        bootstrap_seed=7,
    )


def test_h4_uses_unified_security_failure_and_base_paired_positive_effect() -> None:
    result = _analyze("positive")

    assert result["status"] == "PASS", result["errors"]
    assert result["outcome_field"] == "security_failure"
    assert result["paired_unit"] == "family_id_then_base_episode_id"
    assert result["family_count"] == 2
    assert result["generation_resource_ledger"]["validation_status"] == "PASS"
    for arm in ARMS:
        assert result["pass_at_10"]["feedback_guided"][arm] == 0.5
        assert result["pass_at_10"]["static_sampling"][arm] == 0.0
        assert result["paired_guided_minus_static"][arm]["effect"] == 0.5
        assert result["paired_guided_minus_static"][arm]["weighting"] == "equal_family"
        sensitivity = result["paired_guided_minus_static"][arm]["small_family_sensitivity"]
        assert sensitivity["family_only_bootstrap"]["valid_iterations"] == 100
        assert sensitivity["family_level_t_interval"]["family_count"] == 2
        assert result["paired_guided_minus_static"][arm]["discordant_counts"] == {
            "guided_only_failure": 1,
            "static_only_failure": 0,
            "equal_outcome": 1,
        }
    assert result["research_claim_eligible"] is False
    assert result["claim_disposition"]["H4"] == "INCONCLUSIVE_SEMANTIC_PRESERVATION_UNVERIFIED"
    assert "runtime_telemetry_missing" in result["primary_ineligibility_reasons"]


def test_h4_paired_effect_retains_zero_and_negative_results() -> None:
    zero = _analyze("zero")
    negative = _analyze("negative")

    for arm in ARMS:
        assert zero["paired_guided_minus_static"][arm]["effect"] == 0.0
        assert negative["paired_guided_minus_static"][arm]["effect"] == -0.5
        assert (
            negative["paired_guided_minus_static"][arm]["discordant_counts"]["static_only_failure"]
            == 1
        )


def test_duplicate_fallback_fixture_and_stale_raw_hash_cannot_support_h4() -> None:
    report, validity = _fixture()
    records = report["records"]
    assert isinstance(records, list)
    duplicate = dict(records[0])
    records.append(duplicate)
    report["records_sha256"] = _sha(canonical_json(records))
    duplicate_result = analyze_adaptive_report(
        report=report,
        candidate_validity=validity,
        telemetry_ledger=None,
        strict_capacity_receipt=None,
        generation_resource_ledger=_resource_ledger(),
        source_hashes=_source_hashes(),
        expected_base_count=2,
        bootstrap_iterations=20,
    )
    assert duplicate_result["status"] == "FAIL"
    assert any(
        error.startswith("adaptive_record_duplicate_identity")
        for error in duplicate_result["errors"]
    )
    assert duplicate_result["research_claim_eligible"] is False

    fallback_report, fallback_validity = _fixture()
    fallback_records = fallback_report["records"]
    assert isinstance(fallback_records, list)
    fallback_records[0]["format_fallback"] = True
    fallback_records[0]["fallback_status"] = "used"
    fallback_records[0]["format_status"] = "safe_fallback"
    fallback_records[0]["parse_error_class"] = "json_decode_error"
    fallback_report["records_sha256"] = _sha(canonical_json(fallback_records))
    fallback_result = analyze_adaptive_report(
        report=fallback_report,
        candidate_validity=fallback_validity,
        telemetry_ledger=None,
        strict_capacity_receipt=None,
        generation_resource_ledger=_resource_ledger(),
        source_hashes=_source_hashes(),
        expected_base_count=2,
        bootstrap_iterations=20,
    )
    assert fallback_result["status"] == "PASS"
    assert "format_fallback_present" not in fallback_result["primary_ineligibility_reasons"]
    bounds = fallback_result["unobserved_response_bounds"]
    assert bounds["bounds_required"] is True
    assert bounds["unobserved_response_count"] == 1
    assert fallback_result["research_claim_eligible"] is False

    stale_report, stale_validity = _fixture()
    stale_report["records_sha256"] = "0" * 64
    stale_result = analyze_adaptive_report(
        report=stale_report,
        candidate_validity=stale_validity,
        telemetry_ledger=None,
        strict_capacity_receipt=None,
        generation_resource_ledger=_resource_ledger(),
        source_hashes=_source_hashes(),
        expected_base_count=2,
        bootstrap_iterations=20,
    )
    assert stale_result["status"] == "FAIL"
    assert "adaptive_report_records_hash_mismatch" in stale_result["errors"]


def test_stale_generator_config_binding_is_rejected_before_h4_analysis() -> None:
    errors: list[str] = []
    _validate_generation_bindings(
        {
            "candidate_dataset_sha256": "A" * 64,
            "detector_model_version": "detector-r1",
        },
        {
            "status": "PASS",
            "errors": [],
            "candidate_dataset_sha256": "A" * 64,
            "candidate_validity_manifest_sha256": "B" * 64,
            "config_sha256": "C" * 64,
            "detector_model_version": "detector-r1",
            "generation_resource_ledger": _resource_ledger(),
        },
        candidate_dataset_hash="A" * 64,
        validity_hash="B" * 64,
        config_hash="D" * 64,
        errors=errors,
    )

    assert errors == ["candidate_generation_binding_mismatch:config_sha256"]


def test_missing_source_hash_field_fails_closed_without_crashing() -> None:
    report, validity = _fixture()
    source_hashes = _source_hashes()
    source_hashes.pop("candidate_dataset_sha256")

    result = analyze_adaptive_report(
        report=report,
        candidate_validity=validity,
        telemetry_ledger=None,
        strict_capacity_receipt=None,
        generation_resource_ledger=_resource_ledger(),
        source_hashes=source_hashes,
        expected_base_count=2,
        bootstrap_iterations=20,
    )

    assert result["status"] == "FAIL"
    assert "adaptive_analysis_source_hash_fields_mismatch" in result["errors"]
