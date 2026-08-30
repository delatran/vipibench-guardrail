import copy
import json
from pathlib import Path

import pytest

from vipibench.agent_trajectory import (
    AgentTrajectoryRecord,
    bind_agent_trajectory_record,
    build_agent_request,
    write_agent_trajectory_records,
)
from vipibench.compiler import load_executable_episodes
from vipibench.dataio import sha256_file
from vipibench.episode import EpisodeLabel, ExecutableEpisode
from vipibench.modeling import load_yaml
from vipibench.policy_gate import bind_detector_signal
from vipibench.runtime_capacity import RuntimeProbe, check_runtime_profile
from vipibench.runtime_telemetry import (
    build_strict_capacity_receipt,
    build_telemetry_ledger,
    record_stage_interval,
    strict_capacity_receipt_sha256,
)
from vipibench.system_analysis import LOCKED_METRICS, analyze_static_records, analyze_static_system
from vipibench.system_runner import (
    SystemArm,
    build_fixture_proposed_trajectory,
    build_safe_fallback_trajectory,
    run_four_arms,
)


def _episodes() -> tuple[ExecutableEpisode, ExecutableEpisode]:
    dataset = load_executable_episodes(Path("data/splits/frozen/test.jsonl"))
    injection = next(item for item in dataset if item.label == EpisodeLabel.INJECTION)
    benign = next(item for item in dataset if item.label == EpisodeLabel.BENIGN)
    return injection, benign


def _trajectory_record(
    episode: ExecutableEpisode,
    *,
    latency_seconds: float,
) -> AgentTrajectoryRecord:
    return bind_agent_trajectory_record(
        episode=episode,
        request=build_agent_request(episode),
        trajectory=build_fixture_proposed_trajectory(episode),
        model_id="fixture-target",
        model_revision="fixture-target-revision",
        tokenizer_revision="fixture-tokenizer-revision",
        synthetic_fixture=True,
        observed_model_request_wall_seconds=latency_seconds,
        input_token_count=12,
        output_token_count=5,
    )


def _raw_record(
    episode: ExecutableEpisode,
    trajectory_record: AgentTrajectoryRecord,
    *,
    score: float,
) -> dict[str, object]:
    signal = bind_detector_signal(
        {
            "score": score,
            "episode_sha256": episode.content_sha256,
            "model_version": "observed-detector-fixture",
            "synthetic_fixture": False,
        }
    )
    arm_results = run_four_arms(episode, trajectory_record.trajectory, signal)
    return {
        "episode_id": episode.episode_id,
        "episode_sha256": episode.content_sha256,
        "label": episode.label.value,
        "family_id": episode.metadata.family_id,
        "template_id": episode.metadata.template_id,
        "calibrated_detector_score": score,
        "proposed_trajectory_sha256": trajectory_record.trajectory.trajectory_sha256,
        "detector_signal_sha256": signal.signal_sha256,
        "trajectory_record_sha256": trajectory_record.record_sha256,
        "target_request_sha256": trajectory_record.request_sha256,
        "target_prompt_sha256": trajectory_record.prompt_sha256,
        "format_status": trajectory_record.format_status,
        "fallback_status": trajectory_record.fallback_status,
        "format_fallback": trajectory_record.format_fallback,
        "parse_error_class": trajectory_record.parse_error_class,
        "observed_model_request_wall_seconds": (
            trajectory_record.observed_model_request_wall_seconds
        ),
        "input_token_count": trajectory_record.input_token_count,
        "output_token_count": trajectory_record.output_token_count,
        "arms": [item.model_dump(mode="json") for item in arm_results],
    }


def _fixture_inputs() -> tuple[dict[str, object], dict[str, AgentTrajectoryRecord]]:
    injection, benign = _episodes()
    injection_record = _trajectory_record(injection, latency_seconds=1.0)
    benign_record = _trajectory_record(benign, latency_seconds=3.0)
    records = [
        _raw_record(injection, injection_record, score=0.95),
        _raw_record(benign, benign_record, score=0.05),
    ]
    report = {
        "schema_version": "2.0.0",
        "status": "PASS",
        "errors": [],
        "paired_episode_count": len(records),
        "system_arms": [item.value for item in SystemArm],
        "detector_signal_kind": "observed_model_prediction",
        "synthetic_trajectory_count": 2,
        "format_fallback_count": 0,
        "format_repair_count": 0,
        "parse_failure_count": 0,
        "unresolved_parse_failure_count": 0,
        "records": records,
    }
    trajectories = {
        injection_record.episode_id: injection_record,
        benign_record.episode_id: benign_record,
    }
    return report, trajectories


def _source_hashes() -> dict[str, str]:
    return {
        "four_arm_report_sha256": "A" * 64,
        "trajectory_records_sha256": "B" * 64,
        "runtime_telemetry_sha256": "C" * 64,
        "strict_capacity_receipt_sha256": "D" * 64,
    }


def _strict_live_telemetry() -> tuple[dict[str, object], dict[str, object]]:
    profile = load_yaml(Path("configs/profiles/accelerator_80gb.yaml"))
    runtime_check = check_runtime_profile(
        profile,
        RuntimeProbe(
            compute_available=True,
            device_type="cuda",
            device_name="NVIDIA A100-SXM4-80GB",
            device_index=0,
            device_memory_gib=79.2,
            bf16_supported=True,
            tensor_probe_passed=True,
            system_ram_gib=53.0,
            disk_free_gib=120.0,
            compute_capability="8.0",
            evidence_kind="observed",
        ),
    )
    receipt = build_strict_capacity_receipt(runtime_check)
    receipt_hash = strict_capacity_receipt_sha256(receipt)
    ledger = build_telemetry_ledger(
        [
            record_stage_interval(
                stage_id="target_trajectory_generation",
                interval_id="fixture-target-interval",
                run_id="fixture-target-run",
                attempt_id="initial",
                start_monotonic_seconds=1.0,
                end_monotonic_seconds=3601.0,
                status="completed",
                accelerator_stage=True,
                observed_device_receipt_sha256=receipt_hash,
                input_artifact_hashes={"target_config": "E" * 64},
                output_artifact_hashes={"trajectory_records": "F" * 64},
            )
        ],
        strict_capacity_receipt=receipt,
        local_only=False,
    )
    return ledger, receipt


def test_static_metric_contract_has_exact_schema_denominators_and_paired_intervals() -> None:
    report, trajectories = _fixture_inputs()
    telemetry, receipt = _strict_live_telemetry()

    result = analyze_static_records(
        four_arm_report=report,
        trajectory_records=trajectories,
        telemetry_ledger=telemetry,
        strict_capacity_receipt=receipt,
        source_hashes=_source_hashes(),
        bootstrap_iterations=100,
        bootstrap_seed=91,
    )

    assert result["status"] == "PASS", result["errors"]
    assert result["metric_schema"] == list(LOCKED_METRICS)
    assert result["paired_unit"] == "family_id_then_episode_id_with_all_four_system_arms"
    assert result["paired_episode_count"] == 2
    assert result["family_count"] == 1
    assert result["research_claim_eligible"] is False
    assert "synthetic_trajectory_record" in result["primary_ineligibility_reasons"]
    assert result["telemetry"]["compute_hours"] == pytest.approx(1.0)

    metrics = result["metrics"]
    none = metrics[SystemArm.NONE.value]
    policy = metrics[SystemArm.POLICY_ONLY.value]
    assert none["attack_success_rate"]["value"] == 1.0
    assert policy["attack_success_rate"]["value"] == 0.0
    assert policy["containment_rate"]["value"] == 1.0
    assert none["clean_utility_rate"]["value"] == 1.0
    assert none["target_request_latency_p50_seconds"]["value"] == 2.0
    assert none["target_request_latency_p95_seconds"]["value"] == pytest.approx(2.9)
    assert none["compute_hours"]["value"] == pytest.approx(1.0)
    assert none["unique_failure_discoveries_per_compute_hour"]["value"] == 1.0
    assert result["pareto_frontier"]["status"] == "PASS"
    assert result["pareto_frontier"]["weighting"] == "equal_family"
    assert SystemArm.NONE.value in result["pareto_frontier"]["dominated_by"]
    for arm_metrics in metrics.values():
        assert set(arm_metrics) == set(LOCKED_METRICS)
        for metric in arm_metrics.values():
            assert {
                "numerator",
                "denominator",
                "eligibility",
                "unit",
                "provenance_hashes",
            }.issubset(metric)
    interval = result["confidence_intervals"]["arms"][SystemArm.NONE.value]
    assert 0 < interval["attack_success_rate"]["valid_iterations"] < 100
    assert interval["attack_success_rate"]["null_behavior"] == (
        "resamples with a zero denominator were excluded from percentile interval"
    )
    assert interval["target_request_latency_p95_seconds"]["valid_iterations"] == 100
    sensitivity = result["confidence_intervals"]["small_family_sensitivity"]
    assert sensitivity["family_only_bootstrap"]["arms"][SystemArm.NONE.value][
        "attack_success_rate"
    ]["valid_iterations"] == 100
    assert sensitivity["family_level_t_intervals"]["arms"][SystemArm.NONE.value][
        "attack_success_rate"
    ]["family_count"] == 1


def test_missing_arm_and_duplicate_episode_fail_closed() -> None:
    report, trajectories = _fixture_inputs()
    missing_arm = copy.deepcopy(report)
    missing_arm["records"][0]["arms"] = missing_arm["records"][0]["arms"][:-1]
    missing_result = analyze_static_records(
        four_arm_report=missing_arm,
        trajectory_records=trajectories,
        telemetry_ledger=None,
        strict_capacity_receipt=None,
        source_hashes=_source_hashes(),
        bootstrap_iterations=10,
    )
    assert missing_result["status"] == "FAIL"
    assert any(error.startswith("missing_arm:") for error in missing_result["errors"])

    duplicate = copy.deepcopy(report)
    duplicate["records"].append(copy.deepcopy(duplicate["records"][0]))
    duplicate["paired_episode_count"] = len(duplicate["records"])
    duplicate_result = analyze_static_records(
        four_arm_report=duplicate,
        trajectory_records=trajectories,
        telemetry_ledger=None,
        strict_capacity_receipt=None,
        source_hashes=_source_hashes(),
        bootstrap_iterations=10,
    )
    assert duplicate_result["status"] == "FAIL"
    assert any(error.startswith("duplicate_episode:") for error in duplicate_result["errors"])


def test_zero_denominator_is_explicit_null_and_missing_telemetry_blocks_primary_claim() -> None:
    report, trajectories = _fixture_inputs()
    injection_id = next(
        str(item["episode_id"])
        for item in report["records"]
        if item["label"] == EpisodeLabel.INJECTION.value
    )
    one_injection = copy.deepcopy(report)
    one_injection["records"] = [
        item for item in one_injection["records"] if item["episode_id"] == injection_id
    ]
    one_injection["paired_episode_count"] = 1
    one_injection["synthetic_trajectory_count"] = 1
    injection_trajectories = {injection_id: trajectories[injection_id]}

    result = analyze_static_records(
        four_arm_report=one_injection,
        trajectory_records=injection_trajectories,
        telemetry_ledger=None,
        strict_capacity_receipt=None,
        source_hashes=_source_hashes(),
        bootstrap_iterations=10,
    )

    assert result["status"] == "PASS", result["errors"]
    assert result["research_claim_eligible"] is False
    assert "runtime_telemetry_missing" in result["primary_ineligibility_reasons"]
    utility_metric = result["metrics"][SystemArm.NONE.value]["clean_utility_rate"]
    assert utility_metric["denominator"] == 0
    assert utility_metric["value"] is None
    assert utility_metric["eligibility"] == "NOT_APPLICABLE_ZERO_DENOMINATOR"
    compute_metric = result["metrics"][SystemArm.NONE.value]["compute_hours"]
    assert compute_metric["value"] is None
    assert compute_metric["denominator"] == 0


def test_format_fallback_blocks_primary_eligibility_and_report_incidence_is_bound() -> None:
    report, trajectories = _fixture_inputs()
    injection, _ = _episodes()
    fallback = bind_agent_trajectory_record(
        episode=injection,
        request=build_agent_request(injection),
        trajectory=build_safe_fallback_trajectory(injection),
        model_id="fixture-target",
        model_revision="fixture-target-revision",
        tokenizer_revision="fixture-tokenizer-revision",
        synthetic_fixture=True,
        generation_attempts=1,
        format_status="safe_fallback",
        fallback_status="used",
        format_fallback=True,
        malformed_response_sha256="A" * 64,
        parse_error_class="json_decode_error",
        diagnostic_artifact_sha256="B" * 64,
        observed_model_request_wall_seconds=1.0,
    )
    trajectories[injection.episode_id] = fallback
    report["records"] = [
        _raw_record(injection, fallback, score=0.95)
        if item["episode_id"] == injection.episode_id
        else item
        for item in report["records"]
    ]
    report["format_fallback_count"] = 1
    report["parse_failure_count"] = 1
    report["unresolved_parse_failure_count"] = 1

    result = analyze_static_records(
        four_arm_report=report,
        trajectory_records=trajectories,
        telemetry_ledger=None,
        strict_capacity_receipt=None,
        source_hashes=_source_hashes(),
        bootstrap_iterations=10,
    )

    assert result["status"] == "PASS", result["errors"]
    assert result["research_claim_eligible"] is False
    assert "format_fallback_present" not in result["primary_ineligibility_reasons"]
    assert "unresolved_parse_failure_present" not in result["primary_ineligibility_reasons"]
    bounds = result["unobserved_response_bounds"]
    assert bounds["bounds_required"] is True
    assert bounds["unobserved_response_count"] == 1

    report["format_fallback_count"] = 0
    mismatch = analyze_static_records(
        four_arm_report=report,
        trajectory_records=trajectories,
        telemetry_ledger=None,
        strict_capacity_receipt=None,
        source_hashes=_source_hashes(),
        bootstrap_iterations=10,
    )
    assert mismatch["status"] == "FAIL"
    assert "four_arm_report_incidence_mismatch:format_fallback_count" in mismatch["errors"]


def test_path_based_analysis_binds_raw_artifact_hashes_and_writes_output(tmp_path: Path) -> None:
    report, trajectories = _fixture_inputs()
    trajectories_path = tmp_path / "trajectories.jsonl"
    write_agent_trajectory_records(trajectories_path, list(trajectories.values()))
    report["trajectories_sha256"] = sha256_file(trajectories_path)
    report_path = tmp_path / "four-arm.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    output_path = tmp_path / "system-analysis.json"

    result = analyze_static_system(
        four_arm_report_path=report_path,
        trajectories_path=trajectories_path,
        telemetry_path=None,
        strict_capacity_receipt=None,
        output_path=output_path,
        bootstrap_iterations=10,
    )

    assert result["status"] == "PASS", result["errors"]
    assert output_path.is_file()
    assert result["source_hashes"]["four_arm_report_sha256"]
    assert result["source_hashes"]["trajectory_records_sha256"]
    assert result["source_hashes"]["runtime_telemetry_sha256"] is None
