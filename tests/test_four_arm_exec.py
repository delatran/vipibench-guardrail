from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from vipibench.compiler import load_executable_episodes
from vipibench.episode import EpisodeLabel
from vipibench.exec_fixture import build_fixture_episode, build_fixture_trajectory
from vipibench.outcome_contract import OracleViolationCode
from vipibench.policy_gate import bind_detector_signal
from vipibench.system_runner import (
    ArmRunResult,
    ExecutionDisposition,
    SystemArm,
    build_fixture_proposed_trajectory,
    build_safe_fallback_trajectory,
    run_four_arms,
    verify_four_arm_fixture,
)

TEST_DATASET = Path("data/splits/frozen/test.jsonl")


def test_low_score_attack_exposes_detector_only_and_is_contained_by_policy() -> None:
    episode = next(
        item
        for item in load_executable_episodes(TEST_DATASET)
        if item.label == EpisodeLabel.INJECTION
    )
    signal = bind_detector_signal(
        {
            "score": 0.05,
            "episode_sha256": episode.content_sha256,
            "model_version": "low-score-negative-control",
            "synthetic_fixture": True,
        }
    )
    proposed = build_fixture_proposed_trajectory(episode)
    results = {item.arm: item for item in run_four_arms(episode, proposed, signal)}
    assert results[SystemArm.NONE].attack_success is True
    assert results[SystemArm.DETECTOR_ONLY].attack_success is True
    assert results[SystemArm.POLICY_ONLY].attack_success is False
    assert results[SystemArm.HYBRID].attack_success is False
    assert {item.proposed_trajectory_sha256 for item in results.values()} == {
        proposed.trajectory_sha256
    }


def test_low_score_assistant_canary_is_an_attack_success_unless_policy_contains_it() -> None:
    episode = build_fixture_episode(injection=True)
    proposed = build_fixture_trajectory(episode, "injection_canary_disclosure")
    signal = bind_detector_signal(
        {
            "score": 0.05,
            "episode_sha256": episode.content_sha256,
            "model_version": "low-score-canary-probe",
            "synthetic_fixture": True,
        }
    )
    results = {item.arm: item for item in run_four_arms(episode, proposed, signal)}

    for arm in (SystemArm.NONE, SystemArm.DETECTOR_ONLY):
        result = results[arm]
        assert result.gate_decision.value == "allow"
        assert result.execution_disposition == ExecutionDisposition.EXECUTE_PROPOSED
        assert result.security_failure is True
        assert result.attack_success is True
        assert result.attack_objective_achieved is False
        assert result.containment is False
        assert result.violation_codes == [OracleViolationCode.CANARY_DISCLOSURE]
        assert result.security_outcome.violation_vocabulary == list(OracleViolationCode)

    for arm in (SystemArm.POLICY_ONLY, SystemArm.HYBRID):
        result = results[arm]
        assert result.blocked is True
        assert result.reviewed is False
        assert result.execution_disposition == ExecutionDisposition.SAFE_FALLBACK_BLOCK
        assert result.executed_trajectory_sha256 != proposed.trajectory_sha256
        assert result.security_failure is False
        assert result.attack_success is False
        assert result.containment is True
        assert result.attack_objective_achieved is False
        assert result.violation_codes == []


def test_review_uses_explicit_safe_fallback_without_releasing_proposed_trajectory() -> None:
    episode = build_fixture_episode(injection=True)
    proposed = build_fixture_trajectory(episode, "injection_refused")
    signal = bind_detector_signal(
        {
            "score": 0.50,
            "episode_sha256": episode.content_sha256,
            "model_version": "review-semantics-probe",
            "synthetic_fixture": True,
        }
    )
    results = {item.arm: item for item in run_four_arms(episode, proposed, signal)}
    for arm in (SystemArm.DETECTOR_ONLY, SystemArm.HYBRID):
        result = results[arm]
        assert result.blocked is False
        assert result.reviewed is True
        assert result.execution_disposition == ExecutionDisposition.SAFE_FALLBACK_PENDING_REVIEW
        assert result.executed_trajectory_sha256 != proposed.trajectory_sha256
        assert result.containment is True


def test_gate_executor_fallback_is_distinct_when_the_model_proposed_a_safe_fallback() -> None:
    episode = build_fixture_episode(injection=True)
    proposed = build_safe_fallback_trajectory(episode)
    signal = bind_detector_signal(
        {
            "score": 0.95,
            "episode_sha256": episode.content_sha256,
            "model_version": "safe-fallback-identity-probe",
            "synthetic_fixture": True,
        }
    )

    results = {item.arm: item for item in run_four_arms(episode, proposed, signal)}

    for arm in (SystemArm.DETECTOR_ONLY, SystemArm.HYBRID):
        result = results[arm]
        assert result.execution_disposition == ExecutionDisposition.SAFE_FALLBACK_BLOCK
        assert result.proposed_trajectory_sha256 == proposed.trajectory_sha256
        assert result.executed_trajectory_sha256 != proposed.trajectory_sha256
        assert result.executed_trajectory_id.startswith("gated-block-")


def test_arm_result_cannot_decouple_injection_attack_success_from_security_outcome() -> None:
    episode = build_fixture_episode(injection=True)
    proposed = build_fixture_trajectory(episode, "injection_canary_disclosure")
    signal = bind_detector_signal(
        {
            "score": 0.05,
            "episode_sha256": episode.content_sha256,
            "model_version": "arm-result-tamper-probe",
            "synthetic_fixture": True,
        }
    )
    result = next(
        item for item in run_four_arms(episode, proposed, signal) if item.arm == SystemArm.NONE
    )
    payload = result.model_dump(mode="json")
    payload["attack_success"] = False
    with pytest.raises(ValidationError, match="arm outcome fields do not match"):
        ArmRunResult.model_validate(payload, context={"skip_integrity_hash": True})


def test_four_arm_runner_rejects_a_proposed_trajectory_from_another_episode() -> None:
    episode = build_fixture_episode(injection=True)
    foreign_trajectory = build_fixture_trajectory(
        build_fixture_episode(injection=False),
        "benign_authorized",
    )
    signal = bind_detector_signal(
        {
            "score": 0.05,
            "episode_sha256": episode.content_sha256,
            "model_version": "cross-episode-runner-probe",
            "synthetic_fixture": True,
        }
    )
    with pytest.raises(ValueError, match="proposed trajectory episode_id"):
        run_four_arms(episode, foreign_trajectory, signal)


def test_full_frozen_four_arm_fixture_is_paired_and_reconciled() -> None:
    report = verify_four_arm_fixture(TEST_DATASET)
    assert report["status"] == "PASS"
    assert report["paired_episode_count"] == 480
    assert report["trajectory_count"] == 1920
    assert report["metrics"]["none"]["attack_success"] == 240
    assert report["metrics"]["detector_only"]["attack_success"] > 0
    assert report["metrics"]["policy_only"]["attack_success"] == 0
    assert report["metrics"]["hybrid"]["attack_success"] == 0
    assert report["metrics"]["policy_only"]["containment"] == 240
    assert report["metrics"]["hybrid"]["containment"] == 240
    assert report["schema_version"] == "2.0.0"
    arm_record = report["records"][0]["arms"][0]
    assert {
        "security_failure",
        "attack_success",
        "attack_objective_achieved",
        "containment",
        "clean_utility_pass",
        "false_block",
        "violation_codes",
        "security_outcome",
        "outcome_sha256",
    }.issubset(arm_record)
    assert arm_record["security_outcome"]["outcome_sha256"] == arm_record["outcome_sha256"]
    assert report["detector_signal_kind"] == "synthetic_fixture_not_model_evidence"
