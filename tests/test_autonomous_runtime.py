import json
from copy import deepcopy
from pathlib import Path

from vipibench.autonomous_runtime import (
    load_autonomous_execution_policy,
    validate_autonomous_execution_policy,
)

POLICY_PATH = Path("configs/resources/autonomous_execution.json")


def _policy() -> dict[str, object]:
    value = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_locked_autonomous_execution_policy_passes() -> None:
    policy = load_autonomous_execution_policy(POLICY_PATH)
    assert policy.hard_ceiling_hours == 72
    assert policy.timeout_seconds == 72 * 3600
    assert policy.resource_observation_interval_seconds == 15
    assert policy.snapshot_interval_seconds == 300
    assert policy.snapshot_profile == "confirmatory_resume_v1"
    assert policy.snapshot_stop_timeout_seconds == 300
    assert policy.decision_split == "dev"
    assert policy.decision_metric == "dev_auprc"
    assert policy.final_holdout_feedback_allowed is False
    assert policy.resume_mode == "verified_checkpoint_only"


def test_policy_rejects_final_holdout_feedback() -> None:
    payload = deepcopy(_policy())
    training_stop = payload["training_stop"]
    assert isinstance(training_stop, dict)
    training_stop["final_holdout_feedback_allowed"] = True
    assert "final_holdout_feedback_must_be_false" in validate_autonomous_execution_policy(
        payload
    )


def test_policy_rejects_unbounded_session() -> None:
    payload = deepcopy(_policy())
    session = payload["session"]
    assert isinstance(session, dict)
    session["hard_ceiling_hours"] = 72.1
    assert "hard_ceiling_hours_above_safety_limit" in validate_autonomous_execution_policy(
        payload
    )


def test_policy_rejects_sparse_resource_observation() -> None:
    payload = deepcopy(_policy())
    session = payload["session"]
    assert isinstance(session, dict)
    session["resource_observation_interval_seconds"] = 61
    assert (
        "resource_observation_interval_seconds_outside_5_60"
        in validate_autonomous_execution_policy(payload)
    )


def test_policy_rejects_non_development_stop_signal() -> None:
    payload = deepcopy(_policy())
    training_stop = payload["training_stop"]
    assert isinstance(training_stop, dict)
    training_stop["decision_split"] = "test"
    assert "training_stop_split_must_be_dev" in validate_autonomous_execution_policy(payload)
