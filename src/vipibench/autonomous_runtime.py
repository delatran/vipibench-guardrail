from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

POLICY_SCHEMA_VERSION = "1.0.0"
MAXIMUM_ALLOWED_SESSION_HOURS = 72.0


@dataclass(frozen=True)
class AutonomousExecutionPolicy:
    hard_ceiling_hours: float
    resource_observation_interval_seconds: float
    snapshot_interval_seconds: float
    snapshot_profile: str
    snapshot_stop_timeout_seconds: float
    timeout_action: str
    capacity_selection_source: str
    capacity_selection_metric: str
    fallback_outside_locked_candidate_ladders: bool
    decision_split: str
    decision_metric: str
    training_parameter_source: str
    maximum_epochs_role: str
    final_holdout_feedback_allowed: bool
    resume_mode: str
    stale_checkpoint_action: str
    source_path: str

    @property
    def timeout_seconds(self) -> float:
        return self.hard_ceiling_hours * 3600

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def load_autonomous_execution_policy(path: Path) -> AutonomousExecutionPolicy:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("autonomous execution policy must be a JSON object")
    errors = validate_autonomous_execution_policy(payload)
    if errors:
        raise ValueError(errors)

    session = _mapping(payload, "session")
    capacity = _mapping(payload, "capacity")
    training_stop = _mapping(payload, "training_stop")
    resume = _mapping(payload, "resume")
    return AutonomousExecutionPolicy(
        hard_ceiling_hours=float(session["hard_ceiling_hours"]),
        resource_observation_interval_seconds=float(
            session["resource_observation_interval_seconds"]
        ),
        snapshot_interval_seconds=float(session["snapshot_interval_seconds"]),
        snapshot_profile=str(session["snapshot_profile"]),
        snapshot_stop_timeout_seconds=float(session["snapshot_stop_timeout_seconds"]),
        timeout_action=str(session["timeout_action"]),
        capacity_selection_source=str(capacity["selection_source"]),
        capacity_selection_metric=str(capacity["selection_metric"]),
        fallback_outside_locked_candidate_ladders=bool(
            capacity["fallback_outside_locked_candidate_ladders"]
        ),
        decision_split=str(training_stop["decision_split"]),
        decision_metric=str(training_stop["decision_metric"]),
        training_parameter_source=str(training_stop["parameter_source"]),
        maximum_epochs_role=str(training_stop["maximum_epochs_role"]),
        final_holdout_feedback_allowed=bool(
            training_stop["final_holdout_feedback_allowed"]
        ),
        resume_mode=str(resume["mode"]),
        stale_checkpoint_action=str(resume["stale_or_mismatched_checkpoint_action"]),
        source_path=str(path),
    )


def validate_autonomous_execution_policy(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("schema_version") != POLICY_SCHEMA_VERSION:
        errors.append("schema_version_mismatch")
    if payload.get("status") != "locked_protocol":
        errors.append("policy_not_locked")
    if payload.get("mode") != "autonomous":
        errors.append("mode_must_be_autonomous")

    session = _mapping_or_error(payload, "session", errors)
    capacity = _mapping_or_error(payload, "capacity", errors)
    training_stop = _mapping_or_error(payload, "training_stop", errors)
    resume = _mapping_or_error(payload, "resume", errors)

    hours = _positive_float(session.get("hard_ceiling_hours"), "hard_ceiling_hours", errors)
    if hours is not None and hours > MAXIMUM_ALLOWED_SESSION_HOURS:
        errors.append("hard_ceiling_hours_above_safety_limit")
    resource_interval = _positive_float(
        session.get("resource_observation_interval_seconds"),
        "resource_observation_interval_seconds",
        errors,
    )
    if resource_interval is not None and not 5 <= resource_interval <= 60:
        errors.append("resource_observation_interval_seconds_outside_5_60")
    interval = _positive_float(
        session.get("snapshot_interval_seconds"),
        "snapshot_interval_seconds",
        errors,
    )
    if interval is not None and not 5 <= interval <= 600:
        errors.append("snapshot_interval_seconds_outside_5_600")
    if session.get("snapshot_profile") != "confirmatory_resume_v1":
        errors.append("snapshot_profile_mismatch")
    stop_timeout = _positive_float(
        session.get("snapshot_stop_timeout_seconds"),
        "snapshot_stop_timeout_seconds",
        errors,
    )
    if stop_timeout is not None and not 30 <= stop_timeout <= 600:
        errors.append("snapshot_stop_timeout_seconds_outside_30_600")
    if session.get("timeout_action") != (
        "final_atomic_snapshot_then_verified_checkpoint_resume"
    ):
        errors.append("timeout_action_mismatch")

    if capacity.get("selection_source") != "bounded_live_warmup":
        errors.append("capacity_selection_source_mismatch")
    if capacity.get("selection_metric") != (
        "maximum_throughput_within_memory_reserve"
    ):
        errors.append("capacity_selection_metric_mismatch")
    if capacity.get("fallback_outside_locked_candidate_ladders") is not False:
        errors.append("capacity_fallback_must_be_false")

    if training_stop.get("decision_split") != "dev":
        errors.append("training_stop_split_must_be_dev")
    if training_stop.get("decision_metric") != "dev_auprc":
        errors.append("training_stop_metric_must_be_dev_auprc")
    if training_stop.get("parameter_source") != "configs/models/mdeberta_core.yaml":
        errors.append("training_parameter_source_mismatch")
    if training_stop.get("maximum_epochs_role") != "safety_ceiling":
        errors.append("maximum_epochs_role_mismatch")
    if training_stop.get("final_holdout_feedback_allowed") is not False:
        errors.append("final_holdout_feedback_must_be_false")

    if resume.get("mode") != "verified_checkpoint_only":
        errors.append("resume_mode_mismatch")
    if resume.get("stale_or_mismatched_checkpoint_action") != "fail_closed":
        errors.append("stale_checkpoint_action_mismatch")
    return errors


def _mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload[key]
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be an object")
    return value


def _mapping_or_error(
    payload: dict[str, Any],
    key: str,
    errors: list[str],
) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        errors.append(f"{key}_must_be_object")
        return {}
    return value


def _positive_float(
    value: object,
    label: str,
    errors: list[str],
) -> float | None:
    if isinstance(value, bool):
        errors.append(f"{label}_must_be_positive_number")
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        errors.append(f"{label}_must_be_positive_number")
        return None
    if parsed <= 0:
        errors.append(f"{label}_must_be_positive_number")
        return None
    return parsed
