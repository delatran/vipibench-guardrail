from __future__ import annotations

import gc
import hashlib
import json
import os
import statistics
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vipibench.agent_trajectory import (
    AGENT_TRAJECTORY_SCHEMA_VERSION,
    TIMING_SCHEMA_VERSION,
    AgentTrajectoryRecord,
    bind_agent_trajectory_record,
    build_agent_request,
    parse_agent_response,
    render_agent_prompt,
    text_sha256,
    write_agent_trajectory_records,
    write_malformed_response_diagnostic,
)
from vipibench.dataio import canonical_json, sha256_file, write_json
from vipibench.exec_detector_data import load_executable_episodes
from vipibench.modeling import load_yaml
from vipibench.oracle import (
    TRAJECTORY_EVENT_ID_MAX_LENGTH,
    TRAJECTORY_SCHEMA_VERSION,
)
from vipibench.runtime_capacity import (
    CapacityMeasurement,
    check_runtime_profile_path,
    is_capacity_exhaustion,
    rank_capacity_candidates,
    select_capacity_candidate,
    validate_model_device_placement,
)
from vipibench.runtime_telemetry import (
    build_strict_capacity_receipt,
    record_stage_interval,
    strict_capacity_receipt_sha256,
    write_telemetry_ledger,
)
from vipibench.system_runner import build_safe_fallback_trajectory

TARGET_CHECKPOINT_SCHEMA_VERSION = "5.0.0"
TARGET_CHECKPOINT_BINDING_SCHEMA_VERSION = "5.0.0"
TARGET_FALLBACK_LEDGER_SCHEMA_VERSION = "1.0.0"
TARGET_FALLBACK_LEDGER_FILENAME = "final_holdout_format_fallback_ledger.json"
TARGET_MODEL_ID = "Qwen/Qwen3-8B"
TARGET_MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
TARGET_MAX_NEW_TOKENS = 4096
TARGET_TRUNCATED_RESPONSE_POLICY = "reject_truncated_response_without_delimiter_repair_v1"
TARGET_TRUNCATED_RESPONSE_ERROR_CLASS = "response_truncation_error"
TARGET_UNUSABLE_RESPONSE_POLICY = "bounded_unusable_observation_two_sided_bounds_v1"
TARGET_MAX_UNUSABLE_RESPONSE_FRACTION = 0.01
TARGET_RUNNER_RESUME_COMPATIBLE_SHA256 = frozenset(
    {
        # Pre production-OOM recovery hotfix on the V8 confirmatory lineage.
        "19059DBC6BAB8798D21681D2CB208175FBBC13EDB7DE702AD49EB24C11B86297",
    }
)
TARGET_CHECKPOINT_BINDING_FIELDS = {
    "binding_schema_version",
    "trajectory_schema_version",
    "timing_schema_version",
    "config_sha256",
    "target_runner_sha256",
    "agent_trajectory_sha256",
    "oracle_sha256",
    "oracle_trajectory_schema_version",
    "model_id",
    "model_revision",
    "tokenizer_revision",
    "response_format",
    "max_format_attempts",
    "format_repair_policy",
    "event_id_max_length",
}
TARGET_FALLBACK_LEDGER_ENTRY_FIELDS = {
    "episode_id",
    "episode_sha256",
    "prompt_sha256",
    "record_sha256",
    "request_sha256",
}


@dataclass(frozen=True)
class GeneratedBatch:
    responses: list[str]
    input_token_counts: list[int | None]
    output_token_counts: list[int | None]


def _reached_response_token_ceiling(
    record: AgentTrajectoryRecord,
    max_new_tokens: int | None,
) -> bool:
    """Report whether decoding stopped at the configured response ceiling."""

    if max_new_tokens is None or record.output_token_count is None:
        return False
    return record.output_token_count >= max_new_tokens


def unusable_observation_budget(
    total_episode_count: int,
    fraction: float = TARGET_MAX_UNUSABLE_RESPONSE_FRACTION,
) -> int:
    """Return how many unobserved responses one run may carry before it fails closed.

    The budget stops a systemically broken run early. It is not the eligibility
    rule: whether bounded missing observations can still support a claim is
    decided later by two-sided bounds over the recorded outcomes.
    """

    if total_episode_count < 0:
        raise ValueError("total_episode_count must not be negative")
    if not 0.0 <= fraction < 1.0:
        raise ValueError("unusable-response fraction must fall within [0, 1)")
    return int(total_episode_count * fraction)


def _summarize_target_format_failures(
    records: list[AgentTrajectoryRecord],
    *,
    max_new_tokens: int | None = None,
    total_episode_count: int | None = None,
    unusable_fraction: float = TARGET_MAX_UNUSABLE_RESPONSE_FRACTION,
) -> dict[str, object]:
    """Return a raw-content-free validity disposition for one target run.

    A parse error is an observed formatting incident, but it is not unresolved
    when exactly one one-character terminal-delimiter candidate passes the same
    strict parser. Only a safe fallback leaves the response unobserved.

    A response whose decoding stopped at the configured ceiling is an
    incomplete observation rather than a formatting incident, so it is counted
    separately and is never eligible for terminal-delimiter recovery.

    An unobserved response no longer fails the run by itself. It is retained,
    counted, and published so that the analysis stage can bound its effect from
    both directions. The run fails closed only when unobserved responses exceed
    the registered budget, because that indicates systemic breakage rather than
    a rare incident.
    """

    truncated_response_episode_ids = sorted(
        record.episode_id
        for record in records
        if record.parse_error_class == TARGET_TRUNCATED_RESPONSE_ERROR_CLASS
    )
    response_token_ceiling_episode_ids = sorted(
        record.episode_id
        for record in records
        if _reached_response_token_ceiling(record, max_new_tokens)
    )
    format_fallback_episode_ids = sorted(
        record.episode_id for record in records if record.format_fallback
    )
    format_repair_episode_ids = sorted(
        record.episode_id for record in records if record.format_status == "repaired_json"
    )
    parse_failure_episode_ids = sorted(
        record.episode_id for record in records if record.parse_error_class is not None
    )
    unresolved_parse_failure_episode_ids = sorted(
        record.episode_id for record in records if record.format_status == "safe_fallback"
    )
    error_class_counts = Counter(
        record.parse_error_class
        for record in records
        if record.parse_error_class is not None
    )
    unusable_observation_episode_ids = sorted(
        set(format_fallback_episode_ids) | set(unresolved_parse_failure_episode_ids)
    )
    budget = (
        unusable_observation_budget(total_episode_count, unusable_fraction)
        if total_episode_count is not None
        else 0
    )
    over_budget = len(unusable_observation_episode_ids) > budget
    errors: list[str] = []
    if over_budget:
        errors.append("unobserved_response_budget_exceeded")
    return {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "unusable_observation_count": len(unusable_observation_episode_ids),
        "unusable_observation_episode_ids": unusable_observation_episode_ids,
        "unusable_observation_budget": budget,
        "unusable_observation_fraction_ceiling": unusable_fraction,
        "unusable_observation_policy": TARGET_UNUSABLE_RESPONSE_POLICY,
        "unusable_observation_budget_exceeded": over_budget,
        "truncated_response_count": len(truncated_response_episode_ids),
        "truncated_response_episode_ids": truncated_response_episode_ids,
        "response_token_ceiling": max_new_tokens,
        "response_token_ceiling_reached_count": len(response_token_ceiling_episode_ids),
        "response_token_ceiling_episode_ids": response_token_ceiling_episode_ids,
        "format_fallback_count": len(format_fallback_episode_ids),
        "format_fallback_episode_ids": format_fallback_episode_ids,
        "format_repair_count": len(format_repair_episode_ids),
        "format_repair_episode_ids": format_repair_episode_ids,
        "parse_failure_count": len(parse_failure_episode_ids),
        "parse_failure_episode_ids": parse_failure_episode_ids,
        "unresolved_parse_failure_count": len(unresolved_parse_failure_episode_ids),
        "unresolved_parse_failure_episode_ids": unresolved_parse_failure_episode_ids,
        "parse_error_class_counts": dict(sorted(error_class_counts.items())),
        "raw_response_included": False,
    }


def _write_target_format_abort_result(
    *,
    config_path: Path,
    dataset_path: Path,
    output_path: Path,
    checkpoint_dir: Path,
    checkpoint_binding: dict[str, str],
    records: list[AgentTrajectoryRecord],
    total_episode_count: int,
    max_new_tokens: int,
    unusable_fraction: float = TARGET_MAX_UNUSABLE_RESPONSE_FRACTION,
    trigger_phase: str,
    trigger_batch_id: str | None,
    runtime: dict[str, object],
    strict_capacity_receipt: dict[str, object],
    strict_capacity_receipt_path: Path | None,
    capacity: dict[str, object] | None,
    model_placement: dict[str, object] | None,
) -> dict[str, object]:
    """Persist a safe failure receipt and forbid every later target batch."""

    by_id = {record.episode_id: record for record in records}
    if len(by_id) != len(records):
        raise ValueError("target format abort records contain duplicate episodes")
    if len(by_id) > total_episode_count:
        raise ValueError("target format abort record count exceeds the dataset")
    ordered_records = sorted(records, key=lambda record: record.episode_id)
    summary = _summarize_target_format_failures(
        ordered_records,
        max_new_tokens=max_new_tokens,
        total_episode_count=total_episode_count,
        unusable_fraction=unusable_fraction,
    )
    if summary["status"] != "FAIL":
        raise ValueError("target format abort requires the unobserved-response budget to be spent")
    fallback_episode_ids = summary["format_fallback_episode_ids"]
    parse_failure_episode_ids = summary["parse_failure_episode_ids"]
    unresolved_parse_failure_episode_ids = summary["unresolved_parse_failure_episode_ids"]
    if not isinstance(fallback_episode_ids, list) or not isinstance(
        parse_failure_episode_ids, list
    ) or not isinstance(unresolved_parse_failure_episode_ids, list):
        raise RuntimeError("target format abort summary has invalid episode lists")

    fallback_ledger_path = _final_holdout_fallback_ledger_path(checkpoint_dir)
    result = {
        "schema_version": "2.5.0",
        "status": "FAIL",
        "errors": summary["errors"],
        "config_sha256": sha256_file(config_path),
        "dataset_sha256": sha256_file(dataset_path),
        "model_id": checkpoint_binding["model_id"],
        "model_revision": checkpoint_binding["model_revision"],
        "tokenizer_revision": checkpoint_binding["tokenizer_revision"],
        "checkpoint_binding": checkpoint_binding,
        "runtime": runtime,
        "strict_runtime_capacity_receipt": strict_capacity_receipt,
        "strict_runtime_capacity_receipt_source": (
            str(strict_capacity_receipt_path)
            if strict_capacity_receipt_path is not None
            else None
        ),
        "runtime_telemetry": None,
        "model_device_placement": model_placement,
        "capacity_plan": capacity,
        "trajectory_artifact": None,
        "truncated_response_count": summary["truncated_response_count"],
        "truncated_response_episode_ids": summary["truncated_response_episode_ids"],
        "response_token_ceiling": summary["response_token_ceiling"],
        "response_token_ceiling_reached_count": summary["response_token_ceiling_reached_count"],
        "format_fallback_count": summary["format_fallback_count"],
        "format_fallback_episode_ids": fallback_episode_ids,
        "format_repair_count": summary["format_repair_count"],
        "format_repair_episode_ids": summary["format_repair_episode_ids"],
        "parse_failure_count": summary["parse_failure_count"],
        "parse_failure_episode_ids": parse_failure_episode_ids,
        "unresolved_parse_failure_count": summary["unresolved_parse_failure_count"],
        "unresolved_parse_failure_episode_ids": unresolved_parse_failure_episode_ids,
        "format_failure_summary": summary,
        "synthetic_fixture_count": sum(record.synthetic_fixture for record in records),
        "gold_blind_request_contract": True,
        "target_format_fail_fast": {
            "policy": "abort_when_unobserved_response_budget_is_exceeded",
            "trigger_phase": trigger_phase,
            "trigger_batch_id": trigger_batch_id,
            "recorded_episode_count": len(by_id),
            "total_episode_count": total_episode_count,
            "unprocessed_episode_count": total_episode_count - len(by_id),
            "additional_model_batches_after_trigger": 0,
            "normal_trajectory_artifact_written": False,
            "raw_response_included": False,
            "truncated_response_policy": TARGET_TRUNCATED_RESPONSE_POLICY,
            "response_token_ceiling": max_new_tokens,
            "truncated_response_count": summary["truncated_response_count"],
            "unusable_observation_policy": TARGET_UNUSABLE_RESPONSE_POLICY,
            "unusable_observation_count": summary["unusable_observation_count"],
            "unusable_observation_budget": summary["unusable_observation_budget"],
        },
        "unusable_observation_count": summary["unusable_observation_count"],
        "unusable_observation_episode_ids": summary["unusable_observation_episode_ids"],
        "unusable_observation_budget": summary["unusable_observation_budget"],
        "unusable_observation_policy": TARGET_UNUSABLE_RESPONSE_POLICY,
        "claim_dispositions": {
            claim: "INCONCLUSIVE_UNOBSERVED_RESPONSE_BUDGET_EXCEEDED"
            for claim in ("RQ3", "H3", "H4")
        },
        "final_holdout_resume_contract": {
            "policy": "fail_closed_reuse_of_recorded_format_failures",
            "fallback_ledger_path": str(fallback_ledger_path),
            "fallback_ledger_sha256": (
                sha256_file(fallback_ledger_path) if fallback_ledger_path.is_file() else None
            ),
            "replacement_without_separately_versioned_exploratory_run": "forbidden",
        },
        "claim_boundary": (
            "FAIL proves that responses which were never observed exceeded the registered "
            "budget, so the run stopped before another target batch. At that density the "
            "missing outcomes can no longer be bounded tightly enough to leave any conclusion "
            "standing, which is a different situation from the rare incident the budget allows. "
            "A truncated response is an unobserved completion caused by the registered response "
            "ceiling, not a model formatting defect. The normal trajectory artifact is "
            "intentionally absent; raw diagnostics remain checkpoint-bound and no research "
            "conclusion is eligible."
        ),
    }
    write_json(output_path.with_suffix(".run.json"), result)
    return result


def _positive_integer_or_none(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def validate_target_protocol(config_path: Path) -> dict[str, object]:
    config = load_yaml(config_path)
    errors: list[str] = []
    required = {
        "model_id",
        "model_revision",
        "tokenizer_revision",
        "runtime_profile",
        "batch_candidates",
        "target_memory_utilization",
        "max_input_tokens",
        "max_new_tokens",
        "max_format_attempts",
        "format_repair_policy",
        "truncated_response_policy",
        "unusable_response_policy",
        "max_unusable_response_fraction",
        "event_id_max_length",
        "capacity_probe_new_tokens",
        "capacity_warmup_batches",
        "capacity_measurement_repeats",
        "capacity_reference_batch_size",
        "capacity_validation_new_tokens",
        "capacity_validation_repeats",
        "capacity_stop_after_oom",
        "input_batch_order",
        "repair_batching",
        "use_cache",
    }
    missing = sorted(required - set(config))
    if missing:
        errors.append(f"missing_fields:{','.join(missing)}")
    if config.get("schema_version") != "2.4.0":
        errors.append("schema_version_must_equal_2_4_0")
    if config.get("status") != "locked_protocol":
        errors.append("protocol_not_locked")
    if config.get("model_id") != TARGET_MODEL_ID:
        errors.append("target_model_id_mismatch")
    if config.get("model_revision") != TARGET_MODEL_REVISION:
        errors.append("target_model_revision_mismatch")
    if config.get("model_revision") != config.get("tokenizer_revision"):
        errors.append("model_tokenizer_revision_mismatch")
    if config.get("runtime_profile") != "configs/profiles/accelerator_80gb.yaml":
        errors.append("target_runtime_profile_mismatch")
    if config.get("precision") != "bf16":
        errors.append("target_precision_must_be_bf16")
    if config.get("device_placement") != "auto":
        errors.append("target_device_placement_must_be_auto")
    candidates = config.get("batch_candidates")
    if not isinstance(candidates, list) or not candidates:
        errors.append("batch_candidates_missing")
    elif any(not isinstance(value, int) or value <= 0 for value in candidates):
        errors.append("batch_candidates_invalid")
    elif candidates != [8, 16, 24, 32, 48, 64]:
        errors.append("batch_candidates_must_equal_8_16_24_32_48_64")
    if (
        "max_format_attempts" in config
        and _positive_integer_or_none(config["max_format_attempts"]) is None
    ):
        errors.append("max_format_attempts_invalid")
    if float(config.get("target_memory_utilization", 0)) != 0.88:
        errors.append("target_memory_utilization_must_equal_0_88")
    if int(config.get("max_input_tokens", 0)) != 4096:
        errors.append("max_input_tokens_must_equal_4096")
    if int(config.get("max_new_tokens", 0)) != TARGET_MAX_NEW_TOKENS:
        errors.append(f"max_new_tokens_must_equal_{TARGET_MAX_NEW_TOKENS}")
    if config.get("max_format_attempts") != 1:
        errors.append("max_format_attempts_must_equal_1")
    if config.get("event_id_max_length") != TRAJECTORY_EVENT_ID_MAX_LENGTH:
        errors.append("event_id_max_length_must_equal_96")
    if (
        config.get("format_repair_policy")
        != "unique_single_terminal_delimiter_recovery_v1"
    ):
        errors.append(
            "format_repair_policy_must_equal_"
            "unique_single_terminal_delimiter_recovery_v1"
        )
    if config.get("truncated_response_policy") != TARGET_TRUNCATED_RESPONSE_POLICY:
        errors.append(f"truncated_response_policy_must_equal_{TARGET_TRUNCATED_RESPONSE_POLICY}")
    if config.get("unusable_response_policy") != TARGET_UNUSABLE_RESPONSE_POLICY:
        errors.append(f"unusable_response_policy_must_equal_{TARGET_UNUSABLE_RESPONSE_POLICY}")
    if config.get("max_unusable_response_fraction") != TARGET_MAX_UNUSABLE_RESPONSE_FRACTION:
        errors.append(
            "max_unusable_response_fraction_must_equal_"
            f"{TARGET_MAX_UNUSABLE_RESPONSE_FRACTION}"
        )
    if int(config.get("capacity_probe_new_tokens", 0)) != 128:
        errors.append("capacity_probe_new_tokens_must_equal_128")
    if int(config.get("capacity_warmup_batches", 0)) != 1:
        errors.append("capacity_warmup_batches_must_equal_1")
    if int(config.get("capacity_measurement_repeats", 0)) != 3:
        errors.append("capacity_measurement_repeats_must_equal_3")
    if int(config.get("capacity_reference_batch_size", 0)) != 8:
        errors.append("capacity_reference_batch_size_must_equal_8")
    if int(config.get("capacity_validation_new_tokens", 0)) != int(
        config.get("max_new_tokens", -1)
    ):
        errors.append("capacity_validation_must_use_production_new_tokens")
    if int(config.get("capacity_validation_repeats", 0)) != 1:
        errors.append("capacity_validation_repeats_must_equal_1")
    if config.get("capacity_stop_after_oom") is not True:
        errors.append("capacity_stop_after_oom_must_be_true")
    if config.get("input_batch_order") != "stable_token_length_descending":
        errors.append("input_batch_order_must_be_stable_token_length_descending")
    if config.get("repair_batching") != "not_applicable":
        errors.append("repair_batching_must_be_not_applicable")
    if config.get("use_cache") is not True:
        errors.append("target_generation_cache_must_be_enabled")
    if config.get("response_format") != "strict_trajectory_json":
        errors.append("target_response_format_mismatch")
    if config.get("do_sample") is not False:
        errors.append("confirmatory_target_must_be_deterministic")
    if config.get("enable_thinking") is not False:
        errors.append("strict_json_requires_thinking_disabled")
    if config.get("trust_remote_code") is not False:
        errors.append("remote_code_must_be_disabled")
    if config.get("allow_model_revision_fallback") is not False:
        errors.append("model_revision_fallback_forbidden")
    if config.get("allow_host_fallback") is not False:
        errors.append("host_fallback_forbidden")
    return {
        "schema_version": "1.0.0",
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
    }


def run_target_agent(
    *,
    config_path: Path,
    dataset_path: Path,
    output_path: Path,
    checkpoint_dir: Path,
    strict_capacity_receipt_path: Path | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, object]:
    if os.environ.get("VIPIBENCH_CONFIRMATORY_RUN_APPROVED") != "YES":
        raise PermissionError(
            "VIPIBENCH_CONFIRMATORY_RUN_APPROVED=YES is required for confirmatory execution"
        )
    protocol = validate_target_protocol(config_path)
    if protocol["status"] != "PASS":
        raise ValueError(protocol["errors"])
    config = load_yaml(config_path)
    profile_path = Path(str(config["runtime_profile"]))
    runtime = check_runtime_profile_path(profile_path, dataset_path.parent)
    if runtime["status"] != "PASS" or runtime["hardware_observed"] is not True:
        raise RuntimeError(runtime["errors"])
    project_root = config_path.resolve().parents[2]
    observed_receipt = build_strict_capacity_receipt(
        runtime,
        project_root=project_root,
    )
    strict_capacity_receipt = observed_receipt
    if strict_capacity_receipt_path is not None:
        if not strict_capacity_receipt_path.is_file():
            raise FileNotFoundError(strict_capacity_receipt_path)
        supplied_receipt = json.loads(strict_capacity_receipt_path.read_text(encoding="utf-8"))
        if not isinstance(supplied_receipt, dict):
            raise ValueError("strict capacity receipt must be an object")
        _validate_shared_receipt_matches_observed_runtime(supplied_receipt, runtime)
        strict_capacity_receipt = supplied_receipt
    strict_capacity_receipt_hash = strict_capacity_receipt_sha256(
        strict_capacity_receipt,
        project_root=project_root,
    )
    stage_started = time.perf_counter()

    episodes = load_executable_episodes(dataset_path)
    prompts = [render_agent_prompt(build_agent_request(episode)) for episode in episodes]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    records: list[AgentTrajectoryRecord] = []
    pending: list[tuple[Any, str]] = []
    model_id = str(config["model_id"])
    model_revision = str(config["model_revision"])
    tokenizer_revision = str(config["tokenizer_revision"])
    checkpoint_binding = _target_checkpoint_binding(config_path)
    dataset_sha256 = sha256_file(dataset_path)
    fallback_ledger = _load_final_holdout_fallback_ledger(
        checkpoint_dir,
        checkpoint_binding,
        dataset_sha256,
    )
    episode_ids = {episode.episode_id for episode in episodes}
    orphaned_fallbacks = sorted(set(fallback_ledger) - episode_ids)
    if orphaned_fallbacks:
        raise ValueError(
            "final-holdout format fallback ledger contains episodes outside the frozen dataset: "
            + ",".join(orphaned_fallbacks)
        )
    for episode, prompt in zip(episodes, prompts, strict=True):
        checkpoint = checkpoint_dir / f"{episode.episode_id}.json"
        request = build_agent_request(episode)
        prompt_sha256 = text_sha256(prompt)
        fallback_entry = fallback_ledger.get(episode.episode_id)
        if checkpoint.is_file():
            record = _load_target_checkpoint(
                checkpoint,
                checkpoint_binding,
                expected_record_binding={
                    "episode_sha256": episode.content_sha256,
                    "request_sha256": request.request_sha256,
                    "prompt_sha256": prompt_sha256,
                },
            )
            if (
                record.model_id != model_id
                or record.model_revision != model_revision
                or record.tokenizer_revision != tokenizer_revision
            ):
                raise ValueError(f"checkpoint binding mismatch: {episode.episode_id}")
            if record.format_fallback:
                _validate_final_holdout_fallback_entry(record, fallback_entry)
            elif fallback_entry is not None:
                raise ValueError(
                    "final-holdout fallback ledger conflicts with a non-fallback checkpoint: "
                    f"{episode.episode_id}"
                )
            records.append(record)
        else:
            if fallback_entry is not None:
                raise ValueError(
                    "final-holdout fallback checkpoint is missing and cannot be regenerated: "
                    f"{episode.episode_id}; create a separately versioned exploratory run "
                    "instead of replacing the final-holdout observation"
                )
            pending.append((episode, prompt))

    # Reject a resumed run whose retained records already spent the unobserved-response
    # budget, before loading a model or performing a capacity probe. A successfully
    # recovered record is immutable and reusable; a retained unobserved response is
    # never replaced by a second generation.
    response_token_ceiling = int(config["max_new_tokens"])
    unusable_fraction = float(config["max_unusable_response_fraction"])
    resume_failure_summary = _summarize_target_format_failures(
        records,
        max_new_tokens=response_token_ceiling,
        total_episode_count=len(episodes),
        unusable_fraction=unusable_fraction,
    )
    if resume_failure_summary["status"] == "FAIL":
        return _write_target_format_abort_result(
            config_path=config_path,
            dataset_path=dataset_path,
            output_path=output_path,
            checkpoint_dir=checkpoint_dir,
            checkpoint_binding=checkpoint_binding,
            records=records,
            total_episode_count=len(episodes),
            max_new_tokens=response_token_ceiling,
            unusable_fraction=unusable_fraction,
            trigger_phase="resume_pre_model_load",
            trigger_batch_id=None,
            runtime=runtime,
            strict_capacity_receipt=strict_capacity_receipt,
            strict_capacity_receipt_path=strict_capacity_receipt_path,
            capacity=None,
            model_placement=None,
        )

    torch, tokenizer, model, model_placement = _load_model(config)
    capacity = _measure_capacity(torch, tokenizer, model, prompts, config)
    if capacity["status"] != "PASS":
        raise RuntimeError(capacity["errors"])
    pending, input_batching = _order_pending_by_input_length(tokenizer, pending, config)
    capacity["input_batching"] = input_batching
    validation_prompts = [prompt for _, prompt in pending] or prompts
    capacity, validated_generation, validated_wall_seconds = _validate_production_capacity(
        torch,
        tokenizer,
        model,
        validation_prompts,
        config,
        capacity,
    )
    if capacity["status"] != "PASS":
        raise RuntimeError(capacity["errors"])
    selected = capacity["selected"]
    if not isinstance(selected, dict):
        raise RuntimeError("capacity selection produced no candidate")
    batch_size = int(selected["batch_size"])
    capacity_checkpoint_path = checkpoint_dir / "_capacity_plan.json"
    write_json(
        capacity_checkpoint_path,
        {
            "schema_version": "1.0.0",
            "status": "PASS",
            "config_sha256": sha256_file(config_path),
            "dataset_sha256": dataset_sha256,
            "checkpoint_binding": checkpoint_binding,
            "capacity_plan": capacity,
        },
    )

    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        batch_id = f"target-batch-{start // batch_size + 1:06d}"
        if start == 0:
            generated = validated_generation
            initial_wall_seconds = validated_wall_seconds
            if len(generated.responses) != len(batch):
                raise RuntimeError("production validation batch does not match scheduled batch")
        else:
            generated, initial_wall_seconds = _generate_batch_resilient(
                torch,
                tokenizer,
                model,
                [prompt for _, prompt in batch],
                config,
                max_new_tokens=int(config["max_new_tokens"]),
                clock=clock,
            )
        batch_records = _parse_batch_with_repairs(
            batch,
            generated,
            config,
            batch_id=batch_id,
            batch_size=len(batch),
            initial_model_request_wall_seconds=[initial_wall_seconds / len(batch)] * len(batch),
            diagnostic_dir=checkpoint_dir / "_nonpublic_diagnostics",
        )
        for (episode, _), record in zip(batch, batch_records, strict=True):
            checkpoint = checkpoint_dir / f"{episode.episode_id}.json"
            if record.format_fallback:
                # Persist the immutable fallback fact before its checkpoint. A crash between
                # these writes is intentionally fail-closed on resume rather than allowing a
                # second final-holdout generation call to replace the fallback observation.
                fallback_ledger = _record_final_holdout_fallback(
                    checkpoint_dir,
                    checkpoint_binding,
                    dataset_sha256,
                    fallback_ledger,
                    record,
                )
            _write_target_checkpoint(checkpoint, checkpoint_binding, record)
            records.append(record)

        # The production-validation generation is reused as the first scheduled
        # batch, so this gate adds no model call. A unique strict-valid terminal-
        # delimiter recovery is retained under the registered deterministic policy.
        # An unobserved response is retained and counted rather than ending the
        # run; only exceeding the registered budget stops the next batch.
        batch_failure_summary = _summarize_target_format_failures(
            records,
            max_new_tokens=response_token_ceiling,
            total_episode_count=len(episodes),
            unusable_fraction=unusable_fraction,
        )
        if batch_failure_summary["status"] == "FAIL":
            return _write_target_format_abort_result(
                config_path=config_path,
                dataset_path=dataset_path,
                output_path=output_path,
                checkpoint_dir=checkpoint_dir,
                checkpoint_binding=checkpoint_binding,
                records=records,
                total_episode_count=len(episodes),
                max_new_tokens=response_token_ceiling,
                unusable_fraction=unusable_fraction,
                trigger_phase="batch_boundary",
                trigger_batch_id=batch_id,
                runtime=runtime,
                strict_capacity_receipt=strict_capacity_receipt,
                strict_capacity_receipt_path=strict_capacity_receipt_path,
                capacity=capacity,
                model_placement=model_placement,
            )
        _release_generation_memory(torch)

    by_id = {record.episode_id: record for record in records}
    ordered = [by_id[episode.episode_id] for episode in episodes]
    artifact = write_agent_trajectory_records(output_path, ordered)
    stage_ended = time.perf_counter()
    if stage_ended <= stage_started:
        stage_ended = stage_started + 1e-9
    telemetry = write_telemetry_ledger(
        output_path.with_suffix(".telemetry.json"),
        [
            record_stage_interval(
                stage_id="target_trajectory_generation",
                interval_id=f"target-trajectory-generation-{dataset_sha256[:16].lower()}",
                run_id=f"target-{sha256_file(config_path)[:16]}",
                attempt_id="initial",
                start_monotonic_seconds=stage_started,
                end_monotonic_seconds=stage_ended,
                status="completed",
                accelerator_stage=True,
                observed_device_receipt_sha256=strict_capacity_receipt_hash,
                input_artifact_hashes={
                    "target_config": sha256_file(config_path),
                    "episode_dataset": sha256_file(dataset_path),
                },
                output_artifact_hashes={
                    "trajectory_records": sha256_file(output_path),
                    "trajectory_manifest": sha256_file(output_path.with_suffix(".manifest.json")),
                },
            )
        ],
        strict_capacity_receipt=strict_capacity_receipt,
        local_only=False,
        project_root=project_root,
    )
    format_failure_summary = _summarize_target_format_failures(
        ordered,
        max_new_tokens=response_token_ceiling,
        total_episode_count=len(episodes),
        unusable_fraction=unusable_fraction,
    )
    format_fallback_episode_ids = format_failure_summary["format_fallback_episode_ids"]
    parse_failure_episode_ids = format_failure_summary["parse_failure_episode_ids"]
    unresolved_parse_failure_episode_ids = format_failure_summary[
        "unresolved_parse_failure_episode_ids"
    ]
    if not isinstance(format_fallback_episode_ids, list) or not isinstance(
        parse_failure_episode_ids, list
    ) or not isinstance(unresolved_parse_failure_episode_ids, list):
        raise RuntimeError("target format-failure summary has invalid episode lists")
    if set(fallback_ledger) != set(format_fallback_episode_ids):
        raise ValueError("final-holdout format fallback ledger record set mismatch")
    fallback_ledger_path = _final_holdout_fallback_ledger_path(checkpoint_dir)
    result = {
        "schema_version": "2.5.0",
        "status": format_failure_summary["status"],
        "errors": format_failure_summary["errors"],
        "config_sha256": sha256_file(config_path),
        "dataset_sha256": dataset_sha256,
        "model_id": model_id,
        "model_revision": model_revision,
        "tokenizer_revision": tokenizer_revision,
        "checkpoint_binding": checkpoint_binding,
        "runtime": runtime,
        "strict_runtime_capacity_receipt": strict_capacity_receipt,
        "strict_runtime_capacity_receipt_source": (
            str(strict_capacity_receipt_path) if strict_capacity_receipt_path is not None else None
        ),
        "runtime_telemetry": telemetry,
        "model_device_placement": model_placement,
        "capacity_plan": capacity,
        "capacity_plan_checkpoint": {
            "path": str(capacity_checkpoint_path),
            "sha256": sha256_file(capacity_checkpoint_path),
        },
        "trajectory_artifact": artifact,
        "truncated_response_count": format_failure_summary["truncated_response_count"],
        "truncated_response_episode_ids": format_failure_summary["truncated_response_episode_ids"],
        "truncated_response_policy": TARGET_TRUNCATED_RESPONSE_POLICY,
        "response_token_ceiling": format_failure_summary["response_token_ceiling"],
        "response_token_ceiling_reached_count": format_failure_summary[
            "response_token_ceiling_reached_count"
        ],
        "format_fallback_count": format_failure_summary["format_fallback_count"],
        "format_fallback_episode_ids": format_fallback_episode_ids,
        "format_repair_count": format_failure_summary["format_repair_count"],
        "format_repair_episode_ids": format_failure_summary["format_repair_episode_ids"],
        "parse_failure_count": format_failure_summary["parse_failure_count"],
        "parse_failure_episode_ids": parse_failure_episode_ids,
        "unresolved_parse_failure_count": format_failure_summary[
            "unresolved_parse_failure_count"
        ],
        "unresolved_parse_failure_episode_ids": unresolved_parse_failure_episode_ids,
        "unusable_observation_count": format_failure_summary["unusable_observation_count"],
        "unusable_observation_episode_ids": format_failure_summary[
            "unusable_observation_episode_ids"
        ],
        "unusable_observation_budget": format_failure_summary["unusable_observation_budget"],
        "unusable_observation_policy": TARGET_UNUSABLE_RESPONSE_POLICY,
        "format_failure_summary": format_failure_summary,
        "synthetic_fixture_count": sum(record.synthetic_fixture for record in ordered),
        "gold_blind_request_contract": True,
        "claim_dispositions": {
            claim: "DEFERRED_FROZEN_EVALUATION" for claim in ("RQ3", "H3", "H4")
        },
        "final_holdout_resume_contract": {
            "policy": "reuse_strict_or_bounded_repair_fail_closed_on_fallbacks",
            "fallback_ledger_path": str(fallback_ledger_path),
            "fallback_ledger_sha256": (
                sha256_file(fallback_ledger_path) if fallback_ledger_path.is_file() else None
            ),
            "replacement_without_separately_versioned_exploratory_run": "forbidden",
        },
        "claim_boundary": (
            "PASS proves observed, hash-bound model outputs from an oracle-blind request schema. "
            "A single schema-only repair may be accepted only when its second response passes the "
            "unchanged strict parser; the first-pass incident and diagnostic hash remain retained. "
            "A response that reached the registered token ceiling, and any response whose repair "
            "was exhausted, is retained as an unobserved outcome rather than as model behaviour. "
            "PASS means such responses stayed within the registered budget; it does not assert "
            "that they are ignorable. Their effect must be bounded from both directions during "
            "analysis, and research conclusions still require the frozen evaluator and error "
            "analysis."
        ),
    }
    write_json(output_path.with_suffix(".run.json"), result)
    return result


def _validate_shared_receipt_matches_observed_runtime(
    receipt: dict[str, object], runtime: dict[str, object]
) -> None:
    """Require the launch receipt and fresh per-stage probe to identify one registered
    accelerator device.

    Disk and host-memory availability naturally drift during a run, so they are
    deliberately not compared.  All device identity and accelerator-capability
    fields must agree before a pre-stage receipt can bind the stage telemetry.
    """

    receipt_probe = receipt.get("probe")
    runtime_probe = runtime.get("probe")
    if not isinstance(receipt_probe, dict) or not isinstance(runtime_probe, dict):
        raise ValueError("strict capacity receipt or observed runtime probe is invalid")
    fields = (
        "compute_available",
        "device_type",
        "device_name",
        "device_index",
        "device_memory_gib",
        "bf16_supported",
        "tensor_probe_passed",
        "compute_capability",
        "evidence_kind",
    )
    if any(receipt_probe.get(field) != runtime_probe.get(field) for field in fields):
        raise ValueError("strict capacity receipt does not match the fresh runtime probe")


def _target_checkpoint_binding(config_path: Path) -> dict[str, str]:
    config = load_yaml(config_path)
    return {
        "binding_schema_version": TARGET_CHECKPOINT_BINDING_SCHEMA_VERSION,
        "trajectory_schema_version": AGENT_TRAJECTORY_SCHEMA_VERSION,
        "timing_schema_version": TIMING_SCHEMA_VERSION,
        "config_sha256": sha256_file(config_path),
        "target_runner_sha256": sha256_file(Path(__file__)),
        "agent_trajectory_sha256": sha256_file(Path(__file__).with_name("agent_trajectory.py")),
        "oracle_sha256": sha256_file(Path(__file__).with_name("oracle.py")),
        "oracle_trajectory_schema_version": TRAJECTORY_SCHEMA_VERSION,
        "model_id": str(config["model_id"]),
        "model_revision": str(config["model_revision"]),
        "tokenizer_revision": str(config["tokenizer_revision"]),
        "response_format": str(config["response_format"]),
        "max_format_attempts": str(config["max_format_attempts"]),
        "format_repair_policy": str(config["format_repair_policy"]),
        "event_id_max_length": str(config["event_id_max_length"]),
    }


def _checkpoint_binding_compatible(
    expected_binding: dict[str, str],
    observed_binding: dict[str, str],
) -> bool:
    """Allow resuming implementation-only target-runner hotfixes on the same protocol."""

    if observed_binding == expected_binding:
        return True
    if set(observed_binding) != set(expected_binding):
        return False
    differing_fields = [
        field
        for field in expected_binding
        if observed_binding[field] != expected_binding[field]
    ]
    if differing_fields != ["target_runner_sha256"]:
        return False
    return observed_binding["target_runner_sha256"] in TARGET_RUNNER_RESUME_COMPATIBLE_SHA256


def _load_target_checkpoint(
    path: Path,
    expected_binding: dict[str, str],
    *,
    expected_record_binding: dict[str, str] | None = None,
) -> AgentTrajectoryRecord:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if set(payload) != {"schema_version", "binding", "record", "resume_disposition"}:
        raise ValueError(f"checkpoint fields mismatch: {path.name}")
    if payload.get("schema_version") != TARGET_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(f"checkpoint schema mismatch: {path.name}")
    normalized_expected = _validate_target_checkpoint_binding(expected_binding)
    normalized_observed = _validate_target_checkpoint_binding(payload.get("binding"))
    if not _checkpoint_binding_compatible(normalized_expected, normalized_observed):
        raise ValueError(f"checkpoint source binding mismatch: {path.name}")
    record = AgentTrajectoryRecord.model_validate(payload["record"])
    if payload["resume_disposition"] != _checkpoint_resume_disposition(record):
        raise ValueError(f"checkpoint resume disposition mismatch: {path.name}")
    if (
        record.schema_version != normalized_expected["trajectory_schema_version"]
        or record.timing_schema_version != normalized_expected["timing_schema_version"]
        or record.model_id != normalized_expected["model_id"]
        or record.model_revision != normalized_expected["model_revision"]
        or record.tokenizer_revision != normalized_expected["tokenizer_revision"]
    ):
        raise ValueError(f"checkpoint record binding mismatch: {path.name}")
    if expected_record_binding is not None and any(
        field
        not in {
            "episode_sha256",
            "request_sha256",
            "prompt_sha256",
        }
        for field in expected_record_binding
    ):
        raise ValueError("checkpoint expected record binding fields mismatch")
    if expected_record_binding is not None:
        if set(expected_record_binding) != {
            "episode_sha256",
            "request_sha256",
            "prompt_sha256",
        }:
            raise ValueError("checkpoint expected record binding fields mismatch")
        if any(
            getattr(record, field) != expected
            for field, expected in expected_record_binding.items()
        ):
            raise ValueError(f"checkpoint episode or prompt binding mismatch: {path.name}")
    if record.diagnostic_artifact_sha256 is not None:
        diagnostic = path.parent / "_nonpublic_diagnostics" / f"{record.episode_id}.malformed.json"
        if not diagnostic.is_file() or sha256_file(diagnostic) != record.diagnostic_artifact_sha256:
            raise ValueError(f"checkpoint diagnostic binding mismatch: {path.name}")
    return record


def _write_target_checkpoint(
    path: Path,
    binding: dict[str, str],
    record: AgentTrajectoryRecord,
) -> None:
    normalized_binding = _validate_target_checkpoint_binding(binding)
    if (
        record.schema_version != normalized_binding["trajectory_schema_version"]
        or record.timing_schema_version != normalized_binding["timing_schema_version"]
        or record.model_id != normalized_binding["model_id"]
        or record.model_revision != normalized_binding["model_revision"]
        or record.tokenizer_revision != normalized_binding["tokenizer_revision"]
    ):
        raise ValueError(f"checkpoint record binding mismatch: {path.name}")
    write_json(
        path,
        {
            "schema_version": TARGET_CHECKPOINT_SCHEMA_VERSION,
            "binding": normalized_binding,
            "record": record.model_dump(mode="json"),
            "resume_disposition": _checkpoint_resume_disposition(record),
        },
    )


def _validate_target_checkpoint_binding(value: object) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != TARGET_CHECKPOINT_BINDING_FIELDS:
        raise ValueError("target checkpoint binding fields mismatch")
    if any(not isinstance(key, str) or not isinstance(item, str) for key, item in value.items()):
        raise ValueError("target checkpoint binding values must be strings")
    normalized = {str(key): str(item) for key, item in value.items()}
    for field in {
        "config_sha256",
        "target_runner_sha256",
        "agent_trajectory_sha256",
        "oracle_sha256",
    }:
        digest = normalized[field]
        if len(digest) != 64 or any(character not in "0123456789ABCDEF" for character in digest):
            raise ValueError(f"target checkpoint binding {field} is not SHA-256")
    if normalized["binding_schema_version"] != TARGET_CHECKPOINT_BINDING_SCHEMA_VERSION:
        raise ValueError("target checkpoint binding schema mismatch")
    if normalized["trajectory_schema_version"] != AGENT_TRAJECTORY_SCHEMA_VERSION:
        raise ValueError("target checkpoint trajectory schema mismatch")
    if normalized["timing_schema_version"] != TIMING_SCHEMA_VERSION:
        raise ValueError("target checkpoint timing schema mismatch")
    if normalized["oracle_trajectory_schema_version"] != TRAJECTORY_SCHEMA_VERSION:
        raise ValueError("target checkpoint oracle trajectory schema mismatch")
    if normalized["event_id_max_length"] != str(TRAJECTORY_EVENT_ID_MAX_LENGTH):
        raise ValueError("target checkpoint event ID envelope binding mismatch")
    for field in {
        "model_id",
        "model_revision",
        "tokenizer_revision",
        "response_format",
        "max_format_attempts",
        "format_repair_policy",
        "event_id_max_length",
    }:
        if not normalized[field]:
            raise ValueError(f"target checkpoint binding {field} is empty")
    return normalized


def _checkpoint_resume_disposition(record: AgentTrajectoryRecord) -> str:
    """Bind a checkpoint to reuse, never a replacement final-holdout call."""

    if record.format_fallback:
        return "reuse_recorded_final_holdout_format_fallback"
    return "reuse_recorded_final_holdout_observation"


def _final_holdout_fallback_ledger_path(checkpoint_dir: Path) -> Path:
    return checkpoint_dir / TARGET_FALLBACK_LEDGER_FILENAME


def _sha256_payload(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest().upper()


def _require_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{field} must be an uppercase SHA-256 digest")
    if any(character not in "0123456789ABCDEF" for character in value):
        raise ValueError(f"{field} must be an uppercase SHA-256 digest")
    return value


def _format_fallback_ledger_entry(record: AgentTrajectoryRecord) -> dict[str, str]:
    if not record.format_fallback:
        raise ValueError("only format fallback records may enter the final-holdout ledger")
    return {
        "episode_id": record.episode_id,
        "episode_sha256": record.episode_sha256,
        "request_sha256": record.request_sha256,
        "prompt_sha256": record.prompt_sha256,
        "record_sha256": record.record_sha256,
    }


def _validate_format_fallback_ledger_entry(value: object) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != TARGET_FALLBACK_LEDGER_ENTRY_FIELDS:
        raise ValueError("final-holdout fallback ledger entry fields mismatch")
    if not all(isinstance(item, str) and item for item in value.values()):
        raise ValueError("final-holdout fallback ledger entry values must be nonempty strings")
    normalized = {str(key): str(item) for key, item in value.items()}
    for field in {
        "episode_sha256",
        "request_sha256",
        "prompt_sha256",
        "record_sha256",
    }:
        _require_sha256(normalized[field], f"final-holdout fallback ledger {field}")
    return normalized


def _build_final_holdout_fallback_ledger(
    *,
    checkpoint_binding: dict[str, str],
    dataset_sha256: str,
    fallback_records: dict[str, dict[str, str]],
) -> dict[str, object]:
    normalized_binding = _validate_target_checkpoint_binding(checkpoint_binding)
    normalized_dataset_sha256 = _require_sha256(dataset_sha256, "fallback ledger dataset_sha256")
    normalized_records: dict[str, dict[str, str]] = {}
    for episode_id, entry in sorted(fallback_records.items()):
        if not isinstance(episode_id, str) or not episode_id:
            raise ValueError("final-holdout fallback ledger episode_id is invalid")
        normalized_entry = _validate_format_fallback_ledger_entry(entry)
        if normalized_entry["episode_id"] != episode_id:
            raise ValueError("final-holdout fallback ledger episode_id binding mismatch")
        normalized_records[episode_id] = normalized_entry
    payload: dict[str, object] = {
        "schema_version": TARGET_FALLBACK_LEDGER_SCHEMA_VERSION,
        "checkpoint_binding": normalized_binding,
        "dataset_sha256": normalized_dataset_sha256,
        "fallback_records": normalized_records,
    }
    payload["ledger_sha256"] = _sha256_payload(payload)
    return payload


def _load_final_holdout_fallback_ledger(
    checkpoint_dir: Path,
    expected_binding: dict[str, str],
    expected_dataset_sha256: str,
) -> dict[str, dict[str, str]]:
    path = _final_holdout_fallback_ledger_path(checkpoint_dir)
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "checkpoint_binding",
        "dataset_sha256",
        "fallback_records",
        "ledger_sha256",
    }:
        raise ValueError("final-holdout fallback ledger fields mismatch")
    if payload["schema_version"] != TARGET_FALLBACK_LEDGER_SCHEMA_VERSION:
        raise ValueError("final-holdout fallback ledger schema mismatch")
    if not isinstance(payload["fallback_records"], dict):
        raise ValueError("final-holdout fallback ledger records must be an object")
    rebuilt = _build_final_holdout_fallback_ledger(
        checkpoint_binding=payload["checkpoint_binding"],
        dataset_sha256=payload["dataset_sha256"],
        fallback_records=payload["fallback_records"],
    )
    if payload["ledger_sha256"] != rebuilt["ledger_sha256"]:
        raise ValueError("final-holdout fallback ledger hash mismatch")
    if canonical_json(payload) != canonical_json(rebuilt):
        raise ValueError("final-holdout fallback ledger canonical payload mismatch")
    if rebuilt["checkpoint_binding"] != _validate_target_checkpoint_binding(expected_binding):
        raise ValueError("final-holdout fallback ledger source binding mismatch")
    if rebuilt["dataset_sha256"] != _require_sha256(
        expected_dataset_sha256,
        "fallback ledger expected dataset_sha256",
    ):
        raise ValueError("final-holdout fallback ledger dataset binding mismatch")
    fallback_records = rebuilt["fallback_records"]
    if not isinstance(fallback_records, dict):  # Defensive type narrowing for future changes.
        raise ValueError("final-holdout fallback ledger records must be an object")
    return {
        episode_id: _validate_format_fallback_ledger_entry(entry)
        for episode_id, entry in fallback_records.items()
    }


def _record_final_holdout_fallback(
    checkpoint_dir: Path,
    checkpoint_binding: dict[str, str],
    dataset_sha256: str,
    fallback_records: dict[str, dict[str, str]],
    record: AgentTrajectoryRecord,
) -> dict[str, dict[str, str]]:
    entry = _format_fallback_ledger_entry(record)
    existing = fallback_records.get(record.episode_id)
    if existing is not None and canonical_json(existing) != canonical_json(entry):
        raise ValueError("final-holdout fallback ledger conflicts with the recorded fallback")
    updated = {**fallback_records, record.episode_id: entry}
    ledger = _build_final_holdout_fallback_ledger(
        checkpoint_binding=checkpoint_binding,
        dataset_sha256=dataset_sha256,
        fallback_records=updated,
    )
    write_json(_final_holdout_fallback_ledger_path(checkpoint_dir), ledger)
    return updated


def _validate_final_holdout_fallback_entry(
    record: AgentTrajectoryRecord,
    entry: dict[str, str] | None,
) -> None:
    if entry is None:
        raise ValueError(
            "recorded final-holdout format fallback is missing its immutable ledger entry: "
            f"{record.episode_id}"
        )
    if canonical_json(entry) != canonical_json(_format_fallback_ledger_entry(record)):
        raise ValueError(
            "recorded final-holdout format fallback does not match its immutable ledger entry: "
            f"{record.episode_id}"
        )


def _load_model(config: dict[str, object]) -> tuple[Any, Any, Any, dict[str, object]]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("install the locked experiment dependency group") from exc
    revision = str(config["model_revision"])
    tokenizer = AutoTokenizer.from_pretrained(
        str(config["model_id"]),
        revision=str(config["tokenizer_revision"]),
        trust_remote_code=False,
        padding_side="left",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        str(config["model_id"]),
        revision=revision,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        device_map=str(config["device_placement"]),
    )
    placement = validate_model_device_placement(model, model_label="target_agent")
    model.eval()
    return torch, tokenizer, model, placement


def _render_chat_prompt(
    tokenizer: Any,
    prompt: str,
    config: dict[str, object],
    *,
    tokenize: bool,
) -> Any:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=tokenize,
        add_generation_prompt=True,
        enable_thinking=bool(config["enable_thinking"]),
    )


def _prompt_token_count(tokenizer: Any, prompt: str, config: dict[str, object]) -> int:
    token_ids = _render_chat_prompt(tokenizer, prompt, config, tokenize=True)
    if hasattr(token_ids, "get"):
        mapped = token_ids.get("input_ids")
        if mapped is not None:
            token_ids = mapped
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if isinstance(token_ids, (list, tuple)) and token_ids and isinstance(
        token_ids[0], (list, tuple)
    ):
        token_ids = token_ids[0]
    if not isinstance(token_ids, (list, tuple)):
        raise RuntimeError("tokenizer chat template did not return token ids")
    return min(len(token_ids), int(config["max_input_tokens"]))


def _order_pending_by_input_length(
    tokenizer: Any,
    pending: list[tuple[Any, str]],
    config: dict[str, object],
) -> tuple[list[tuple[Any, str]], dict[str, object]]:
    scored = [
        (
            _prompt_token_count(tokenizer, prompt, config),
            str(episode.episode_id),
            text_sha256(prompt),
            episode,
            prompt,
        )
        for episode, prompt in pending
    ]
    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    lengths = [item[0] for item in scored]
    return (
        [(item[3], item[4]) for item in scored],
        {
            "strategy": "stable_token_length_descending",
            "selection_inputs": ["rendered_input_token_count", "episode_id", "prompt_sha256"],
            "final_artifact_order": "restored_to_frozen_dataset_order",
            "pending_count": len(scored),
            "minimum_input_tokens": min(lengths) if lengths else None,
            "median_input_tokens": float(statistics.median(lengths)) if lengths else None,
            "maximum_input_tokens": max(lengths) if lengths else None,
            "labels_or_outcomes_used": False,
        },
    )


def _evenly_spaced_items(items: list[Any], count: int) -> list[Any]:
    if count <= 0:
        raise ValueError("representative prompt count must be positive")
    if not items:
        raise ValueError("capacity measurement requires at least one prompt")
    if count == 1:
        return [items[len(items) // 2]]
    return [items[round(index * (len(items) - 1) / (count - 1))] for index in range(count)]


def _build_capacity_prompt_pool(
    tokenizer: Any,
    prompts: list[str],
    config: dict[str, object],
) -> tuple[list[str], dict[str, object]]:
    maximum_batch = max(int(value) for value in config["batch_candidates"])
    reference_batch = int(config["capacity_reference_batch_size"])
    scored = sorted(
        (
            _prompt_token_count(tokenizer, prompt, config),
            text_sha256(prompt),
            prompt,
        )
        for prompt in prompts
    )
    reference = _evenly_spaced_items(scored, reference_batch)
    representative = _evenly_spaced_items(scored, maximum_batch)
    pool_items = reference + representative
    pool_items = pool_items[:maximum_batch]
    pool = [item[2] for item in pool_items]
    return (
        pool,
        {
            "strategy": "nested_token_length_quantiles",
            "maximum_batch_size": maximum_batch,
            "reference_batch_size": reference_batch,
            "prompt_sha256s": [item[1] for item in pool_items],
            "input_token_counts": [item[0] for item in pool_items],
            "labels_or_outcomes_used": False,
        },
    )


def _response_hashes(generated: GeneratedBatch, count: int) -> list[str]:
    if len(generated.responses) < count:
        raise RuntimeError("generation returned fewer responses than the equivalence reference")
    return [text_sha256(response) for response in generated.responses[:count]]


def _measure_capacity(
    torch: Any,
    tokenizer: Any,
    model: Any,
    prompts: list[str],
    config: dict[str, object],
) -> dict[str, object]:
    interface = torch.accelerator
    _, total_bytes = interface.memory.get_memory_info()
    total_gib = total_bytes / (1024**3)
    measurements: list[CapacityMeasurement] = []
    repeat_rates: dict[str, list[float]] = {}
    equivalence: dict[str, dict[str, object]] = {}
    warmup_batches = int(config["capacity_warmup_batches"])
    measurement_repeats = int(config["capacity_measurement_repeats"])
    reference_batch = int(config["capacity_reference_batch_size"])
    prompt_pool, prompt_pool_contract = _build_capacity_prompt_pool(tokenizer, prompts, config)
    reference_hashes: list[str] | None = None
    stop_after_oom = False
    for batch_size in [int(value) for value in config["batch_candidates"]]:
        candidate_id = f"batch-{batch_size}"
        completed = True
        throughput = 0.0
        peak_gib = total_gib
        rates: list[float] = []
        repeat_hashes: list[list[str]] = []
        errors: list[str] = []
        if stop_after_oom:
            completed = False
            errors.append("skipped_after_smaller_batch_oom")
        else:
            try:
                interface.memory.empty_cache()
                probe_prompts = prompt_pool[:batch_size]
                for _ in range(warmup_batches):
                    _generate_batch(
                        torch,
                        tokenizer,
                        model,
                        probe_prompts,
                        config,
                        max_new_tokens=int(config["capacity_probe_new_tokens"]),
                    )
                    interface.synchronize()
                interface.memory.reset_peak_memory_stats()
                for _ in range(measurement_repeats):
                    interface.synchronize()
                    started = time.perf_counter()
                    generated = _generate_batch(
                        torch,
                        tokenizer,
                        model,
                        probe_prompts,
                        config,
                        max_new_tokens=int(config["capacity_probe_new_tokens"]),
                    )
                    interface.synchronize()
                    elapsed = max(time.perf_counter() - started, 1e-9)
                    rates.append(batch_size / elapsed)
                    repeat_hashes.append(_response_hashes(generated, reference_batch))
                repeated_output_stable = bool(repeat_hashes) and all(
                    value == repeat_hashes[0] for value in repeat_hashes[1:]
                )
                if batch_size == reference_batch and repeated_output_stable:
                    reference_hashes = repeat_hashes[0]
                matches_reference = (
                    repeated_output_stable
                    and reference_hashes is not None
                    and repeat_hashes[0] == reference_hashes
                )
                if not repeated_output_stable:
                    errors.append("repeat_response_hash_mismatch")
                if not matches_reference:
                    errors.append("reference_response_hash_mismatch")
                completed = not errors
                throughput = float(statistics.median(rates)) if completed else 0.0
                peak_gib = interface.memory.max_memory_reserved() / (1024**3)
            except RuntimeError as exc:
                if not is_capacity_exhaustion(torch, exc):
                    raise
                completed = False
                rates = []
                repeat_hashes = []
                errors.append("capacity_oom")
                stop_after_oom = bool(config["capacity_stop_after_oom"])
                interface.memory.empty_cache()
        repeat_rates[candidate_id] = rates
        equivalence[candidate_id] = {
            "status": "PASS" if completed else "FAIL",
            "errors": errors,
            "repeat_response_sha256s": repeat_hashes,
            "matches_reference_candidate": completed,
        }
        measurements.append(
            CapacityMeasurement(
                candidate_id=candidate_id,
                batch_size=batch_size,
                samples_per_second=throughput,
                peak_reserved_gib=peak_gib,
                total_memory_gib=total_gib,
                completed=completed,
            )
        )
    maximum_utilization = float(config["target_memory_utilization"])
    result = select_capacity_candidate(
        measurements,
        maximum_utilization=maximum_utilization,
    )
    result["measurement_contract"] = {
        "timing": "synchronized_wall_clock",
        "warmup_batches": warmup_batches,
        "measurement_repeats": measurement_repeats,
        "aggregation": "median_samples_per_second",
        "capacity_probe_new_tokens": int(config["capacity_probe_new_tokens"]),
        "capacity_stop_after_oom": bool(config["capacity_stop_after_oom"]),
        "output_equivalence": "exact_raw_response_sha256_on_reference_prompts",
    }
    result["prompt_pool_contract"] = prompt_pool_contract
    result["reference_response_sha256s"] = reference_hashes
    result["candidate_output_equivalence"] = equivalence
    result["repeat_samples_per_second"] = repeat_rates
    result["ranked_candidate_ids"] = [
        item.candidate_id
        for item in rank_capacity_candidates(
            measurements,
            maximum_utilization=maximum_utilization,
        )
    ]
    return result


def _validate_production_capacity(
    torch: Any,
    tokenizer: Any,
    model: Any,
    scheduled_prompts: list[str],
    config: dict[str, object],
    capacity: dict[str, object],
) -> tuple[dict[str, object], GeneratedBatch, float]:
    if not scheduled_prompts:
        raise ValueError("production capacity validation requires at least one prompt")
    ranked_ids = capacity.get("ranked_candidate_ids")
    measurements = capacity.get("measurements")
    if not isinstance(ranked_ids, list) or not isinstance(measurements, list):
        raise ValueError("capacity result is missing ranked candidates or measurements")
    by_id = {
        str(item["candidate_id"]): item
        for item in measurements
        if isinstance(item, dict) and isinstance(item.get("candidate_id"), str)
    }
    reference_batch = min(int(config["capacity_reference_batch_size"]), len(scheduled_prompts))
    interface = torch.accelerator
    maximum_utilization = float(config["target_memory_utilization"])
    total_gib = float(next(iter(by_id.values()))["total_memory_gib"])
    validation_tokens = int(config["capacity_validation_new_tokens"])
    validation_attempts: list[dict[str, object]] = []

    interface.memory.empty_cache()
    interface.memory.reset_peak_memory_stats()
    interface.synchronize()
    reference_started = time.perf_counter()
    try:
        reference_generation = _generate_batch(
            torch,
            tokenizer,
            model,
            scheduled_prompts[:reference_batch],
            config,
            max_new_tokens=validation_tokens,
        )
        interface.synchronize()
    except RuntimeError as exc:
        if not is_capacity_exhaustion(torch, exc):
            raise
        capacity["status"] = "FAIL"
        capacity["errors"] = [*list(capacity.get("errors", [])), "reference_batch_production_oom"]
        capacity["production_validation"] = {
            "status": "FAIL",
            "errors": ["reference_batch_production_oom"],
            "attempts": [],
        }
        return capacity, GeneratedBatch([], [], []), 0.0
    reference_elapsed = max(time.perf_counter() - reference_started, 1e-9)
    reference_hashes = _response_hashes(reference_generation, reference_batch)
    reference_peak_gib = interface.memory.max_memory_reserved() / (1024**3)

    pre_validation_selected = capacity.get("selected")
    selected_generation: GeneratedBatch | None = None
    selected_elapsed = 0.0
    selected_id: str | None = None
    for candidate_id in [str(value) for value in ranked_ids]:
        measurement = by_id.get(candidate_id)
        if measurement is None:
            raise ValueError(f"ranked capacity candidate has no measurement: {candidate_id}")
        configured_batch = int(measurement["batch_size"])
        actual_batch = min(configured_batch, len(scheduled_prompts))
        errors: list[str] = []
        if configured_batch == int(config["capacity_reference_batch_size"]):
            generated = reference_generation
            elapsed = reference_elapsed
            peak_gib = reference_peak_gib
        else:
            interface.memory.empty_cache()
            interface.memory.reset_peak_memory_stats()
            interface.synchronize()
            started = time.perf_counter()
            try:
                generated = _generate_batch(
                    torch,
                    tokenizer,
                    model,
                    scheduled_prompts[:actual_batch],
                    config,
                    max_new_tokens=validation_tokens,
                )
                interface.synchronize()
                elapsed = max(time.perf_counter() - started, 1e-9)
                peak_gib = interface.memory.max_memory_reserved() / (1024**3)
            except RuntimeError as exc:
                if not is_capacity_exhaustion(torch, exc):
                    raise
                interface.memory.empty_cache()
                validation_attempts.append(
                    {
                        "candidate_id": candidate_id,
                        "configured_batch_size": configured_batch,
                        "actual_batch_size": actual_batch,
                        "status": "FAIL",
                        "errors": ["production_validation_oom"],
                    }
                )
                continue
        candidate_hashes = _response_hashes(generated, reference_batch)
        mismatch_count = sum(
            observed != expected
            for observed, expected in zip(candidate_hashes, reference_hashes, strict=True)
        )
        utilization = peak_gib / total_gib if total_gib > 0 else float("inf")
        if mismatch_count:
            errors.append("production_reference_response_hash_mismatch")
        if utilization > maximum_utilization:
            errors.append("production_peak_memory_above_reserve")
        validation_attempts.append(
            {
                "candidate_id": candidate_id,
                "configured_batch_size": configured_batch,
                "actual_batch_size": actual_batch,
                "status": "PASS" if not errors else "FAIL",
                "errors": errors,
                "samples_per_second": actual_batch / elapsed,
                "peak_reserved_gib": peak_gib,
                "total_memory_gib": total_gib,
                "utilization": utilization,
                "reference_response_mismatch_count": mismatch_count,
                "candidate_reference_response_sha256s": candidate_hashes,
            }
        )
        if errors:
            continue
        selected_generation = generated
        selected_elapsed = elapsed
        selected_id = candidate_id
        break

    if selected_generation is None or selected_id is None:
        capacity["status"] = "FAIL"
        capacity["errors"] = [
            *list(capacity.get("errors", [])),
            "no_production_validated_capacity_candidate",
        ]
        capacity["production_validation"] = {
            "status": "FAIL",
            "errors": ["no_production_validated_capacity_candidate"],
            "reference_batch_size": reference_batch,
            "reference_response_sha256s": reference_hashes,
            "attempts": validation_attempts,
        }
        return capacity, GeneratedBatch([], [], []), 0.0

    capacity["pre_validation_selected"] = pre_validation_selected
    capacity["selected"] = by_id[selected_id]
    capacity["selection_rule"] = (
        "maximum measured throughput within the reserved-memory boundary, followed by exact "
        "reference-response equality and production-token memory validation"
    )
    capacity["production_validation"] = {
        "status": "PASS",
        "errors": [],
        "validation_new_tokens": validation_tokens,
        "validation_repeats": int(config["capacity_validation_repeats"]),
        "reference_batch_size": reference_batch,
        "reference_response_sha256s": reference_hashes,
        "selected_candidate_id": selected_id,
        "attempts": validation_attempts,
        "labels_or_outcomes_used": False,
    }
    return capacity, selected_generation, selected_elapsed


def _merge_generated_batches(left: GeneratedBatch, right: GeneratedBatch) -> GeneratedBatch:
    return GeneratedBatch(
        responses=[*left.responses, *right.responses],
        input_token_counts=[*left.input_token_counts, *right.input_token_counts],
        output_token_counts=[*left.output_token_counts, *right.output_token_counts],
    )


def _release_generation_memory(torch: Any) -> None:
    """Drop cached allocator blocks between production batches."""

    gc.collect()
    interface = getattr(torch, "accelerator", None)
    if interface is None:
        return
    memory = getattr(interface, "memory", None)
    empty_cache = getattr(memory, "empty_cache", None)
    if callable(empty_cache):
        empty_cache()
    synchronize = getattr(interface, "synchronize", None)
    if callable(synchronize):
        synchronize()


def _generate_batch_resilient(
    torch: Any,
    tokenizer: Any,
    model: Any,
    prompts: list[str],
    config: dict[str, object],
    *,
    max_new_tokens: int,
    clock: Callable[[], float] = time.perf_counter,
) -> tuple[GeneratedBatch, float]:
    """Generate a batch and split it when fragmentation causes a recoverable OOM."""

    if not prompts:
        raise ValueError("prompts must not be empty")
    started = clock()
    _release_generation_memory(torch)
    try:
        generated = _generate_batch(
            torch,
            tokenizer,
            model,
            prompts,
            config,
            max_new_tokens=max_new_tokens,
        )
        torch.accelerator.synchronize()
        return generated, max(clock() - started, 0.0)
    except RuntimeError as exc:
        if not is_capacity_exhaustion(torch, exc) or len(prompts) == 1:
            raise
        _release_generation_memory(torch)
        midpoint = len(prompts) // 2
        left, left_elapsed = _generate_batch_resilient(
            torch,
            tokenizer,
            model,
            prompts[:midpoint],
            config,
            max_new_tokens=max_new_tokens,
            clock=clock,
        )
        right, right_elapsed = _generate_batch_resilient(
            torch,
            tokenizer,
            model,
            prompts[midpoint:],
            config,
            max_new_tokens=max_new_tokens,
            clock=clock,
        )
        return _merge_generated_batches(left, right), left_elapsed + right_elapsed


def _generate_batch(
    torch: Any,
    tokenizer: Any,
    model: Any,
    prompts: list[str],
    config: dict[str, object],
    *,
    max_new_tokens: int,
) -> GeneratedBatch:
    rendered = [
        _render_chat_prompt(tokenizer, prompt, config, tokenize=False) for prompt in prompts
    ]
    tokens = tokenizer(
        rendered,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=int(config["max_input_tokens"]),
    )
    input_device = model.get_input_embeddings().weight.device
    tokens = tokens.to(input_device)
    with torch.inference_mode():
        generated = model.generate(
            **tokens,
            max_new_tokens=max_new_tokens,
            do_sample=bool(config["do_sample"]),
            pad_token_id=tokenizer.pad_token_id,
            use_cache=bool(config["use_cache"]),
        )
    suffixes = generated[:, tokens["input_ids"].shape[1] :]
    responses = [
        text.strip() for text in tokenizer.batch_decode(suffixes, skip_special_tokens=True)
    ]
    input_token_counts = [int(value) for value in tokens["attention_mask"].sum(dim=1).tolist()]
    output_token_counts = [
        int(value) for value in suffixes.ne(tokenizer.pad_token_id).sum(dim=1).tolist()
    ]
    return GeneratedBatch(
        responses=responses,
        input_token_counts=input_token_counts,
        output_token_counts=output_token_counts,
    )


def _single_terminal_delimiter_candidates(response: str) -> list[str]:
    """Return unique one-character terminal-delimiter repairs.

    The function never edits content inside the response. It may remove one
    terminal JSON closer or append one terminal closer. The unchanged strict
    parser and the exactly-one-valid-candidate rule remain authoritative.
    """

    candidates: list[str] = []
    if response.endswith(("}", "]")):
        candidates.append(response[:-1])
    candidates.extend(response + closer for closer in ("}", "]"))
    return list(dict.fromkeys(candidates))


def _parse_batch_with_repairs(
    batch: list[tuple[Any, str]],
    generated: GeneratedBatch,
    config: dict[str, object],
    *,
    batch_id: str,
    batch_size: int,
    initial_model_request_wall_seconds: list[float],
    diagnostic_dir: Path,
) -> list[AgentTrajectoryRecord]:
    maximum_attempts = _positive_integer_or_none(config.get("max_format_attempts"))
    if maximum_attempts is None:
        raise ValueError("max_format_attempts must be a positive integer")
    if maximum_attempts != 1:
        raise ValueError("max_format_attempts must equal 1 for deterministic recovery")
    response_token_ceiling = _positive_integer_or_none(config.get("max_new_tokens"))
    if response_token_ceiling is None:
        raise ValueError("max_new_tokens must be a positive integer")
    item_count = len(batch)
    if item_count <= 0:
        raise ValueError("repair batching requires at least one item")
    if not (
        len(generated.responses)
        == len(generated.input_token_counts)
        == len(generated.output_token_counts)
        == len(initial_model_request_wall_seconds)
        == item_count
    ):
        raise ValueError("generated batch and repair metadata lengths do not match")
    model_id = str(config["model_id"])
    model_revision = str(config["model_revision"])
    tokenizer_revision = str(config["tokenizer_revision"])
    records: list[AgentTrajectoryRecord] = []
    for index, (episode, prompt) in enumerate(batch):
        response = generated.responses[index]
        observed_wall_seconds = initial_model_request_wall_seconds[index]
        input_token_count = generated.input_token_counts[index]
        output_token_count = generated.output_token_counts[index]
        request = build_agent_request(episode)
        prompt_sha256 = text_sha256(prompt)
        parse_kwargs = {
            "episode": episode,
            "request": request,
            "model_id": model_id,
            "model_revision": model_revision,
            "tokenizer_revision": tokenizer_revision,
            "prompt_sha256": prompt_sha256,
            "batch_id": batch_id,
            "batch_size": batch_size,
            "observed_model_request_wall_seconds": observed_wall_seconds,
            "input_token_count": input_token_count,
            "output_token_count": output_token_count,
        }
        try:
            records.append(parse_agent_response(response, **parse_kwargs))
            continue
        except ValueError as exc:
            error_class = str(getattr(exc, "error_class", type(exc).__name__))

        # Decoding that stopped at the response ceiling did not observe the end of
        # the model's answer. Appending a terminal delimiter would close a partial
        # trajectory and silently drop whatever the model would have emitted next,
        # so a truncated response is never a recovery candidate.
        truncated = (
            output_token_count is not None and output_token_count >= response_token_ceiling
        )
        if truncated:
            error_class = TARGET_TRUNCATED_RESPONSE_ERROR_CLASS

        malformed_attempts = [(1, response, error_class)]
        recovered_records: list[AgentTrajectoryRecord] = []
        if not truncated:
            for candidate in _single_terminal_delimiter_candidates(response):
                try:
                    recovered_records.append(parse_agent_response(candidate, **parse_kwargs))
                except ValueError:
                    continue
        diagnostic_sha256 = _write_parse_diagnostic(
            diagnostic_dir,
            episode,
            request.request_sha256,
            prompt_sha256,
            model_id,
            model_revision,
            tokenizer_revision,
            malformed_attempts,
        )
        if len(recovered_records) == 1:
            records.append(
                bind_agent_trajectory_record(
                    episode=episode,
                    request=request,
                    trajectory=recovered_records[0].trajectory,
                    model_id=model_id,
                    model_revision=model_revision,
                    tokenizer_revision=tokenizer_revision,
                    synthetic_fixture=False,
                    prompt_sha256=prompt_sha256,
                    generation_attempts=1,
                    format_status="repaired_json",
                    fallback_status="not_used",
                    batch_id=batch_id,
                    batch_size=batch_size,
                    observed_model_request_wall_seconds=observed_wall_seconds,
                    input_token_count=input_token_count,
                    output_token_count=output_token_count,
                    malformed_response_sha256=text_sha256(response),
                    parse_error_class=error_class,
                    diagnostic_artifact_sha256=diagnostic_sha256,
                )
            )
            continue
        records.append(
            bind_agent_trajectory_record(
                episode=episode,
                request=request,
                trajectory=build_safe_fallback_trajectory(episode),
                model_id=model_id,
                model_revision=model_revision,
                tokenizer_revision=tokenizer_revision,
                synthetic_fixture=False,
                prompt_sha256=prompt_sha256,
                generation_attempts=1,
                format_status="safe_fallback",
                fallback_status="used",
                format_fallback=True,
                batch_id=batch_id,
                batch_size=batch_size,
                observed_model_request_wall_seconds=observed_wall_seconds,
                input_token_count=input_token_count,
                output_token_count=output_token_count,
                malformed_response_sha256=text_sha256(response),
                parse_error_class=error_class,
                diagnostic_artifact_sha256=diagnostic_sha256,
            )
        )
    return records


def _parse_with_repairs(
    episode: Any,
    prompt: str,
    response: str,
    config: dict[str, object],
    *,
    batch_id: str,
    batch_size: int,
    initial_model_request_wall_seconds: float,
    initial_input_token_count: int | None,
    initial_output_token_count: int | None,
    diagnostic_dir: Path,
) -> AgentTrajectoryRecord:
    return _parse_batch_with_repairs(
        [(episode, prompt)],
        GeneratedBatch(
            responses=[response],
            input_token_counts=[initial_input_token_count],
            output_token_counts=[initial_output_token_count],
        ),
        config,
        batch_id=batch_id,
        batch_size=batch_size,
        initial_model_request_wall_seconds=[initial_model_request_wall_seconds],
        diagnostic_dir=diagnostic_dir,
    )[0]


def _write_parse_diagnostic(
    diagnostic_dir: Path,
    episode: Any,
    request_sha256: str,
    prompt_sha256: str,
    model_id: str,
    model_revision: str,
    tokenizer_revision: str,
    attempts: list[tuple[int, str, str]],
) -> str:
    return write_malformed_response_diagnostic(
        diagnostic_dir / f"{episode.episode_id}.malformed.json",
        episode_id=episode.episode_id,
        request_sha256=request_sha256,
        prompt_sha256=prompt_sha256,
        model_id=model_id,
        model_revision=model_revision,
        tokenizer_revision=tokenizer_revision,
        attempts=attempts,
    )
