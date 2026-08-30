"""Versioned preparation artifacts required before the final post-run audit."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from vipibench.dataio import sha256_file, write_json
from vipibench.postrun_audit import (
    POSTRUN_AUDIT_FILENAME,
    REQUIRED_ARTIFACTS,
    REQUIRED_PROJECT_BINDINGS,
    validate_postrun_audit_for_finalization,
)
from vipibench.run_protocol import LOCKED_INPUT_MODES, LOCKED_SEEDS
from vipibench.runtime_telemetry import strict_capacity_receipt_sha256

SCHEMA_VERSION = "1.0.0"
SUPPORTING_EVIDENCE_FILENAME = "postrun_supporting_evidence.json"
REQUIRED_PROTOCOL_AMENDMENT = "a100_80gb-response-truncation-guard-v8-2026-08-12"
REQUIRED_DURABLE_LINEAGE = "fp32-a100_80gb-response-truncation-guard-v8-2026-08-12"
BOUNDED_UNUSABLE_OBSERVATION_POLICY = "bounded_unusable_observation_two_sided_bounds_v1"


def _target_run_within_bounded_unusable_budget(run: Mapping[str, object]) -> bool:
    if run.get("status") != "PASS":
        return False
    summary = run.get("format_failure_summary")
    if isinstance(summary, Mapping):
        return (
            summary.get("status") == "PASS"
            and summary.get("unusable_observation_budget_exceeded") is False
            and summary.get("unusable_observation_policy") == BOUNDED_UNUSABLE_OBSERVATION_POLICY
        )
    return (
        run.get("unusable_observation_budget_exceeded") is False
        and run.get("unusable_observation_policy") == BOUNDED_UNUSABLE_OBSERVATION_POLICY
    )


def prepare_postrun_supporting_evidence(
    *,
    project_root: Path,
    output_root: Path,
    launch_authorization_source: Path,
    strict_capacity_receipt_path: Path,
    output_path: Path | None = None,
) -> dict[str, object]:
    """Create only deterministic support artifacts from retained run records.

    The command never manufactures success: source authorization, the shared
    receipt, all encoder decisions, and both target-run summaries must already
    be present. It preserves bounded repair observations without treating them
    as unresolved failures, while exhausted repairs remain fail-closed.
    """

    root = project_root.resolve()
    run_root = output_root.resolve()
    result_path = output_path or run_root / SUPPORTING_EVIDENCE_FILENAME
    errors: list[str] = []

    authorization = _load_object(launch_authorization_source, "launch_authorization_source", errors)
    receipt = _load_object(strict_capacity_receipt_path, "strict_capacity_receipt", errors)
    if receipt:
        try:
            strict_capacity_receipt_sha256(receipt, project_root=root)
        except ValueError as exc:
            errors.append(f"strict_capacity_receipt_invalid:{exc}")
    if (
        strict_capacity_receipt_path.resolve()
        != (run_root / "strict_capacity_receipt.json").resolve()
    ):
        errors.append("strict_capacity_receipt_path_must_be_run_root_artifact")

    launch_artifact = _launch_artifact(authorization, launch_authorization_source, errors)
    matrix_artifact = _matrix_artifact(run_root, errors)
    fallback_artifact, failure_artifact = _target_run_ledgers(run_root, errors)

    artifacts = {
        "launch_authorization": run_root / REQUIRED_ARTIFACTS["launch_authorization"],
        "encoder_matrix": run_root / REQUIRED_ARTIFACTS["encoder_matrix"],
        "fallback_ledger": run_root / REQUIRED_ARTIFACTS["fallback_ledger"],
        "failure_ledger": run_root / REQUIRED_ARTIFACTS["failure_ledger"],
    }
    write_json(artifacts["launch_authorization"], launch_artifact)
    write_json(artifacts["encoder_matrix"], matrix_artifact)
    write_json(artifacts["fallback_ledger"], fallback_artifact)
    write_json(artifacts["failure_ledger"], failure_artifact)

    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS" if not errors else "FAIL",
        "errors": sorted(set(errors)),
        "output_root": str(run_root),
        "artifact_sha256": {
            name: sha256_file(path) for name, path in artifacts.items() if path.is_file()
        },
        "claim_boundary": (
            "Support artifacts bind observed records for the post-run audit; they do not "
            "establish any research result or final E3 eligibility."
        ),
    }
    write_json(result_path, result)
    return result


def write_postrun_run_context(
    *,
    project_root: Path,
    output_root: Path,
    output_path: Path | None = None,
) -> dict[str, object]:
    """Hash-bind the complete post-run input set before raw-manifest creation."""

    root = project_root.resolve()
    run_root = output_root.resolve()
    result_path = output_path or run_root / "run_manifest.pre_audit.json"
    errors: list[str] = []
    artifact_bindings: dict[str, str] = {}
    for name, relative_path in REQUIRED_ARTIFACTS.items():
        path = run_root / relative_path
        if not path.is_file() or path.is_symlink():
            errors.append(f"required_artifact_missing_or_unsafe:{name}")
        else:
            artifact_bindings[name] = sha256_file(path)

    project_bindings: dict[str, str] = {}
    for relative_path in REQUIRED_PROJECT_BINDINGS:
        path = root / relative_path
        if not path.is_file() or path.is_symlink():
            errors.append(f"project_binding_missing_or_unsafe:{relative_path}")
        else:
            project_bindings[relative_path] = sha256_file(path)

    raw_row_counts: dict[str, int] = {}
    for relative_path in _raw_allowlist():
        path = run_root / relative_path
        try:
            raw_row_counts[relative_path] = _count_jsonl_rows(path)
        except (OSError, ValueError) as exc:
            errors.append(f"raw_artifact_invalid:{relative_path}:{exc}")

    from vipibench.manifest import runtime_source_fingerprint

    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS" if not errors else "FAIL",
        "errors": sorted(set(errors)),
        "runtime_source_fingerprint": runtime_source_fingerprint(root),
        "project_bindings": project_bindings,
        "artifact_bindings": artifact_bindings,
        "raw_row_counts": raw_row_counts,
        "claim_boundary": (
            "This pre-audit manifest binds inputs only. It is not a completed run, "
            "a hypothesis result, or an evidence-eligibility determination."
        ),
    }
    write_json(result_path, result)
    return result


def finalize_confirmatory_run(
    *,
    project_root: Path,
    output_root: Path,
    postrun_audit_path: Path,
    output_path: Path | None = None,
) -> dict[str, object]:
    """Write terminal status only from a complete, bound current-run audit."""

    root = project_root.resolve()
    run_root = output_root.resolve()
    result_path = output_path or run_root / "run_manifest.json"
    audit, errors = validate_postrun_audit_for_finalization(
        project_root=root,
        output_root=run_root,
        postrun_audit_path=postrun_audit_path,
    )
    dispositions = audit.get("dispositions") if isinstance(audit.get("dispositions"), dict) else {}
    run_complete = (
        not errors
        and dispositions.get("RUN_COMPLETE") == "PASS"
        and audit.get("status") == "PASS"
    )
    research_eligible = run_complete and audit.get("research_evidence_eligible") is True
    if not run_complete:
        errors.append("postrun_audit_did_not_confirm_run_complete")
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS" if not errors else "FAIL",
        "errors": sorted(set(errors)),
        "postrun_audit_sha256": (
            sha256_file(postrun_audit_path)
            if postrun_audit_path.resolve() == (run_root / POSTRUN_AUDIT_FILENAME).resolve()
            and postrun_audit_path.is_file()
            and not postrun_audit_path.is_symlink()
            else None
        ),
        "RUN_COMPLETE": "PASS" if run_complete else "FAIL",
        "RESEARCH_EVIDENCE_ELIGIBLE": research_eligible,
        "final_claim_audit_created": audit.get("final_claim_audit_created") is True,
        "final_claim_audit_sha256": audit.get("final_claim_audit_sha256"),
        "claim_boundary": (
            "RUN_COMPLETE records only the independent post-run audit outcome. "
            "RESEARCH_EVIDENCE_ELIGIBLE is copied from that audit and may never be "
            "inferred from file presence or a successful notebook cell."
        ),
    }
    write_json(result_path, result)
    return result


def _launch_artifact(
    source: Mapping[str, object], source_path: Path, errors: list[str]
) -> dict[str, object]:
    scopes = source.get("scopes")
    if (
        source.get("status") != "AUTHORIZED"
        or not isinstance(scopes, Mapping)
        or scopes.get("drive_upload") is not True
        or scopes.get("paid_compute") is not True
        or source.get("protocol_amendment") != REQUIRED_PROTOCOL_AMENDMENT
        or source.get("durable_lineage") != REQUIRED_DURABLE_LINEAGE
    ):
        errors.append("launch_authorization_source_invalid")
    source_valid = "launch_authorization_source_invalid" not in errors
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS" if source_valid else "FAIL",
        "errors": [] if source_valid else ["source_invalid"],
        "confirmatory_execution_approved": source_valid,
        "paid_compute_approved": source_valid,
        "protocol_amendment": source.get("protocol_amendment"),
        "durable_lineage": source.get("durable_lineage"),
        "source_path": str(source_path.resolve()),
        "source_sha256": sha256_file(source_path) if source_path.is_file() else None,
    }


def _matrix_artifact(run_root: Path, errors: list[str]) -> dict[str, object]:
    run_ids = [f"mdeberta-{mode}-s{seed}" for mode in LOCKED_INPUT_MODES for seed in LOCKED_SEEDS]
    completed: list[str] = []
    for run_id in run_ids:
        run_dir = run_root / "mdeberta" / run_id
        training = _load_object(run_dir / "training_decision.json", f"training:{run_id}", errors)
        test = _load_object(run_dir / "test_manifest.json", f"test:{run_id}", errors)
        if (
            training.get("status") == "PASS"
            and training.get("run_id") == run_id
            and training.get("test_accessed") is False
            and test.get("status") == "PASS"
            and test.get("run_id") == run_id
        ):
            completed.append(run_id)
        else:
            errors.append(f"encoder_matrix_run_invalid:{run_id}")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS" if len(completed) == len(run_ids) else "FAIL",
        "errors": []
        if len(completed) == len(run_ids)
        else ["encoder_matrix_incomplete_or_invalid"],
        "required_run_count": len(run_ids),
        "completed_runs": completed,
    }


def _target_run_ledgers(
    run_root: Path, errors: list[str]
) -> tuple[dict[str, object], dict[str, object]]:
    run_paths = {
        "core": run_root / "core_target_trajectories.run.json",
        "attack": run_root / "attack_target_trajectories.run.json",
    }
    fallback_records: dict[str, object] = {}
    events: list[dict[str, object]] = []
    repair_observations: list[dict[str, object]] = []
    for label, path in run_paths.items():
        run = _load_object(path, f"target_run:{label}", errors)
        if run.get("status") != "PASS":
            errors.append(f"target_run_not_pass:{label}")
            events.append({"stage": label, "kind": "run_not_pass"})
            continue
        fallback_ids = run.get("format_fallback_episode_ids")
        parse_ids = run.get("parse_failure_episode_ids")
        repair_ids = run.get("format_repair_episode_ids")
        unresolved_ids = run.get("unresolved_parse_failure_episode_ids")
        if not all(
            isinstance(value, list)
            for value in (fallback_ids, parse_ids, repair_ids, unresolved_ids)
        ):
            errors.append(f"target_run_failure_fields_invalid:{label}")
            events.append({"stage": label, "kind": "failure_fields_invalid"})
            continue
        normalized_fallback_ids = sorted(str(item) for item in fallback_ids)
        normalized_parse_ids = sorted(str(item) for item in parse_ids)
        normalized_repair_ids = sorted(str(item) for item in repair_ids)
        normalized_unresolved_ids = sorted(str(item) for item in unresolved_ids)
        if set(normalized_fallback_ids) != set(normalized_unresolved_ids) or set(
            normalized_parse_ids
        ) != set(normalized_repair_ids) | set(normalized_unresolved_ids):
            errors.append(f"target_run_failure_field_sets_mismatch:{label}")
            events.append({"stage": label, "kind": "failure_field_sets_mismatch"})
            continue
        if fallback_ids:
            fallback_records[label] = normalized_fallback_ids
            if not _target_run_within_bounded_unusable_budget(run):
                errors.append("final_holdout_format_fallback_observed")
        if unresolved_ids and not _target_run_within_bounded_unusable_budget(run):
            events.append(
                {
                    "stage": label,
                    "kind": "unresolved_format_failure",
                    "count": len(normalized_unresolved_ids),
                }
            )
        if repair_ids:
            repair_observations.append(
                {
                    "stage": label,
                    "kind": "bounded_schema_repair",
                    "count": len(normalized_repair_ids),
                }
            )
    fallback = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS" if "final_holdout_format_fallback_observed" not in errors else "FAIL",
        "errors": (
            ["final_holdout_format_fallback_observed"]
            if "final_holdout_format_fallback_observed" in errors
            else []
        ),
        "fallback_records": fallback_records,
    }
    if events:
        errors.append("target_run_failures_observed")
    failure = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS" if not events else "FAIL",
        "errors": [] if not events else ["target_run_failures_observed"],
        "events": events,
        "repair_observations": repair_observations,
    }
    return fallback, failure


def _raw_allowlist() -> tuple[str, ...]:
    from vipibench.postrun_audit import RAW_PREDICTION_PATHS, RAW_TRAJECTORY_PATHS

    return (*RAW_PREDICTION_PATHS, *RAW_TRAJECTORY_PATHS)


def _count_jsonl_rows(path: Path) -> int:
    if not path.is_file() or path.is_symlink():
        raise ValueError("missing_or_unsafe")
    count = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            raise ValueError("blank_line")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError("nonobject_row")
        count += 1
    if count <= 0:
        raise ValueError("empty")
    return count


def _load_object(path: Path, label: str, errors: list[str]) -> dict[str, object]:
    if not path.is_file() or path.is_symlink():
        errors.append(f"{label}_missing_or_unsafe")
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"{label}_invalid_json:{exc}")
        return {}
    if not isinstance(value, dict):
        errors.append(f"{label}_not_object")
        return {}
    return value
