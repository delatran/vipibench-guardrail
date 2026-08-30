from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from vipibench.checkpoint import StageLedger
from vipibench.dataio import sha256_file, write_json

PLAN_SCHEMA_VERSION = "1.0.0"
PLAN_STATUS = "locked_controller"
PLAN_EXECUTION_MODE = "sequential_single_a100_80gb_runtime"
GROUP_RECEIPT_SCHEMA_VERSION = "1.0.0"
RUN_BINDING_FIELDS = (
    "runtime_source_fingerprint",
    "launch_hashes",
    "launch_authorization_sha256",
    "stage_plan_sha256",
    "protocol_amendment",
    "durable_lineage",
)
RUN_BINDING_IDENTITY_FIELDS = (
    "stage_plan_sha256",
    "protocol_amendment",
    "durable_lineage",
)
_IDENTIFIER_CHARACTERS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789-_")


def load_stage_plan(path: Path) -> dict[str, object]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("confirmatory stage plan must be an object")
    required_root_fields = {
        "schema_version",
        "status",
        "execution_mode",
        "stages",
        "claim_boundary",
    }
    if set(raw) != required_root_fields:
        raise ValueError("confirmatory stage plan fields are invalid")
    if raw["schema_version"] != PLAN_SCHEMA_VERSION:
        raise ValueError("confirmatory stage plan schema version mismatch")
    if raw["status"] != PLAN_STATUS:
        raise ValueError("confirmatory stage plan is not locked")
    if raw["execution_mode"] != PLAN_EXECUTION_MODE:
        raise ValueError("confirmatory stages must use one sequential A100 80 GB runtime")
    if not isinstance(raw["claim_boundary"], str) or not raw["claim_boundary"].strip():
        raise ValueError("confirmatory stage plan claim boundary is missing")

    stages = raw["stages"]
    if not isinstance(stages, list) or not stages:
        raise ValueError("confirmatory stage plan must contain stages")
    expected_stage_fields = {"id", "prerequisite", "member_stage_ids", "description"}
    seen_stage_ids: set[str] = set()
    seen_member_ids: set[str] = set()
    previous_stage_id: str | None = None
    for index, stage in enumerate(stages):
        if not isinstance(stage, dict) or set(stage) != expected_stage_fields:
            raise ValueError(f"confirmatory stage fields are invalid at index {index}")
        stage_id = _validate_identifier(stage["id"], "public stage")
        if stage_id == "all" or stage_id in seen_stage_ids:
            raise ValueError(f"duplicate or reserved public stage id: {stage_id}")
        seen_stage_ids.add(stage_id)
        if stage["prerequisite"] != previous_stage_id:
            raise ValueError(f"public stage prerequisite chain is invalid: {stage_id}")
        members = stage["member_stage_ids"]
        if not isinstance(members, list):
            raise ValueError(f"member_stage_ids must be a list: {stage_id}")
        for member in members:
            member_id = _validate_identifier(member, "member stage")
            if member_id in seen_member_ids:
                raise ValueError(f"member stage appears in more than one group: {member_id}")
            seen_member_ids.add(member_id)
        if not isinstance(stage["description"], str) or not stage["description"].strip():
            raise ValueError(f"public stage description is missing: {stage_id}")
        previous_stage_id = stage_id
    if stages[0]["id"] != "preflight" or stages[-1]["id"] != "finalize":
        raise ValueError("confirmatory stage plan must begin with preflight and end with finalize")
    return raw


def public_stage_ids(plan: dict[str, object]) -> list[str]:
    return [str(stage["id"]) for stage in _stage_entries(plan)]


def validate_stage_selection(plan: dict[str, object], selected_stage: str) -> str:
    selected = selected_stage.strip().lower()
    allowed = {*public_stage_ids(plan), "all"}
    if selected not in allowed:
        raise ValueError(
            f"unknown confirmatory stage {selected_stage!r}; expected one of {sorted(allowed)}"
        )
    return selected


def stage_enabled(selected_stage: str, stage_id: str) -> bool:
    return selected_stage == "all" or selected_stage == stage_id


def stage_spec(plan: dict[str, object], stage_id: str) -> dict[str, object]:
    for stage in _stage_entries(plan):
        if stage["id"] == stage_id:
            return stage
    raise KeyError(stage_id)


def run_bindings_equivalent_for_maintenance_resume(
    stored: object,
    current: dict[str, object],
) -> bool:
    """Allow hash-only drift when the locked lineage identity is unchanged."""

    if not isinstance(stored, dict):
        return False
    current = validate_run_binding(current)
    return all(stored.get(field) == current.get(field) for field in RUN_BINDING_IDENTITY_FIELDS)


def validate_run_binding(binding: dict[str, object]) -> dict[str, object]:
    if set(binding) != set(RUN_BINDING_FIELDS):
        raise ValueError("stage run binding fields are invalid")
    for field in (
        "runtime_source_fingerprint",
        "launch_authorization_sha256",
        "stage_plan_sha256",
    ):
        _validate_sha256(binding[field], field)
    launch_hashes = binding["launch_hashes"]
    if not isinstance(launch_hashes, dict) or not launch_hashes:
        raise ValueError("launch_hashes must be a non-empty object")
    for name, value in launch_hashes.items():
        if not isinstance(name, str) or not name:
            raise ValueError("launch hash name is invalid")
        _validate_sha256(value, f"launch_hashes.{name}")
    for field in ("protocol_amendment", "durable_lineage"):
        if not isinstance(binding[field], str) or not str(binding[field]).strip():
            raise ValueError(f"{field} must be a non-empty string")
    return binding


def checkpoint_metadata(
    command: list[object],
    run_binding: dict[str, object],
) -> dict[str, object]:
    binding = validate_run_binding(run_binding)
    return {
        "command": [str(value) for value in command],
        **binding,
    }


def complete_stage_group(
    *,
    plan: dict[str, object],
    stage_id: str,
    stage_ledger: StageLedger,
    group_ledger: StageLedger,
    output_root: Path,
    run_binding: dict[str, object],
    direct_outputs: list[Path] | None = None,
) -> dict[str, object]:
    binding = validate_run_binding(run_binding)
    spec = stage_spec(plan, stage_id)
    member_markers: dict[str, str] = {}
    for member_id in _member_ids(spec):
        error = _member_stage_error(stage_ledger, member_id, binding)
        if error is not None:
            raise RuntimeError(f"cannot complete public stage {stage_id}: {error}")
        marker_path = stage_ledger.marker_path(member_id)
        member_markers[member_id] = sha256_file(marker_path)

    direct_hashes: dict[str, str] = {}
    for path in direct_outputs or []:
        relative = _relative_output_path(output_root, path)
        if not path.is_file():
            raise FileNotFoundError(f"public stage output is missing: {path}")
        direct_hashes[relative] = sha256_file(path)

    receipt_path = output_root / "stage_groups" / f"{stage_id}.receipt.json"
    receipt = {
        "schema_version": GROUP_RECEIPT_SCHEMA_VERSION,
        "status": "PASS",
        "stage_id": stage_id,
        "prerequisite": spec["prerequisite"],
        "run_binding": binding,
        "member_markers": member_markers,
        "direct_outputs": direct_hashes,
    }
    write_json(receipt_path, receipt)
    group_ledger.complete(
        stage_id,
        [receipt_path],
        _group_metadata(spec, binding),
    )
    return receipt


def verify_stage_group(
    *,
    plan: dict[str, object],
    stage_id: str,
    stage_ledger: StageLedger,
    group_ledger: StageLedger,
    output_root: Path,
    run_binding: dict[str, object],
) -> dict[str, object]:
    binding = validate_run_binding(run_binding)
    spec = stage_spec(plan, stage_id)
    errors: list[str] = []
    expected_group_metadata = _group_metadata(spec, binding)
    group_marker = group_ledger.marker_path(stage_id)
    group_marker_valid = False
    if group_marker.is_file():
        try:
            group_payload = json.loads(group_marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            group_payload = None
        stored_metadata = (
            group_payload.get("metadata") if isinstance(group_payload, dict) else None
        )
        if isinstance(stored_metadata, dict):
            stored_binding = stored_metadata.get("run_binding")
            group_marker_valid = (
                stored_metadata.get("prerequisite") == expected_group_metadata["prerequisite"]
                and stored_metadata.get("member_stage_ids")
                == expected_group_metadata["member_stage_ids"]
                and isinstance(stored_binding, dict)
                and run_bindings_equivalent_for_maintenance_resume(stored_binding, binding)
                and group_payload.get("status") == "PASS"
            )
    if not group_marker_valid:
        errors.append("group_marker_missing_mismatched_or_stale")
    receipt_path = output_root / "stage_groups" / f"{stage_id}.receipt.json"
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        receipt = None
        errors.append("group_receipt_missing_or_invalid")
    if isinstance(receipt, dict):
        expected_fields = {
            "schema_version",
            "status",
            "stage_id",
            "prerequisite",
            "run_binding",
            "member_markers",
            "direct_outputs",
        }
        if set(receipt) != expected_fields:
            errors.append("group_receipt_fields_invalid")
        stored_binding = receipt.get("run_binding")
        if (
            receipt.get("schema_version") != GROUP_RECEIPT_SCHEMA_VERSION
            or receipt.get("status") != "PASS"
            or receipt.get("stage_id") != stage_id
            or receipt.get("prerequisite") != spec["prerequisite"]
            or not isinstance(stored_binding, dict)
            or not run_bindings_equivalent_for_maintenance_resume(stored_binding, binding)
        ):
            errors.append("group_receipt_contract_mismatch")
        marker_hashes = receipt.get("member_markers")
        if not isinstance(marker_hashes, dict):
            errors.append("group_member_markers_invalid")
            marker_hashes = {}
        if set(marker_hashes) != set(_member_ids(spec)):
            errors.append("group_member_marker_set_mismatch")
        for member_id in _member_ids(spec):
            error = _member_stage_error(stage_ledger, member_id, binding)
            if error is not None:
                errors.append(error)
                continue
            if marker_hashes.get(member_id) != sha256_file(stage_ledger.marker_path(member_id)):
                errors.append(f"group_member_marker_hash_mismatch:{member_id}")
        direct_hashes = receipt.get("direct_outputs")
        if not isinstance(direct_hashes, dict):
            errors.append("group_direct_outputs_invalid")
            direct_hashes = {}
        for relative, expected_hash in direct_hashes.items():
            path = _safe_output_member(output_root, relative)
            if path is None or not path.is_file() or sha256_file(path) != expected_hash:
                errors.append(f"group_direct_output_hash_mismatch:{relative}")
    return {
        "stage_id": stage_id,
        "status": "PASS" if not errors else "FAIL",
        "errors": sorted(set(errors)),
    }


def require_stage_prerequisite(
    *,
    plan: dict[str, object],
    stage_id: str,
    stage_ledger: StageLedger,
    group_ledger: StageLedger,
    output_root: Path,
    run_binding: dict[str, object],
) -> None:
    prerequisite = stage_spec(plan, stage_id)["prerequisite"]
    if prerequisite is None:
        return
    results: list[dict[str, object]] = []
    for spec in _stage_entries(plan):
        prior_stage_id = str(spec["id"])
        if prior_stage_id == stage_id:
            break
        result = verify_stage_group(
            plan=plan,
            stage_id=prior_stage_id,
            stage_ledger=stage_ledger,
            group_ledger=group_ledger,
            output_root=output_root,
            run_binding=run_binding,
        )
        results.append(result)
        if result["status"] != "PASS":
            raise RuntimeError(
                {
                    "error": "public_stage_prerequisite_not_verified",
                    "selected_stage": stage_id,
                    "required_stage": prior_stage_id,
                    "details": results,
                }
            )


def build_stage_status(
    *,
    plan: dict[str, object],
    stage_ledger: StageLedger,
    group_ledger: StageLedger,
    output_root: Path,
    run_binding: dict[str, object],
) -> dict[str, object]:
    verified: dict[str, bool] = {}
    stages: list[dict[str, object]] = []
    for spec in _stage_entries(plan):
        stage_id = str(spec["id"])
        result = verify_stage_group(
            plan=plan,
            stage_id=stage_id,
            stage_ledger=stage_ledger,
            group_ledger=group_ledger,
            output_root=output_root,
            run_binding=run_binding,
        )
        prerequisite = spec["prerequisite"]
        prerequisite_complete = prerequisite is None or verified.get(str(prerequisite), False)
        complete = result["status"] == "PASS" and prerequisite_complete
        errors = list(result["errors"])
        if result["status"] == "PASS" and not prerequisite_complete:
            errors.append(f"prerequisite_chain_not_verified:{prerequisite}")
        verified[stage_id] = complete
        if complete:
            state = "COMPLETE"
        elif prerequisite_complete:
            state = "READY"
        else:
            state = "BLOCKED"
        stages.append(
            {
                "id": stage_id,
                "state": state,
                "prerequisite": prerequisite,
                "verification_errors": sorted(set(errors)),
            }
        )
    return {
        "schema_version": "1.0.0",
        "status": "COMPLETE" if all(verified.values()) else "IN_PROGRESS",
        "execution_mode": PLAN_EXECUTION_MODE,
        "parallel_gpu_processes": 1,
        "stages": stages,
    }


def _group_metadata(
    spec: dict[str, object],
    run_binding: dict[str, object],
) -> dict[str, object]:
    return {
        "run_binding": run_binding,
        "prerequisite": spec["prerequisite"],
        "member_stage_ids": _member_ids(spec),
    }


def _member_stage_error(
    stage_ledger: StageLedger,
    member_id: str,
    run_binding: dict[str, object],
) -> str | None:
    if not stage_ledger.verified_complete(member_id):
        return f"member_stage_missing_mismatched_or_stale:{member_id}"
    marker_path = stage_ledger.marker_path(member_id)
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return f"member_stage_marker_invalid:{member_id}"
    metadata = marker.get("metadata")
    if not isinstance(metadata, dict):
        return f"member_stage_metadata_invalid:{member_id}"
    if not run_bindings_equivalent_for_maintenance_resume(metadata, run_binding):
        return f"member_stage_run_binding_mismatch:{member_id}:maintenance"
    return None


def _stage_entries(plan: dict[str, object]) -> list[dict[str, object]]:
    stages = plan.get("stages")
    if not isinstance(stages, list):
        raise ValueError("stage plan has not been validated")
    return [stage for stage in stages if isinstance(stage, dict)]


def _member_ids(spec: dict[str, object]) -> list[str]:
    members = spec.get("member_stage_ids")
    if not isinstance(members, list):
        raise ValueError("stage specification has not been validated")
    return [str(member) for member in members]


def _validate_identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or any(
        character not in _IDENTIFIER_CHARACTERS for character in value
    ):
        raise ValueError(f"{label} id is invalid: {value!r}")
    return value


def _validate_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789ABCDEF" for character in value
    ):
        raise ValueError(f"{label} must be an uppercase SHA-256 digest")
    return value


def _relative_output_path(output_root: Path, path: Path) -> str:
    root = output_root.resolve()
    resolved = path.resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(f"public stage output must stay within output root: {path}") from exc


def _safe_output_member(output_root: Path, value: object) -> Path | None:
    relative = Path(str(value))
    if relative.is_absolute() or ".." in relative.parts:
        return None
    root = output_root.resolve()
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        return None
    return resolved
