from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from pathlib import Path

from vipibench import runtime_capacity
from vipibench.dataio import canonical_json, sha256_file, write_json
from vipibench.manifest import runtime_source_fingerprint
from vipibench.modeling import load_yaml
from vipibench.runtime_capacity import RuntimeProbe, check_runtime_profile

SCHEMA_VERSION = "1.0.0"
STRICT_RECEIPT_TYPE = "strict_runtime_capacity_80gb"
LEDGER_KIND = "runtime_telemetry_ledger"

_STRICT_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "receipt_type",
        "status",
        "errors",
        "profile",
        "profile_sha256",
        "runtime_source_fingerprint",
        "runtime_capacity_source_sha256",
        "hardware_observed",
        "probe",
    }
)
_RUNTIME_CHECK_FIELDS = frozenset({"status", "profile", "errors", "probe", "hardware_observed"})
_PROBE_FIELDS = frozenset(
    {
        "compute_available",
        "device_type",
        "device_name",
        "device_index",
        "device_memory_gib",
        "bf16_supported",
        "tensor_probe_passed",
        "system_ram_gib",
        "disk_free_gib",
        "compute_capability",
        "evidence_kind",
    }
)
_STAGE_FIELDS = frozenset(
    {
        "schema_version",
        "stage_id",
        "interval_id",
        "run_id",
        "attempt_id",
        "start_monotonic_seconds",
        "end_monotonic_seconds",
        "elapsed_seconds",
        "accelerator_stage",
        "observed_device_receipt_sha256",
        "input_artifact_hashes",
        "output_artifact_hashes",
        "status",
        "resume_lineage",
        "record_sha256",
    }
)
_RESUME_LINEAGE_FIELDS = frozenset({"resumed_from_interval_ids"})
_LEDGER_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "validation_status",
        "local_only",
        "records",
        "record_count",
        "unique_interval_count",
        "deduplicated_replay_count",
        "deduplicated_interval_ids",
        "strict_capacity_receipt_sha256",
        "strict_capacity_profile_sha256",
        "strict_capacity_runtime_source_fingerprint",
        "strict_capacity_runtime_capacity_source_sha256",
        "observed_runtime_seconds",
        "accelerator_elapsed_seconds",
        "compute_hours",
        "billed_cost",
        "hardware_observed",
        "execution_status",
        "ledger_sha256",
    }
)
_ALLOWED_STAGE_STATUSES = frozenset({"completed", "failed"})


def strict_capacity_receipt_bindings(
    project_root: Path | None = None,
) -> dict[str, str]:
    """Return the authoritative hash bindings required by a strict accelerator receipt."""

    root = _resolve_project_root(project_root)
    profile_path = root / "configs" / "profiles" / "accelerator_80gb.yaml"
    source_path = root / "src" / "vipibench" / "runtime_capacity.py"
    imported_source = _runtime_capacity_source_path()
    source_sha256 = sha256_file(source_path)
    if sha256_file(imported_source) != source_sha256:
        raise ValueError("runtime_capacity_validator_source_mismatch")
    return {
        "profile_sha256": sha256_file(profile_path),
        "runtime_source_fingerprint": runtime_source_fingerprint(root),
        "runtime_capacity_source_sha256": source_sha256,
    }


def strict_capacity_receipt_sha256(
    receipt: Mapping[str, object],
    *,
    project_root: Path | None = None,
) -> str:
    """Hash only a versioned receipt that passes the authoritative accelerator validator."""

    validated, _ = _validate_strict_capacity_receipt(receipt, project_root=project_root)
    return _sha256_payload(validated)


def build_strict_capacity_receipt(
    runtime_check: Mapping[str, object],
    *,
    project_root: Path | None = None,
) -> dict[str, object]:
    """Convert one canonical strict-capacity check into a bound live receipt.

    Callers may not hand-assemble the telemetry receipt.  This function
    accepts only the exact result shape emitted by
    ``runtime_capacity.check_runtime_profile``, re-runs that authoritative
    validator against the strict accelerator profile, then attaches the source/profile/runtime
    bindings required by the ledger contract.
    """

    candidate = _mapping_copy(runtime_check, "runtime capacity check")
    _require_exact_keys(candidate, _RUNTIME_CHECK_FIELDS, "runtime capacity check")
    profile = _load_authoritative_accelerator_profile(project_root)
    probe = _runtime_probe(candidate["probe"])
    checked = check_runtime_profile(profile, probe)
    if canonical_json(candidate) != canonical_json(checked):
        raise ValueError("runtime capacity check does not match authoritative validator")
    if checked["status"] != "PASS" or checked["hardware_observed"] is not True:
        raise ValueError("strict accelerator runtime-capacity check is not observed PASS")

    receipt = {
        "schema_version": SCHEMA_VERSION,
        "receipt_type": STRICT_RECEIPT_TYPE,
        "status": checked["status"],
        "errors": checked["errors"],
        "profile": checked["profile"],
        **strict_capacity_receipt_bindings(project_root),
        "hardware_observed": checked["hardware_observed"],
        "probe": checked["probe"],
    }
    validated, _ = _validate_strict_capacity_receipt(
        receipt,
        project_root=project_root,
    )
    return validated


def record_stage_interval(
    *,
    stage_id: str,
    interval_id: str,
    run_id: str,
    attempt_id: str,
    start_monotonic_seconds: float,
    end_monotonic_seconds: float,
    status: str,
    accelerator_stage: bool,
    observed_device_receipt_sha256: str | None,
    input_artifact_hashes: Mapping[str, str],
    output_artifact_hashes: Mapping[str, str],
    resumed_from_interval_ids: Iterable[str] = (),
) -> dict[str, object]:
    """Create one hash-bound stage interval without granting a hardware claim."""

    start = _finite_number(start_monotonic_seconds, "start_monotonic_seconds")
    end = _finite_number(end_monotonic_seconds, "end_monotonic_seconds")
    if end < start:
        raise ValueError("negative duration")
    if end == start:
        raise ValueError("non-positive duration")
    record: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "stage_id": stage_id,
        "interval_id": interval_id,
        "run_id": run_id,
        "attempt_id": attempt_id,
        "start_monotonic_seconds": start,
        "end_monotonic_seconds": end,
        "elapsed_seconds": end - start,
        "accelerator_stage": accelerator_stage,
        "observed_device_receipt_sha256": observed_device_receipt_sha256,
        "input_artifact_hashes": dict(input_artifact_hashes),
        "output_artifact_hashes": dict(output_artifact_hashes),
        "status": status,
        "resume_lineage": {
            "resumed_from_interval_ids": list(resumed_from_interval_ids),
        },
    }
    record["record_sha256"] = _sha256_payload(record)
    return _validate_stage_record(record)


def build_telemetry_ledger(
    records: Iterable[Mapping[str, object]],
    *,
    strict_capacity_receipt: Mapping[str, object] | None = None,
    local_only: bool,
    project_root: Path | None = None,
) -> dict[str, object]:
    """Build a hash-bound ledger whose live claims require observed strict accelerator evidence."""

    if not isinstance(local_only, bool):
        raise ValueError("local_only must be boolean")
    validated_records = [_validate_stage_record(record) for record in records]

    receipt_hash: str | None = None
    receipt_bindings: dict[str, str] | None = None
    if local_only:
        if strict_capacity_receipt is not None:
            raise ValueError("local-only ledger cannot include a strict capacity receipt")
        if any(bool(record["accelerator_stage"]) for record in validated_records):
            raise ValueError("local-only ledger cannot include accelerator-stage intervals")
    else:
        if strict_capacity_receipt is None:
            raise ValueError("live ledger requires a strict capacity receipt")
        validated_receipt, receipt_bindings = _validate_strict_capacity_receipt(
            strict_capacity_receipt,
            project_root=project_root,
        )
        receipt_hash = _sha256_payload(validated_receipt)
        accelerated_records = [
            record for record in validated_records if bool(record["accelerator_stage"])
        ]
        if not accelerated_records:
            raise ValueError("live ledger requires at least one accelerator interval")
        for record in validated_records:
            expected_hash = receipt_hash if record["accelerator_stage"] else None
            if record["observed_device_receipt_sha256"] != expected_hash:
                raise ValueError("stage receipt binding mismatch")

    unique_records, deduplicated_interval_ids = _deduplicate_records(validated_records)
    _validate_resume_lineage(unique_records)
    _validate_non_overlapping_intervals(unique_records)

    observed_runtime_seconds = sum(float(record["elapsed_seconds"]) for record in unique_records)
    accelerator_elapsed_seconds: float | None
    compute_hours: float | None
    if local_only:
        accelerator_elapsed_seconds = None
        compute_hours = None
    else:
        accelerator_elapsed_seconds = sum(
            float(record["elapsed_seconds"])
            for record in unique_records
            if bool(record["accelerator_stage"])
        )
        compute_hours = accelerator_elapsed_seconds / 3600

    if not unique_records:
        execution_status = "not_started"
    elif all(record["status"] == "completed" for record in unique_records):
        execution_status = "completed"
    else:
        execution_status = "failed"

    ledger: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "kind": LEDGER_KIND,
        "validation_status": "PASS",
        "local_only": local_only,
        "records": validated_records,
        "record_count": len(validated_records),
        "unique_interval_count": len(unique_records),
        "deduplicated_replay_count": len(validated_records) - len(unique_records),
        "deduplicated_interval_ids": deduplicated_interval_ids,
        "strict_capacity_receipt_sha256": receipt_hash,
        "strict_capacity_profile_sha256": (
            receipt_bindings["profile_sha256"] if receipt_bindings is not None else None
        ),
        "strict_capacity_runtime_source_fingerprint": (
            receipt_bindings["runtime_source_fingerprint"] if receipt_bindings is not None else None
        ),
        "strict_capacity_runtime_capacity_source_sha256": (
            receipt_bindings["runtime_capacity_source_sha256"]
            if receipt_bindings is not None
            else None
        ),
        "observed_runtime_seconds": observed_runtime_seconds,
        "accelerator_elapsed_seconds": accelerator_elapsed_seconds,
        "compute_hours": compute_hours,
        "billed_cost": None,
        "hardware_observed": not local_only,
        "execution_status": execution_status,
    }
    ledger["ledger_sha256"] = _sha256_payload(ledger)
    return ledger


def verify_telemetry_ledger(
    ledger: Mapping[str, object],
    *,
    strict_capacity_receipt: Mapping[str, object] | None = None,
    project_root: Path | None = None,
) -> dict[str, object]:
    """Reject tampered, noncanonical, or unbound telemetry ledgers."""

    candidate = _validate_ledger_envelope(ledger)
    if _sha256_payload(_without(candidate, "ledger_sha256")) != candidate["ledger_sha256"]:
        raise ValueError("ledger hash mismatch")
    records = candidate["records"]
    if not isinstance(records, list):
        raise ValueError("ledger records invalid")
    rebuilt = build_telemetry_ledger(
        records,
        strict_capacity_receipt=strict_capacity_receipt,
        local_only=candidate["local_only"],
        project_root=project_root,
    )
    if canonical_json(rebuilt) != canonical_json(candidate):
        raise ValueError("ledger contents mismatch")
    return candidate


def write_telemetry_ledger(
    output_path: Path,
    records: Iterable[Mapping[str, object]],
    *,
    strict_capacity_receipt: Mapping[str, object] | None = None,
    local_only: bool,
    project_root: Path | None = None,
) -> dict[str, object]:
    """Persist a telemetry ledger only after its complete fail-closed validation."""

    ledger = build_telemetry_ledger(
        records,
        strict_capacity_receipt=strict_capacity_receipt,
        local_only=local_only,
        project_root=project_root,
    )
    write_json(output_path, ledger)
    return ledger


def consolidate_live_telemetry_ledgers(
    input_paths: Iterable[Path],
    *,
    strict_capacity_receipt_path: Path,
    output_path: Path,
    project_root: Path | None = None,
) -> dict[str, object]:
    """Combine independently retained live telemetry ledgers under one receipt.

    Each source ledger is revalidated before its signed interval records are
    admitted.  This intentionally does not accept a mix of receipts: a
    combined compute-time claim needs one strict, source-bound accelerator receipt.
    """

    paths = [Path(path) for path in input_paths]
    if not paths:
        raise ValueError("at least one telemetry ledger is required")
    if len({path.resolve() for path in paths}) != len(paths):
        raise ValueError("telemetry ledger paths must be unique")
    if not strict_capacity_receipt_path.is_file():
        raise FileNotFoundError(strict_capacity_receipt_path)
    receipt_raw = json.loads(strict_capacity_receipt_path.read_text(encoding="utf-8"))
    if not isinstance(receipt_raw, Mapping):
        raise ValueError("strict capacity receipt must be an object")
    receipt_hash = strict_capacity_receipt_sha256(receipt_raw, project_root=project_root)

    records: list[dict[str, object]] = []
    for path in paths:
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(path)
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            raise ValueError(f"telemetry ledger must be an object: {path}")
        ledger = verify_telemetry_ledger(
            raw,
            strict_capacity_receipt=receipt_raw,
            project_root=project_root,
        )
        if ledger["local_only"] is True or ledger["hardware_observed"] is not True:
            raise ValueError(f"telemetry ledger is not observed live: {path}")
        if ledger["strict_capacity_receipt_sha256"] != receipt_hash:
            raise ValueError(f"telemetry ledger receipt mismatch: {path}")
        source_records = ledger["records"]
        if not isinstance(source_records, list):  # Defensive narrowing after validation.
            raise ValueError(f"telemetry ledger records invalid: {path}")
        records.extend(source_records)

    return write_telemetry_ledger(
        output_path,
        records,
        strict_capacity_receipt=receipt_raw,
        local_only=False,
        project_root=project_root,
    )


def _validate_strict_capacity_receipt(
    receipt: Mapping[str, object],
    *,
    project_root: Path | None,
) -> tuple[dict[str, object], dict[str, str]]:
    candidate = _mapping_copy(receipt, "strict capacity receipt")
    _require_exact_keys(candidate, _STRICT_RECEIPT_FIELDS, "strict capacity receipt")
    if candidate["schema_version"] != SCHEMA_VERSION:
        raise ValueError("strict capacity receipt schema version mismatch")
    if candidate["receipt_type"] != STRICT_RECEIPT_TYPE:
        raise ValueError("strict capacity receipt type mismatch")
    if candidate["status"] != "PASS":
        raise ValueError("strict capacity receipt status is not PASS")
    if candidate["errors"] != []:
        raise ValueError("strict capacity receipt errors must be empty")
    if candidate["hardware_observed"] is not True:
        raise ValueError("strict capacity receipt hardware_observed must be true")

    profile = _load_authoritative_accelerator_profile(project_root)
    if candidate["profile"] != profile["name"]:
        raise ValueError("strict capacity receipt profile mismatch")
    bindings = strict_capacity_receipt_bindings(project_root)
    for field, expected in bindings.items():
        if candidate[field] != expected:
            raise ValueError(f"strict capacity receipt {field} mismatch")

    probe = _runtime_probe(candidate["probe"])
    if probe.evidence_kind != "observed":
        raise ValueError("capacity_probe_not_observed")
    if not isinstance(probe.device_index, int) or isinstance(probe.device_index, bool):
        raise ValueError("capacity_probe_device_index_invalid")
    if probe.device_index < 0:
        raise ValueError("capacity_probe_device_index_invalid")
    checked = check_runtime_profile(profile, probe)
    if checked["status"] != "PASS":
        raise ValueError(
            "strict accelerator runtime-capacity validation failed:"
            + ",".join(str(error) for error in checked["errors"])
        )
    if checked["hardware_observed"] is not True:
        raise ValueError("capacity_probe_not_observed")
    return candidate, bindings


def _load_authoritative_accelerator_profile(project_root: Path | None) -> dict[str, object]:
    root = _resolve_project_root(project_root)
    profile = load_yaml(root / "configs" / "profiles" / "accelerator_80gb.yaml")
    if not isinstance(profile, dict):
        raise ValueError("authoritative strict accelerator profile invalid")
    if (
        profile.get("name") != "accelerator_80gb"
        or profile.get("compute_required") is not True
        or profile.get("accelerator_strict") is not True
        or profile.get("required_device_type") != "cuda"
        or profile.get("required_device_names")
        != [
            "NVIDIA A100-SXM4-80GB",
            "NVIDIA A100-PCIE-80GB",
            "NVIDIA A100 80GB",
        ]
        or profile.get("required_compute_capability") != "8.0"
        or profile.get("allow_host_fallback") is not False
    ):
        raise ValueError("authoritative strict accelerator profile invalid")
    return profile


def _resolve_project_root(project_root: Path | None) -> Path:
    if project_root is not None:
        candidates = [project_root.resolve()]
    else:
        candidates = [Path.cwd().resolve(), *Path(__file__).resolve().parents]
    for candidate in candidates:
        profile_path = candidate / "configs" / "profiles" / "accelerator_80gb.yaml"
        source_path = candidate / "src" / "vipibench" / "runtime_capacity.py"
        if profile_path.is_file() and source_path.is_file():
            return candidate
    raise ValueError("authoritative strict accelerator project source unavailable")


def _runtime_capacity_source_path() -> Path:
    module_path = getattr(runtime_capacity, "__file__", None)
    if not isinstance(module_path, str):
        raise ValueError("runtime_capacity_validator_source_unavailable")
    path = Path(module_path)
    if not path.is_file():
        raise ValueError("runtime_capacity_validator_source_unavailable")
    return path


def _runtime_probe(value: object) -> RuntimeProbe:
    candidate = _mapping_copy(value, "capacity probe")
    _require_exact_keys(candidate, _PROBE_FIELDS, "capacity probe")
    for field in (
        "compute_available",
        "bf16_supported",
        "tensor_probe_passed",
    ):
        if not isinstance(candidate[field], bool):
            raise ValueError(f"capacity probe {field} invalid")
    for field in ("device_type", "device_name", "compute_capability", "evidence_kind"):
        _require_nonempty_string(candidate[field], f"capacity probe {field}")
    device_index = candidate["device_index"]
    if not isinstance(device_index, int) or isinstance(device_index, bool) or device_index < 0:
        raise ValueError("capacity_probe_device_index_invalid")
    for field in ("device_memory_gib", "system_ram_gib", "disk_free_gib"):
        _finite_number(candidate[field], f"capacity probe {field}")
    return RuntimeProbe(**candidate)


def _validate_stage_record(record: Mapping[str, object]) -> dict[str, object]:
    candidate = _mapping_copy(record, "stage record")
    _require_exact_keys(candidate, _STAGE_FIELDS, "stage record")
    if candidate["schema_version"] != SCHEMA_VERSION:
        raise ValueError("stage record schema version mismatch")
    for field in ("stage_id", "interval_id", "run_id", "attempt_id"):
        _require_nonempty_string(candidate[field], field)
    start = _finite_number(candidate["start_monotonic_seconds"], "start_monotonic_seconds")
    end = _finite_number(candidate["end_monotonic_seconds"], "end_monotonic_seconds")
    elapsed = _finite_number(candidate["elapsed_seconds"], "elapsed_seconds")
    if end < start:
        raise ValueError("negative duration")
    if end == start:
        raise ValueError("non-positive duration")
    if not math.isclose(elapsed, end - start, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("elapsed duration mismatch")
    if not isinstance(candidate["accelerator_stage"], bool):
        raise ValueError("accelerator_stage invalid")
    if candidate["status"] not in _ALLOWED_STAGE_STATUSES:
        raise ValueError("stage status invalid")
    _validate_hash_mapping(
        candidate["input_artifact_hashes"],
        "input_artifact_hashes",
        allow_empty=False,
    )
    output_hashes = _validate_hash_mapping(
        candidate["output_artifact_hashes"],
        "output_artifact_hashes",
        allow_empty=candidate["status"] == "failed",
    )
    if candidate["status"] == "completed" and not output_hashes:
        raise ValueError("completed stage requires output hashes")
    _validate_resume_lineage_value(candidate["resume_lineage"])
    receipt_hash = candidate["observed_device_receipt_sha256"]
    if candidate["accelerator_stage"]:
        _require_sha256(receipt_hash, "observed_device_receipt_sha256")
    elif receipt_hash is not None:
        raise ValueError("non-accelerator stage cannot bind a device receipt")
    record_hash = _require_sha256(candidate["record_sha256"], "record_sha256")
    if _sha256_payload(_without(candidate, "record_sha256")) != record_hash:
        raise ValueError("stage record hash mismatch")
    return candidate


def _validate_ledger_envelope(ledger: Mapping[str, object]) -> dict[str, object]:
    candidate = _mapping_copy(ledger, "ledger")
    _require_exact_keys(candidate, _LEDGER_FIELDS, "ledger")
    if candidate["schema_version"] != SCHEMA_VERSION:
        raise ValueError("ledger schema version mismatch")
    if candidate["kind"] != LEDGER_KIND:
        raise ValueError("ledger kind mismatch")
    if not isinstance(candidate["local_only"], bool):
        raise ValueError("ledger local_only invalid")
    if not isinstance(candidate["records"], list):
        raise ValueError("ledger records invalid")
    _require_sha256(candidate["ledger_sha256"], "ledger_sha256")
    return candidate


def _deduplicate_records(
    records: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[str]]:
    unique: list[dict[str, object]] = []
    by_interval_id: dict[str, dict[str, object]] = {}
    replayed: list[str] = []
    for record in records:
        interval_id = str(record["interval_id"])
        existing = by_interval_id.get(interval_id)
        if existing is None:
            by_interval_id[interval_id] = record
            unique.append(record)
        elif canonical_json(existing) != canonical_json(record):
            raise ValueError("duplicate interval id has conflicting content")
        elif interval_id not in replayed:
            replayed.append(interval_id)
    return unique, replayed


def _validate_resume_lineage(records: list[dict[str, object]]) -> None:
    by_interval_id = {str(record["interval_id"]): record for record in records}
    for record in records:
        lineage = _validate_resume_lineage_value(record["resume_lineage"])
        for reference_id in lineage:
            referenced = by_interval_id.get(reference_id)
            if referenced is None:
                raise ValueError("resume reference missing")
            if referenced["run_id"] != record["run_id"]:
                raise ValueError("cross-run resume reference")
            if referenced["stage_id"] != record["stage_id"]:
                raise ValueError("cross-stage resume reference")
            if float(referenced["end_monotonic_seconds"]) > float(
                record["start_monotonic_seconds"]
            ):
                raise ValueError("resume reference is not earlier than resumed interval")


def _validate_non_overlapping_intervals(records: list[dict[str, object]]) -> None:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for record in records:
        key = (str(record["run_id"]), str(record["stage_id"]))
        grouped.setdefault(key, []).append(record)
    for group in grouped.values():
        previous_end: float | None = None
        for record in sorted(
            group,
            key=lambda item: (
                float(item["start_monotonic_seconds"]),
                float(item["end_monotonic_seconds"]),
                str(item["interval_id"]),
            ),
        ):
            start = float(record["start_monotonic_seconds"])
            if previous_end is not None and start < previous_end:
                raise ValueError("overlapping stage intervals")
            previous_end = float(record["end_monotonic_seconds"])


def _validate_resume_lineage_value(value: object) -> list[str]:
    candidate = _mapping_copy(value, "resume lineage")
    _require_exact_keys(candidate, _RESUME_LINEAGE_FIELDS, "resume lineage")
    references = candidate["resumed_from_interval_ids"]
    if not isinstance(references, list):
        raise ValueError("resume lineage references invalid")
    parsed = [_require_nonempty_string(item, "resume interval id") for item in references]
    if len(parsed) != len(set(parsed)):
        raise ValueError("duplicate resume reference")
    return parsed


def _validate_hash_mapping(
    value: object,
    label: str,
    *,
    allow_empty: bool,
) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} invalid")
    parsed: dict[str, str] = {}
    for key, item in value.items():
        name = _require_nonempty_string(key, f"{label} key")
        parsed[name] = _require_sha256(item, f"{label}:{name}")
    if not allow_empty and not parsed:
        raise ValueError(f"{label} cannot be empty")
    return parsed


def _mapping_copy(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} invalid")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} fields invalid")
    return dict(value)


def _require_exact_keys(
    value: Mapping[str, object],
    expected: frozenset[str],
    label: str,
) -> None:
    unknown = sorted(key for key in value if key not in expected)
    if unknown:
        raise ValueError(f"{label} unknown fields: {','.join(unknown)}")
    missing = sorted(expected.difference(value))
    if missing:
        raise ValueError(f"{label} missing fields: {','.join(missing)}")


def _require_nonempty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} invalid")
    return value


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} invalid")
    if value != value.upper() or any(character not in "0123456789ABCDEF" for character in value):
        raise ValueError(f"{label} invalid")
    return value


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} invalid")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} invalid")
    return number


def _sha256_payload(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest().upper()


def _without(value: Mapping[str, object], key: str) -> dict[str, object]:
    return {name: item for name, item in value.items() if name != key}
