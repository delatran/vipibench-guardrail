from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from vipibench.checkpoint import StageLedger
from vipibench.dataio import canonical_json, sha256_file, write_json
from vipibench.manifest import runtime_source_fingerprint
from vipibench.modeling import load_yaml
from vipibench.runtime_capacity import (
    RuntimeProbe,
    check_runtime_profile,
    observe_runtime,
)
from vipibench.runtime_telemetry import (
    build_strict_capacity_receipt,
    strict_capacity_receipt_sha256,
)
from vipibench.stage_orchestration import (
    load_stage_plan,
    public_stage_ids,
    validate_run_binding,
    verify_stage_group,
)

SCHEMA_VERSION = "1.0.0"
ANALYSIS_CPU_PROFILE_NAME = "analysis_cpu"
CPU_ANALYSIS_STAGES = frozenset({"analysis", "finalize"})
_PROFILE_FIELDS = frozenset(
    {
        "name",
        "mode",
        "compute_required",
        "require_no_accelerator",
        "minimum_system_ram_gib",
        "minimum_disk_free_gib",
        "accelerator_work_allowed",
        "allow_accelerator_substitution",
    }
)
_CPU_CHECK_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "profile",
        "errors",
        "runtime_observed",
        "accelerator_hardware_observed",
        "accelerator_work_allowed",
        "probe",
    }
)
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
_A100_DEVICE_NAMES = frozenset(
    {"NVIDIA A100-SXM4-80GB", "NVIDIA A100-PCIE-80GB", "NVIDIA A100 80GB"}
)


def check_analysis_cpu_runtime(
    profile: Mapping[str, object],
    probe: RuntimeProbe,
) -> dict[str, object]:
    """Validate a no-accelerator runtime for analysis/finalization only."""

    candidate = dict(profile)
    errors: list[str] = []
    if set(candidate) != _PROFILE_FIELDS:
        errors.append("analysis_cpu_profile_fields_invalid")
    expected = {
        "name": ANALYSIS_CPU_PROFILE_NAME,
        "mode": "confirmatory_analysis",
        "compute_required": False,
        "require_no_accelerator": True,
        "accelerator_work_allowed": False,
        "allow_accelerator_substitution": False,
    }
    for field, value in expected.items():
        if candidate.get(field) != value:
            errors.append(f"analysis_cpu_profile_{field}_mismatch")
    for field, minimum in (
        ("minimum_system_ram_gib", 10.0),
        ("minimum_disk_free_gib", 20.0),
    ):
        try:
            observed = float(candidate.get(field, 0))
        except (TypeError, ValueError):
            errors.append(f"analysis_cpu_profile_{field}_invalid")
        else:
            if observed < minimum:
                errors.append(f"analysis_cpu_profile_{field}_below_minimum")

    base = check_runtime_profile(candidate, probe)
    errors.extend(str(value) for value in base["errors"])
    if probe.evidence_kind != "observed":
        errors.append("analysis_cpu_probe_not_observed")
    if probe.compute_available:
        errors.append("analysis_cpu_accelerator_present")
    if any(
        value is not None
        for value in (
            probe.device_type,
            probe.device_name,
            probe.device_index,
            probe.compute_capability,
        )
    ) or probe.device_memory_gib != 0:
        errors.append("analysis_cpu_device_fields_present")
    if probe.bf16_supported or probe.tensor_probe_passed:
        errors.append("analysis_cpu_accelerator_probe_present")
    unique_errors = sorted(set(errors))
    status = "PASS" if not unique_errors else "FAIL"
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "profile": ANALYSIS_CPU_PROFILE_NAME,
        "errors": unique_errors,
        "runtime_observed": status == "PASS",
        "accelerator_hardware_observed": False,
        "accelerator_work_allowed": False,
        "probe": probe.as_dict(),
    }


def check_analysis_cpu_runtime_path(
    profile_path: Path,
    path_for_disk: Path,
) -> dict[str, object]:
    profile = load_yaml(profile_path)
    if not isinstance(profile, dict):
        raise ValueError("analysis CPU profile must be an object")
    return check_analysis_cpu_runtime(profile, observe_runtime(path_for_disk))


def load_bound_accelerator_preflight(
    *,
    project_root: Path,
    output_root: Path,
    selected_cpu_stage: str,
) -> dict[str, object]:
    """Load the prior A100 preflight without trusting a CPU session as hardware evidence."""

    selected_stage = _validate_cpu_stage(selected_cpu_stage)
    root = project_root.resolve()
    run_root = output_root.resolve()
    launch_path = run_root / "launch_record.json"
    launch = _load_object(launch_path, "prior accelerator launch record")
    allowed_launch_stages = (
        {"attack-evaluate"} if selected_stage == "analysis" else {"attack-evaluate", "analysis"}
    )
    if (
        launch.get("schema_version") != SCHEMA_VERSION
        or launch.get("status") != "PASS"
        or launch.get("mode") != "confirmatory"
        or launch.get("selected_public_stage") not in allowed_launch_stages
    ):
        raise ValueError("prior accelerator launch record is not eligible for CPU transition")
    session_id = _safe_session_id(launch.get("session_id"), "prior accelerator session id")
    launch_hashes = launch.get("launch_hashes")
    if not isinstance(launch_hashes, dict):
        raise ValueError("prior accelerator launch hashes are missing")
    current_fingerprint = runtime_source_fingerprint(root)
    if launch_hashes.get("runtime_source_fingerprint") != current_fingerprint:
        raise ValueError("prior accelerator launch source fingerprint mismatch")

    session_root = run_root / "session_evidence" / "runtime_sessions" / session_id
    preflight_path = session_root / "prelaunch_readiness.json"
    accelerator_path = session_root / "resource_measurement.json"
    preflight = _load_object(preflight_path, "prior accelerator preflight")
    accelerator = _load_object(accelerator_path, "prior accelerator measurement")
    if (
        preflight.get("status") != "PASS"
        or preflight.get("milestone") != "READY_FOR_CONFIRMATORY_LAUNCH"
        or preflight.get("launch_hashes") != launch_hashes
    ):
        raise ValueError("prior accelerator preflight is invalid")
    if sha256_file(preflight_path) != _sha_field(launch, "preflight_sha256"):
        raise ValueError("prior accelerator preflight hash mismatch")
    if sha256_file(accelerator_path) != _sha_field(launch, "accelerator_sha256"):
        raise ValueError("prior accelerator measurement hash mismatch")

    receipt_path = run_root / "strict_capacity_receipt.json"
    strict_receipt = _load_object(receipt_path, "strict capacity receipt")
    receipt_hash = strict_capacity_receipt_sha256(strict_receipt, project_root=root)
    rebuilt_receipt = build_strict_capacity_receipt(accelerator, project_root=root)
    if strict_capacity_receipt_sha256(rebuilt_receipt, project_root=root) != receipt_hash:
        raise ValueError("prior accelerator measurement and strict receipt mismatch")
    if launch.get("strict_capacity_receipt_sha256") != receipt_hash:
        raise ValueError("prior accelerator launch strict receipt mismatch")
    probe = strict_receipt.get("probe")
    if not isinstance(probe, dict) or probe.get("device_name") not in _A100_DEVICE_NAMES:
        raise ValueError("prior accelerator receipt is not an eligible A100 80GB receipt")
    return {
        "launch_path": launch_path,
        "launch": launch,
        "preflight_path": preflight_path,
        "preflight": preflight,
        "accelerator_path": accelerator_path,
        "strict_capacity_receipt_path": receipt_path,
        "strict_capacity_receipt": strict_receipt,
        "strict_capacity_receipt_sha256": receipt_hash,
        "prior_accelerator_session_id": session_id,
    }


def write_cpu_analysis_transition_receipt(
    *,
    selected_stage: str,
    session_id: str,
    project_root: Path,
    output_root: Path,
    stage_plan_path: Path,
    run_binding: dict[str, object],
    runtime_check_path: Path,
    environment_compatibility_path: Path,
    output_path: Path,
) -> dict[str, object]:
    """Bind one CPU-only session to verified prior A100 stages and current analysis software."""

    stage = _validate_cpu_stage(selected_stage)
    current_session_id = _safe_session_id(session_id, "current CPU session id")
    root = project_root.resolve()
    run_root = output_root.resolve()
    session_root = run_root / "session_evidence" / "runtime_sessions" / current_session_id
    expected_runtime_check = session_root / "analysis_runtime_measurement.json"
    expected_environment = session_root / "analysis_environment_compatibility.json"
    expected_output = session_root / "cpu_analysis_transition.json"
    for observed, expected, label in (
        (runtime_check_path.resolve(), expected_runtime_check.resolve(), "runtime check"),
        (
            environment_compatibility_path.resolve(),
            expected_environment.resolve(),
            "environment compatibility",
        ),
        (output_path.resolve(), expected_output.resolve(), "transition receipt"),
    ):
        if observed != expected:
            raise ValueError(f"CPU analysis {label} path mismatch")

    runtime_check = _load_object(runtime_check_path, "CPU analysis runtime check")
    profile_path = root / "configs" / "profiles" / "analysis_cpu.yaml"
    profile = load_yaml(profile_path)
    if not isinstance(profile, dict):
        raise ValueError("analysis CPU profile must be an object")
    probe = _runtime_probe(runtime_check.get("probe"))
    expected_check = check_analysis_cpu_runtime(profile, probe)
    if set(runtime_check) != _CPU_CHECK_FIELDS or canonical_json(runtime_check) != canonical_json(
        expected_check
    ):
        raise ValueError("CPU analysis runtime check does not match the authoritative validator")
    if runtime_check.get("status") != "PASS":
        raise ValueError("CPU analysis runtime check is not PASS")

    environment = _load_object(
        environment_compatibility_path,
        "analysis environment compatibility",
    )
    if (
        environment.get("status") != "PASS"
        or environment.get("dependency_profile") != "analysis-cpu"
        or environment.get("runtime_source_fingerprint") != runtime_source_fingerprint(root)
        or environment.get("accelerator_stack_present") is not False
    ):
        raise ValueError("analysis environment compatibility is not an eligible PASS artifact")

    binding = validate_run_binding(run_binding)
    prior = load_bound_accelerator_preflight(
        project_root=root,
        output_root=run_root,
        selected_cpu_stage=stage,
    )
    plan = load_stage_plan(stage_plan_path)
    stage_ledger = StageLedger(run_root / "orchestration_ledger", artifact_root=run_root)
    group_ledger = StageLedger(run_root / "stage_group_ledger", artifact_root=run_root)
    verified_prior_stages: dict[str, str] = {}
    for prior_stage in public_stage_ids(plan):
        if prior_stage == stage:
            break
        verification = verify_stage_group(
            plan=plan,
            stage_id=prior_stage,
            stage_ledger=stage_ledger,
            group_ledger=group_ledger,
            output_root=run_root,
            run_binding=binding,
        )
        if verification["status"] != "PASS":
            raise RuntimeError(
                {
                    "error": "CPU analysis prior stage verification failed",
                    "stage": prior_stage,
                    "details": verification,
                }
            )
        receipt_path = run_root / "stage_groups" / f"{prior_stage}.receipt.json"
        verified_prior_stages[prior_stage] = sha256_file(receipt_path)

    receipt: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "receipt_type": "cpu_analysis_runtime_transition",
        "status": "PASS",
        "selected_public_stage": stage,
        "session_id": current_session_id,
        "execution_role": "cpu_analysis_only",
        "accelerator_work_allowed": False,
        "current_runtime_observed": True,
        "current_accelerator_hardware_observed": False,
        "analysis_cpu_profile_sha256": sha256_file(profile_path),
        "analysis_runtime_measurement_sha256": sha256_file(runtime_check_path),
        "analysis_environment_compatibility_sha256": sha256_file(
            environment_compatibility_path
        ),
        "runtime_source_fingerprint": runtime_source_fingerprint(root),
        "prior_accelerator_session_id": prior["prior_accelerator_session_id"],
        "prior_accelerator_launch_record_sha256": sha256_file(prior["launch_path"]),
        "prior_accelerator_preflight_sha256": sha256_file(prior["preflight_path"]),
        "prior_accelerator_measurement_sha256": sha256_file(prior["accelerator_path"]),
        "strict_capacity_receipt_sha256": prior["strict_capacity_receipt_sha256"],
        "verified_prior_stage_receipts": verified_prior_stages,
        "run_binding": binding,
        "claim_boundary": (
            "This receipt authorizes CPU-only statistical analysis or finalization after the "
            "listed public stages and their original A100 80GB receipt have been hash-verified. "
            "It is not current accelerator evidence and cannot authorize training, prediction, "
            "target execution, attack generation, or attack evaluation."
        ),
    }
    receipt["receipt_sha256"] = _payload_sha256(receipt)
    write_json(output_path, receipt)
    return receipt


def _runtime_probe(value: object) -> RuntimeProbe:
    if not isinstance(value, Mapping) or set(value) != _PROBE_FIELDS:
        raise ValueError("runtime probe fields are invalid")
    if not isinstance(value["compute_available"], bool):
        raise ValueError("runtime probe compute_available is invalid")
    if not isinstance(value["bf16_supported"], bool) or not isinstance(
        value["tensor_probe_passed"], bool
    ):
        raise ValueError("runtime probe accelerator booleans are invalid")
    for field in ("device_memory_gib", "system_ram_gib", "disk_free_gib"):
        if not isinstance(value[field], (int, float)) or isinstance(value[field], bool):
            raise ValueError(f"runtime probe {field} is invalid")
    if not isinstance(value["evidence_kind"], str):
        raise ValueError("runtime probe evidence_kind is invalid")
    try:
        return RuntimeProbe(
            compute_available=value["compute_available"],
            device_type=value["device_type"],
            device_name=value["device_name"],
            device_index=value["device_index"],
            device_memory_gib=value["device_memory_gib"],
            bf16_supported=value["bf16_supported"],
            tensor_probe_passed=value["tensor_probe_passed"],
            system_ram_gib=value["system_ram_gib"],
            disk_free_gib=value["disk_free_gib"],
            compute_capability=value["compute_capability"],
            evidence_kind=value["evidence_kind"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("runtime probe values are invalid") from exc


def _validate_cpu_stage(value: object) -> str:
    stage = str(value).strip().lower()
    if stage not in CPU_ANALYSIS_STAGES:
        raise ValueError("CPU analysis runtime is allowed only for analysis or finalize")
    return stage


def _safe_session_id(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or Path(value).name != value or value in {".", ".."}:
        raise ValueError(f"{label} is invalid")
    return value


def _load_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"{label} is missing or unsafe: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _sha_field(value: Mapping[str, object], field: str) -> str:
    digest = value.get(field)
    if not isinstance(digest, str) or len(digest) != 64 or any(
        character not in "0123456789ABCDEF" for character in digest
    ):
        raise ValueError(f"{field} is not an uppercase SHA-256 digest")
    return digest


def _payload_sha256(value: Mapping[str, object]) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest().upper()
