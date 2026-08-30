from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import jsonschema

from vipibench.autonomous_runtime import load_autonomous_execution_policy
from vipibench.dataio import sha256_file, write_json
from vipibench.manifest import (
    readiness_manifest_paths,
    runtime_source_fingerprint,
    verify_manifest,
)
from vipibench.modeling import load_yaml
from vipibench.notebook_check import check_notebook
from vipibench.readiness import evaluate_launch_readiness
from vipibench.resource_estimate import validate_resource_estimate

MILESTONE = "READY_FOR_CONFIRMATORY_LAUNCH"
LOCAL_MILESTONE = "LOCAL_PREFLIGHT_CONTRACT_PASS"
NOT_READY = "NOT_READY"

REQUIRED_MANIFEST_PATHS = {
    "configs/experiments/exec_system.yaml",
    "configs/experiments/confirmatory_analysis.yaml",
    "configs/models/mdeberta_core.yaml",
    "configs/models/public_detector.yaml",
    "configs/models/target_agent.yaml",
    "configs/profiles/accelerator_80gb.yaml",
    "configs/resources/autonomous_execution.json",
    "configs/resources/confirmatory_stage_plan.json",
    "configs/resources/resource_estimate.yaml",
    "data/processed/provenance_contrast.jsonl",
    "data/processed/vipibench_exec.jsonl",
    "data/splits/frozen/split_manifest.json",
    "data/splits/confirmatory_final/manifest.json",
    "data/splits/confirmatory_final/test.jsonl",
    "notebooks/confirmatory_run.ipynb",
    "outputs/prelaunch_readiness.schema.json",
    "outputs/resource_estimate_validation.json",
    "outputs/confirmatory_analysis_validation.json",
    "requirements-experiment.lock",
    "src/vipibench/preflight.py",
    "src/vipibench/cache_contract.py",
    "src/vipibench/runtime_capacity.py",
    "src/vipibench/durable_snapshot.py",
    "src/vipibench/stage_orchestration.py",
    "src/vipibench/analysis_protocol.py",
    "src/vipibench/autonomous_runtime.py",
    "src/vipibench/resource_estimate.py",
}

LAUNCH_HASH_PATHS = {
    "artifact_manifest": "artifact_manifest.json",
    "launch_notebook": "notebooks/confirmatory_run.ipynb",
    "accelerator_profile": "configs/profiles/accelerator_80gb.yaml",
    "dependency_lock": "requirements-experiment.lock",
    "encoder_config": "configs/models/mdeberta_core.yaml",
    "experiment_protocol": "configs/experiments/exec_system.yaml",
    "confirmatory_analysis_protocol": "configs/experiments/confirmatory_analysis.yaml",
    "confirmatory_holdout_manifest": "data/splits/confirmatory_final/manifest.json",
    "confirmatory_holdout_test": "data/splits/confirmatory_final/test.jsonl",
    "frozen_split_manifest": "data/splits/frozen/split_manifest.json",
    "provenance_benchmark": "data/processed/provenance_contrast.jsonl",
    "public_detector_config": "configs/models/public_detector.yaml",
    "autonomous_execution_policy": "configs/resources/autonomous_execution.json",
    "confirmatory_stage_plan": "configs/resources/confirmatory_stage_plan.json",
    "resource_estimate": "configs/resources/resource_estimate.yaml",
    "resource_estimate_validation": "outputs/resource_estimate_validation.json",
    "target_agent_config": "configs/models/target_agent.yaml",
    "vipibench_exec": "data/processed/vipibench_exec.jsonl",
}


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def _profile_contract(profile: dict[str, object]) -> tuple[bool, list[str]]:
    errors: list[str] = []
    expected = {
        "name": "accelerator_80gb",
        "mode": "confirmatory",
        "compute_required": True,
        "accelerator_strict": True,
        "required_device_type": "cuda",
        "required_device_names": [
            "NVIDIA A100-SXM4-80GB",
            "NVIDIA A100-PCIE-80GB",
            "NVIDIA A100 80GB",
        ],
        "required_compute_capability": "8.0",
        "bf16_required": True,
        "tensor_probe_required": True,
        "allow_host_fallback": False,
        "human_approval_required": True,
    }
    for key, value in expected.items():
        if profile.get(key) != value:
            errors.append(f"{key}_mismatch")
    for key, minimum in (
        ("minimum_device_memory_gib", 70),
        ("minimum_system_ram_gib", 40),
        ("minimum_disk_free_gib", 80),
    ):
        try:
            observed = float(profile.get(key, 0))
        except (TypeError, ValueError):
            errors.append(f"{key}_invalid")
            continue
        if observed < minimum:
            errors.append(f"{key}_below_{minimum}")
    try:
        maximum_memory = float(profile.get("maximum_device_memory_gib", 0))
    except (TypeError, ValueError):
        errors.append("maximum_device_memory_gib_invalid")
    else:
        if not 80 <= maximum_memory <= 84:
            errors.append("maximum_device_memory_gib_outside_80_class")
    try:
        utilization = float(profile.get("target_memory_utilization", 0))
    except (TypeError, ValueError):
        errors.append("target_memory_utilization_invalid")
    else:
        if not 0.80 <= utilization <= 0.90:
            errors.append("target_memory_utilization_outside_reserve")
    return not errors, errors


def _execution_profile_contract(
    expected_name: str,
    profile: dict[str, object],
) -> tuple[bool, list[str]]:
    if expected_name == "accelerator_80gb":
        return _profile_contract(profile)
    errors: list[str] = []
    if profile.get("name") != expected_name:
        errors.append("name_mismatch")
    if expected_name == "local_smoke":
        if profile.get("mode") != "smoke":
            errors.append("mode_mismatch")
        if profile.get("compute_required") is not False:
            errors.append("compute_required_must_be_false")
    elif expected_name == "standard_compute":
        expected = {
            "mode": "development",
            "compute_required": True,
            "required_device_type": "cuda",
            "advertised_device_memory_class_gib": [16, 24],
            "allow_host_fallback": False,
        }
        for key, value in expected.items():
            if profile.get(key) != value:
                errors.append(f"{key}_mismatch")
        try:
            minimum = float(profile.get("minimum_device_memory_gib", 0))
            maximum = float(profile.get("maximum_expected_device_memory_gib", 0))
        except (TypeError, ValueError):
            errors.append("device_memory_range_invalid")
        else:
            if not 14.5 <= minimum <= 16 or maximum != 24:
                errors.append("device_memory_range_not_16_24_class")
    elif expected_name == "analysis_cpu":
        expected = {
            "mode": "confirmatory_analysis",
            "compute_required": False,
            "require_no_accelerator": True,
            "minimum_system_ram_gib": 10,
            "minimum_disk_free_gib": 20,
            "accelerator_work_allowed": False,
            "allow_accelerator_substitution": False,
        }
        for key, value in expected.items():
            if profile.get(key) != value:
                errors.append(f"{key}_mismatch")
    else:
        errors.append("unknown_profile_contract")
    return not errors, errors


def _manifest_binding(project_root: Path) -> dict[str, object]:
    manifest_path = project_root / "artifact_manifest.json"
    verification = (
        verify_manifest(
            project_root,
            manifest_path,
            expected_paths=readiness_manifest_paths(project_root),
        )
        if manifest_path.is_file()
        else {"status": "FAIL", "errors": ["artifact_manifest_missing"]}
    )
    entries: dict[str, dict[str, object]] = {}
    if manifest_path.is_file():
        manifest = _load_json(manifest_path)
        entries = {
            str(item["path"]): item
            for item in manifest.get("artifacts", [])
            if isinstance(item, dict) and "path" in item
        }
    missing = sorted(REQUIRED_MANIFEST_PATHS - set(entries))
    mismatched = []
    for relative_path in sorted(REQUIRED_MANIFEST_PATHS & set(entries)):
        path = project_root / relative_path
        expected = str(entries[relative_path].get("sha256"))
        if not path.is_file() or sha256_file(path) != expected:
            mismatched.append(relative_path)
    return {
        "status": (
            "PASS"
            if verification.get("status") == "PASS" and not missing and not mismatched
            else "FAIL"
        ),
        "manifest_verification": verification,
        "required_path_count": len(REQUIRED_MANIFEST_PATHS),
        "missing_required_paths": missing,
        "mismatched_required_paths": mismatched,
    }


def _fingerprint_artifact(
    project_root: Path,
    relative_path: str,
    fingerprint: str,
) -> dict[str, object]:
    path = project_root / relative_path
    evidence: dict[str, object] = {"path": relative_path, "exists": path.is_file()}
    if not path.is_file():
        return {"status": "FAIL", "evidence": evidence}
    try:
        payload = _load_json(path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        evidence["error"] = f"{type(exc).__name__}:{exc}"
        return {"status": "FAIL", "evidence": evidence}
    recorded = payload.get("runtime_source_fingerprint")
    evidence.update(
        {
            "artifact_status": payload.get("status"),
            "recorded_runtime_source_fingerprint": recorded,
            "current_runtime_source_fingerprint": fingerprint,
            "test_count": payload.get("test_count"),
            "sha256": sha256_file(path),
        }
    )
    passed = (
        payload.get("status") == "PASS"
        and recorded == fingerprint
        and int(payload.get("test_count", 0)) > 0
    )
    return {"status": "PASS" if passed else "FAIL", "evidence": evidence}


def evaluate_confirmatory_launch_readiness(
    project_root: Path,
    *,
    verify_hash: bool,
    output_path: Path | None = None,
    require_clean_environment: bool = True,
    runtime_environment_compatibility_path: Path | None = None,
    active_isolated_runtime: bool = False,
) -> dict[str, object]:
    root = project_root.resolve()
    checks: list[dict[str, object]] = []

    def record(name: str, passed: bool, evidence: object) -> None:
        checks.append({"name": name, "status": "PASS" if passed else "FAIL", "evidence": evidence})

    environment_overrides = (
        {"environment_compatibility": runtime_environment_compatibility_path}
        if runtime_environment_compatibility_path is not None
        else {}
    )
    using_active_runtime = (
        active_isolated_runtime and runtime_environment_compatibility_path is not None
    )
    if active_isolated_runtime and runtime_environment_compatibility_path is None:
        record(
            "active_isolated_runtime",
            False,
            {"error": "runtime_environment_compatibility_path_required"},
        )
    elif active_isolated_runtime:
        record(
            "active_isolated_runtime",
            True,
            {"environment_compatibility_path": str(runtime_environment_compatibility_path)},
        )
    base = evaluate_launch_readiness(
        root,
        output_path=None,
        require_clean_environment=require_clean_environment and not using_active_runtime,
        artifact_overrides=environment_overrides,
    )
    record("confirmatory_readiness", base.get("status") == "PASS", base)
    record("verify_hash_requested", verify_hash, {"verify_hash": verify_hash})

    profile_paths = {
        "local_smoke": root / "configs/profiles/local_smoke.yaml",
        "standard_compute": root / "configs/profiles/standard_compute.yaml",
        "accelerator_80gb": root / "configs/profiles/accelerator_80gb.yaml",
    }
    profile_evidence: dict[str, object] = {}
    profiles_pass = True
    for expected_name, path in profile_paths.items():
        if not path.is_file():
            profile_evidence[expected_name] = {"exists": False, "path": str(path)}
            profiles_pass = False
            continue
        profile = load_yaml(path)
        contract_pass, contract_errors = _execution_profile_contract(expected_name, profile)
        profile_evidence[expected_name] = {
            "exists": True,
            "path": str(path),
            "sha256": sha256_file(path),
            "observed_name": profile.get("name"),
            "contract_errors": contract_errors,
        }
        profiles_pass = profiles_pass and contract_pass
    record("execution_profiles", profiles_pass, profile_evidence)

    accelerator_profile_path = profile_paths["accelerator_80gb"]
    accelerator_profile = (
        load_yaml(accelerator_profile_path) if accelerator_profile_path.is_file() else {}
    )
    profile_ok, profile_errors = _profile_contract(accelerator_profile)
    record(
        "strict_accelerator_profile",
        profile_ok,
        {"profile": accelerator_profile, "errors": profile_errors},
    )

    resource_estimate = validate_resource_estimate(
        root,
        root / "configs/resources/resource_estimate.yaml",
    )
    record(
        "resource_estimate_contract",
        resource_estimate.get("status") == "PASS",
        resource_estimate,
    )

    autonomous_policy_path = root / "configs/resources/autonomous_execution.json"
    try:
        autonomous_policy = load_autonomous_execution_policy(autonomous_policy_path)
        autonomous_policy_evidence: dict[str, object] = autonomous_policy.as_dict()
        autonomous_policy_passed = True
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        autonomous_policy_evidence = {
            "path": str(autonomous_policy_path),
            "error": f"{type(exc).__name__}:{exc}",
        }
        autonomous_policy_passed = False
    record(
        "autonomous_execution_policy",
        autonomous_policy_passed,
        autonomous_policy_evidence,
    )

    notebook_path = root / "notebooks/confirmatory_run.ipynb"
    notebook = (
        check_notebook(notebook_path)
        if notebook_path.is_file()
        else {"status": "FAIL", "errors": ["notebook_missing"], "path": str(notebook_path)}
    )
    record("launch_notebook_structure", notebook.get("status") == "PASS", notebook)

    manifest_binding = _manifest_binding(root)
    record(
        "launch_artifact_manifest_binding",
        manifest_binding["status"] == "PASS",
        manifest_binding,
    )

    fingerprint = runtime_source_fingerprint(root)
    if require_clean_environment and not using_active_runtime:
        for name, relative_path in (
            ("clean_environment", "outputs/clean_environment_verification.json"),
            ("current_wheel", "outputs/current_wheel_verification.json"),
        ):
            artifact = _fingerprint_artifact(root, relative_path, fingerprint)
            record(name, artifact["status"] == "PASS", artifact["evidence"])
    else:
        record(
            "bootstrap_evidence_boundary",
            True,
            {
                "clean_and_current_wheel_deferred": True,
                "reason": "verification artifact is being regenerated by the active bootstrap",
            },
        )

    launch_hashes: dict[str, str | None] = {
        "runtime_source_fingerprint": fingerprint,
    }
    launch_hashes.update(
        {
            name: sha256_file(root / relative_path) if (root / relative_path).is_file() else None
            for name, relative_path in LAUNCH_HASH_PATHS.items()
        }
    )
    valid_hashes = all(
        isinstance(value, str) and len(value) == 64 for value in launch_hashes.values()
    )
    record("complete_launch_hash_set", verify_hash and valid_hashes, launch_hashes)

    failures = [str(check["name"]) for check in checks if check["status"] != "PASS"]
    status = "PASS" if not failures else "FAIL"
    milestone = (
        MILESTONE
        if status == "PASS" and (require_clean_environment or using_active_runtime)
        else LOCAL_MILESTONE
        if status == "PASS"
        else NOT_READY
    )
    result: dict[str, object] = {
        "schema_version": "2.0.0",
        "generated_at": datetime.now(UTC).isoformat(),
        "milestone": milestone,
        "mode": "prelaunch",
        "status": status,
        "checks": checks,
        "failed_checks": failures,
        "launch_hashes": launch_hashes,
        "hardware_observed": False,
        "paid_compute_authorized": False,
        "hardware_note": (
            "Local PASS is not accelerator evidence. The allocated session must run the strict "
            "accelerator checker and observe an exact registered NVIDIA A100 80GB device name, "
            "CUDA compute capability 8.0, at least 70 GiB VRAM, BF16 tensor execution, at least "
            "40 GiB system RAM, and at least 80 GiB free disk."
        ),
        "launch_gate": (
            "A hash-bound launch authorization with explicit upload and paid-compute scopes plus "
            "the autonomous per-session safety ceiling, a PASS from check-accelerator inside the "
            "allocated session, and "
            "an atomically snapshotted durable output path"
        ),
        "required_live_checks": [
            "strict observed accelerator PASS",
            "exact NVIDIA A100-SXM4-80GB device name observed",
            "atomic durable snapshot confirmed",
            "hash-bound upload and paid-compute authorization confirmed",
            "session launch record written before confirmatory execution",
        ],
        "external_actions_performed": [],
        "claim_boundary": (
            "PASS proves a current, hash-bound local package is ready to enter the fail-closed "
            "launch workflow. It does not prove hardware allocation, paid-run approval, "
            "runtime performance, cost, model quality, or any research hypothesis."
        ),
    }
    if output_path is not None:
        schema = _load_json(root / "outputs/prelaunch_readiness.schema.json")
        jsonschema.validate(result, schema)
        write_json(output_path, result)
    return result
