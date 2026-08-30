from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from vipibench.adaptive_runner import adaptive_execution_plan
from vipibench.analysis_protocol import validate_confirmatory_analysis_protocol
from vipibench.compiler import verify_confirmatory_holdout_package
from vipibench.coverage import audit_proposal_coverage
from vipibench.dataio import sha256_file, write_json
from vipibench.exec_splits import verify_frozen_split_package
from vipibench.experiment_protocol import validate_exec_experiment_protocol
from vipibench.manifest import (
    readiness_manifest_paths,
    runtime_source_fingerprint,
    verify_manifest,
)
from vipibench.modeling import load_yaml
from vipibench.notebook_check import check_notebook
from vipibench.provenance import verify_provenance, verify_training_authorization
from vipibench.provenance_contrast import audit_provenance_contrast_path
from vipibench.run_protocol import validate_encoder_protocol, validate_public_detector_protocol
from vipibench.sample_size import validate_sample_size_protocol

MILESTONE = "READY_FOR_CONFIRMATORY_EXECUTION"
PASS_ARTIFACTS = {
    "benchmark_validation": "outputs/executable_benchmark_validation.json",
    "composition_audit": "outputs/exec_composition_audit.json",
    "split_audit": "data/splits/frozen/split_audit.json",
    "oracle_verification": "outputs/exec_oracle_verification.json",
    "role_leakage_audit": "outputs/role_label_leakage.json",
    "template_generator_leakage_audit": "outputs/template_generator_leakage.json",
    "policy_gate_verification": "outputs/policy_gate_verification.json",
    "four_arm_fixture": "outputs/four_arm_fixture_verification.json",
    "experiment_protocol": "outputs/experiment_protocol_validation.json",
    "confirmatory_analysis": "outputs/confirmatory_analysis_validation.json",
    "provenance": "outputs/provenance_verification.json",
    "training_authorization": "outputs/training_authorization_verification.json",
    "secret_scan": "outputs/secret_scan.json",
    "clean_environment": "outputs/clean_environment_verification.json",
    "environment_compatibility": "outputs/environment_compatibility.json",
    "resource_estimate": "outputs/resource_estimate_validation.json",
    "tfidf_baseline": "outputs/full/tfidf/baseline_manifest.json",
    "provenance_contrast": "outputs/provenance_contrast_manifest.json",
    "artifact_manifest": "artifact_manifest.json",
}


def _artifact_check(
    root: Path,
    name: str,
    relative_path: str,
    *,
    artifact_path: Path | None = None,
) -> dict[str, object]:
    path = artifact_path.resolve() if artifact_path is not None else root / relative_path
    evidence: dict[str, object] = {
        "path": relative_path,
        "resolved_path": str(path),
        "exists": path.is_file(),
    }
    status = "FAIL"
    if path.is_file():
        evidence["sha256"] = sha256_file(path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            evidence["artifact_status"] = payload.get("status")
            status = "PASS" if payload.get("status") == "PASS" else "FAIL"
            if name in {"clean_environment", "environment_compatibility"}:
                current_fingerprint = runtime_source_fingerprint(root)
                evidence["recorded_runtime_source_fingerprint"] = payload.get(
                    "runtime_source_fingerprint"
                )
                evidence["current_runtime_source_fingerprint"] = current_fingerprint
                evidence["runtime_source_fingerprint_matched"] = (
                    payload.get("runtime_source_fingerprint") == current_fingerprint
                )
                if not evidence["runtime_source_fingerprint_matched"]:
                    status = "FAIL"
        except json.JSONDecodeError:
            evidence["artifact_status"] = "INVALID_JSON"
    return {"name": name, "status": status, "evidence": evidence}


def _result_check(name: str, result: dict[str, object]) -> dict[str, object]:
    return {
        "name": name,
        "status": "PASS" if result.get("status") == "PASS" else "FAIL",
        "evidence": result,
    }


def evaluate_launch_readiness(
    project_root: Path,
    *,
    output_path: Path | None = None,
    require_clean_environment: bool = True,
    artifact_overrides: Mapping[str, Path] | None = None,
) -> dict[str, object]:
    root = project_root.resolve()
    artifact_contract = {
        name: path
        for name, path in PASS_ARTIFACTS.items()
        if require_clean_environment or name != "clean_environment"
    }
    overrides = dict(artifact_overrides or {})
    unknown_overrides = sorted(set(overrides) - set(artifact_contract))
    checks = [
        _artifact_check(root, name, path, artifact_path=overrides.get(name))
        for name, path in artifact_contract.items()
    ]
    if unknown_overrides:
        checks.append(
            {
                "name": "artifact_override_scope",
                "status": "FAIL",
                "evidence": {"unknown_override_names": unknown_overrides},
            }
        )
    coverage = audit_proposal_coverage(root, root / "docs/proposal_coverage.yaml")
    coverage_ok = (
        coverage.get("status") == "PASS"
        and coverage.get("coverage_complete") is True
        and coverage.get("covered_external_gates") == ["resource_measurement"]
    )
    checks.append(
        {
            "name": "proposal_coverage_prelaunch_boundary",
            "status": "PASS" if coverage_ok else "FAIL",
            "evidence": coverage,
        }
    )
    contrast_path = root / "data/processed/provenance_contrast.jsonl"
    checks.extend(
        [
            _result_check(
                "encoder_protocol",
                validate_encoder_protocol(root / "configs/models/mdeberta_core.yaml"),
            ),
            _result_check(
                "public_detector_protocol",
                validate_public_detector_protocol(root / "configs/models/public_detector.yaml"),
            ),
            _result_check(
                "sample_size_protocol",
                validate_sample_size_protocol(
                    root / "configs/experiments/sample_size_scaling.yaml"
                ),
            ),
            _result_check(
                "exec_experiment_protocol_live",
                validate_exec_experiment_protocol(
                    root,
                    root / "configs/experiments/exec_system.yaml",
                ),
            ),
            _result_check(
                "frozen_split_package_live",
                verify_frozen_split_package(
                    root / "data/splits/frozen",
                    root / "data/processed/vipibench_exec.jsonl",
                    root / "configs/benchmark/exec_catalog.yaml",
                ),
            ),
            _result_check("provenance_live", verify_provenance(root)),
            _result_check("training_authorization_live", verify_training_authorization(root)),
            _result_check(
                "provenance_contrast_live",
                audit_provenance_contrast_path(contrast_path)
                if contrast_path.is_file()
                else {"status": "FAIL", "errors": ["dataset_missing"]},
            ),
            _result_check(
                "notebook_structure",
                check_notebook(root / "notebooks/experiment_workflow.ipynb"),
            ),
            _result_check("adaptive_execution_plan", adaptive_execution_plan()),
            _result_check(
                "confirmatory_holdout_package",
                verify_confirmatory_holdout_package(
                    root / "configs/benchmark/exec_catalog.yaml",
                    root / "data/splits/frozen",
                    root / "data/splits/confirmatory_final",
                ),
            ),
            _result_check(
                "confirmatory_analysis_protocol",
                validate_confirmatory_analysis_protocol(
                    root,
                    root / "configs/experiments/confirmatory_analysis.yaml",
                ),
            ),
            _result_check(
                "artifact_manifest_live",
                verify_manifest(
                    root,
                    root / "artifact_manifest.json",
                    expected_paths=readiness_manifest_paths(root),
                )
                if (root / "artifact_manifest.json").is_file()
                else {"status": "FAIL", "errors": ["artifact_manifest_missing"]},
            ),
        ]
    )
    profile = load_yaml(root / "configs/profiles/accelerator_80gb.yaml")
    profile_ok = (
        profile.get("name") == "accelerator_80gb"
        and profile.get("compute_required") is True
        and profile.get("accelerator_strict") is True
        and profile.get("required_device_type") == "cuda"
        and profile.get("required_device_names")
        == [
            "NVIDIA A100-SXM4-80GB",
            "NVIDIA A100-PCIE-80GB",
            "NVIDIA A100 80GB",
        ]
        and profile.get("required_compute_capability") == "8.0"
        and float(profile.get("minimum_device_memory_gib", 0)) >= 70
        and 80 <= float(profile.get("maximum_device_memory_gib", 0)) <= 84
        and float(profile.get("minimum_system_ram_gib", 0)) >= 40
        and float(profile.get("minimum_disk_free_gib", 0)) >= 80
        and profile.get("bf16_required") is True
        and profile.get("tensor_probe_required") is True
        and profile.get("allow_host_fallback") is False
        and 0.80 <= float(profile.get("target_memory_utilization", 0)) <= 0.90
    )
    checks.append(
        {
            "name": "accelerator_fail_closed_profile",
            "status": "PASS" if profile_ok else "FAIL",
            "evidence": profile,
        }
    )
    required_runner_files = [
        "src/vipibench/transformer_runner.py",
        "src/vipibench/system_runner.py",
        "src/vipibench/adaptive_runner.py",
        "src/vipibench/agent_trajectory.py",
        "src/vipibench/checkpoint.py",
        "src/vipibench/durable_snapshot.py",
        "src/vipibench/analysis_protocol.py",
        "src/vipibench/autonomous_runtime.py",
    ]
    runner_hashes = {
        path: sha256_file(root / path) if (root / path).is_file() else None
        for path in required_runner_files
    }
    checks.append(
        {
            "name": "full_runner_surface",
            "status": "PASS" if all(runner_hashes.values()) else "FAIL",
            "evidence": runner_hashes,
        }
    )
    failed = [str(check["name"]) for check in checks if check["status"] != "PASS"]
    status = "PASS" if not failed else "FAIL"
    result: dict[str, object] = {
        "schema_version": "1.0.0",
        "milestone": (
            MILESTONE
            if status == "PASS" and require_clean_environment
            else "LOCAL_SMOKE_CONTRACT_PASS"
            if status == "PASS"
            else "NOT_READY"
        ),
        "mode": "prelaunch",
        "status": status,
        "checks": checks,
        "failed_checks": failed,
        "hardware_observed": False,
        "paid_compute_authorized": False,
        "hardware_note": (
            "Hardware is intentionally not claimed locally. Launch must observe an exact "
            "registered NVIDIA A100 80GB device name, at least 70 GiB device memory, BF16 tensor "
            "execution, 40 GiB system memory, and 80 GiB free disk."
        ),
        "launch_gate": (
            "A hash-bound launch authorization with upload and paid-compute scopes, the locked "
            "autonomous execution policy, plus an observed exact registered NVIDIA A100 80GB "
            "runtime profile PASS"
        ),
        "external_actions_performed": [],
        "claim_boundary": (
            "PASS means local artifacts are ready for fail-closed confirmatory execution. "
            "It is not evidence that hardware is allocated or hypotheses hold."
        ),
    }
    if output_path is not None:
        write_json(output_path, result)
    return result
