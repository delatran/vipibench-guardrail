from __future__ import annotations

import os
from pathlib import Path

from vipibench.dataio import sha256_file, write_json
from vipibench.modeling import load_yaml
from vipibench.validation import validate_path


def _tree_size(root: Path) -> int:
    excluded = {".git", ".venv", "build", "outputs", "__pycache__"}
    total = 0
    for current, directories, filenames in os.walk(root):
        directories[:] = [directory for directory in directories if directory not in excluded]
        total += sum((Path(current) / filename).stat().st_size for filename in filenames)
    return total


def resource_measurement_contract() -> dict[str, object]:
    """Describe evidence classes without promoting estimates into observations."""

    return {
        "schema_version": "1.0.0",
        "measurement_classes": {
            "observed_hardware_telemetry": {
                "evidence": "strict_runtime_capacity_receipt",
                "hardware_observed": False,
                "claim_boundary": (
                    "null until an exact versioned observed accelerator receipt, bound to the "
                    "strict accelerator profile and runtime source, passes"
                ),
            },
            "observed_runtime_timing": {
                "evidence": "hash_bound_runtime_telemetry_ledger",
                "compute_hours": None,
                "definition": (
                    "deduplicated measured accelerator-stage monotonic wall time divided by 3600"
                ),
                "claim_boundary": "local-only timing cannot be reported as accelerator compute",
            },
            "observed_public_stage_resources": {
                "evidence": "hash_bound_public_stage_resource_measurement",
                "hardware_observed": False,
                "required_public_stage_count": 9,
                "claim_boundary": (
                    "null until all nine public stages have raw, hash-verified GPU, host RAM, "
                    "CPU, durable-disk, and ephemeral-scratch samples from the A100 runtime"
                ),
            },
            "analytical_model_memory_budget": {
                "evidence": "parameter_count_formula",
                "claim_boundary": "analytical budget, not observed peak device memory",
            },
            "planning_scenarios": {
                "evidence": "resource_estimate_yaml",
                "claim_boundary": "planning envelope, not observed runtime or billed cost",
            },
        },
        "post_run_input_requirements": [
            {
                "name": "strict_runtime_capacity_receipt",
                "schema_version": "1.0.0",
                "receipt_type": "strict_runtime_capacity_80gb",
                "required_for": ["hardware_observed", "compute_hours"],
                "hash_bound": True,
                "required_profile": "accelerator_80gb",
                "profile_hash_bound": True,
                "runtime_source_hash_bound": True,
                "unknown_fields_rejected": True,
                "required_fields": [
                    "schema_version",
                    "receipt_type",
                    "profile",
                    "profile_sha256",
                    "runtime_source_fingerprint",
                    "runtime_capacity_source_sha256",
                    "probe",
                ],
                "authoritative_validator": "vipibench.runtime_capacity.check_runtime_profile",
            },
            {
                "name": "runtime_telemetry_ledger",
                "schema_version": "1.0.0",
                "required_for": ["observed_runtime_timing", "compute_hours"],
                "hash_bound": True,
                "unknown_fields_rejected": True,
                "cross_run_resume_policy": "reject",
                "empty_live_ledger_policy": "reject",
                "required_fields": [
                    "stage_id",
                    "start_monotonic_seconds",
                    "end_monotonic_seconds",
                    "elapsed_seconds",
                    "observed_device_receipt_sha256",
                    "input_artifact_hashes",
                    "output_artifact_hashes",
                    "status",
                    "resume_lineage",
                ],
            },
            {
                "name": "public_stage_resource_measurement",
                "schema_version": "1.0.0",
                "required_for": [
                    "observed_public_stage_resources",
                    "gpu_utilization",
                    "gpu_memory_utilization",
                    "host_cpu_utilization",
                    "host_memory_utilization",
                    "durable_disk_utilization",
                    "ephemeral_scratch_utilization",
                ],
                "path": "resource_measurement.json",
                "hash_bound": True,
                "unknown_fields_rejected": True,
                "required_status": "PASS",
                "required_completed_public_stage_count": 9,
                "raw_sample_reverification_required": True,
                "required_fields": [
                    "measurement_kind",
                    "hardware_observed",
                    "strict_capacity_receipt_sha256",
                    "expected_public_stages",
                    "completed_public_stages",
                    "missing_public_stages",
                    "stage_attempts",
                    "summary_file_count",
                    "public_stage_observed_seconds",
                    "measurement_sha256",
                ],
                "authoritative_validator": (
                    "vipibench.resource_observation.verify_resource_measurement"
                ),
            },
        ],
        "hardware_observed": False,
        "compute_hours": None,
        "billed_cost": None,
    }


def measure_resources(
    project_root: Path,
    dataset_path: Path,
    agreement_path: Path,
    split_dir: Path,
    output_path: Path,
) -> dict[str, object]:
    errors: list[str] = []
    for path in (dataset_path, agreement_path, split_dir / "split_manifest.json"):
        if not path.is_file():
            errors.append(f"missing:{path}")
    validation = None
    if not errors:
        validation = validate_path(
            dataset_path,
            require_research_gates=True,
            agreement_report_path=agreement_path,
        )
        if validation.status != "PASS":
            errors.append("research_dataset_gate_failed")

    estimate = load_yaml(project_root / "configs/resources/resource_estimate.yaml")
    source_bytes = _tree_size(project_root)
    dataset_bytes = dataset_path.stat().st_size if dataset_path.is_file() else None
    split_bytes = (
        sum(path.stat().st_size for path in split_dir.glob("*.jsonl"))
        if split_dir.is_dir()
        else None
    )
    total_parameters = 278_000_000
    memory_budget = {
        "bf16_weights_gib": total_parameters * 2 / 1024**3,
        "bf16_gradients_gib": total_parameters * 2 / 1024**3,
        "fp32_reference_weights_gib": total_parameters * 4 / 1024**3,
        "adam_moments_gib": total_parameters * 8 / 1024**3,
        "non_activation_training_state_gib": total_parameters * 16 / 1024**3,
        "locked_peak_vram_budget_gib": 16,
    }
    planning_scenarios = estimate.get("timing_estimate", estimate.get("profiles", {}))
    measurement_contract = resource_measurement_contract()
    measurement_classes = measurement_contract["measurement_classes"]
    assert isinstance(measurement_classes, dict)
    measurement_classes["analytical_model_memory_budget"]["value"] = memory_budget
    measurement_classes["planning_scenarios"]["value"] = planning_scenarios
    result = {
        "schema_version": "1.0.0",
        "status": "PASS" if not errors else "FAIL",
        "measurement_kind": "local_analytical_resource_plan",
        "measurement_classes": measurement_classes,
        "errors": errors,
        "project_source_bytes": source_bytes,
        "dataset_bytes": dataset_bytes,
        "split_jsonl_bytes": split_bytes,
        "dataset_sha256": sha256_file(dataset_path) if dataset_path.is_file() else None,
        "split_manifest_sha256": (
            sha256_file(split_dir / "split_manifest.json")
            if (split_dir / "split_manifest.json").is_file()
            else None
        ),
        "research_record_count": validation.research_record_count if validation else None,
        "model_memory_budget": memory_budget,
        "profile_planning": planning_scenarios,
        "hardware_observed": False,
        "compute_hours": None,
        "billed_cost": None,
        "post_run_input_requirements": measurement_contract["post_run_input_requirements"],
        "not_hardware_evidence": True,
        "runtime_requirement": (
            "The execution session must still observe an exact registered NVIDIA A100 80GB "
            "device name, CUDA "
            "compute capability 8.0, at least 70 GiB device memory, BF16 tensor execution, "
            "40 GiB RAM, 80 GiB free disk, and CUDA-only model placement before a full run."
        ),
    }
    write_json(output_path, result)
    return result
