from pathlib import Path

import yaml

from vipibench.resource_estimate import validate_resource_estimate
from vipibench.resources import measure_resources, resource_measurement_contract


def test_live_resource_estimate_reproduces_locked_workload_math(tmp_path: Path) -> None:
    output = tmp_path / "resource_estimate_validation.json"

    result = validate_resource_estimate(
        Path.cwd(),
        Path("configs/resources/resource_estimate.yaml"),
        output,
    )

    assert result["status"] == "PASS", result["errors"]
    assert result["calculated"]["generated_attack_candidates"] == 4800
    assert result["calculated"]["target_agent_calls"] == 5280
    assert result["calculated"]["total_generated_token_upper_bound"] == 23470080
    assert result["calculated"]["maximum_confirmatory_trajectories"] == 16320
    assert result["live_measurement_required"] is True
    assert output.is_file()


def test_resource_estimate_rejects_optimistic_token_budget(tmp_path: Path) -> None:
    config = yaml.safe_load(
        Path("configs/resources/resource_estimate.yaml").read_text(encoding="utf-8")
    )
    config["generation_upper_bound"]["total_generated_tokens"] = 1000
    path = tmp_path / "resource_estimate.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = validate_resource_estimate(Path.cwd(), path)

    assert result["status"] == "FAIL"
    assert "generation_upper_bound_mismatch:total_generated_tokens" in result["errors"]


def test_resource_estimate_rejects_a_non_accelerator_hardware_contract(tmp_path: Path) -> None:
    config = yaml.safe_load(
        Path("configs/resources/resource_estimate.yaml").read_text(encoding="utf-8")
    )
    config["hardware_contract_profile"] = "configs/profiles/standard_compute.yaml"
    path = tmp_path / "resource_estimate.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = validate_resource_estimate(Path.cwd(), path)

    assert result["status"] == "FAIL"
    assert "hardware_contract_profile_must_bind_accelerator_80gb" in result["errors"]


def test_resource_measurement_contract_separates_observations_and_estimates() -> None:
    contract = resource_measurement_contract()
    measurement_classes = contract["measurement_classes"]

    assert set(measurement_classes) == {
        "observed_hardware_telemetry",
        "observed_runtime_timing",
        "observed_public_stage_resources",
        "analytical_model_memory_budget",
        "planning_scenarios",
    }
    assert contract["hardware_observed"] is False
    assert contract["compute_hours"] is None
    assert contract["billed_cost"] is None
    requirements = {item["name"]: item for item in contract["post_run_input_requirements"]}
    strict_receipt = requirements["strict_runtime_capacity_receipt"]
    telemetry = requirements["runtime_telemetry_ledger"]
    stage_resources = requirements["public_stage_resource_measurement"]
    assert strict_receipt["hash_bound"] is True
    assert strict_receipt["schema_version"] == "1.0.0"
    assert strict_receipt["receipt_type"] == "strict_runtime_capacity_80gb"
    assert strict_receipt["required_profile"] == "accelerator_80gb"
    assert strict_receipt["profile_hash_bound"] is True
    assert strict_receipt["runtime_source_hash_bound"] is True
    assert strict_receipt["unknown_fields_rejected"] is True
    assert telemetry["hash_bound"] is True
    assert telemetry["schema_version"] == "1.0.0"
    assert telemetry["cross_run_resume_policy"] == "reject"
    assert telemetry["empty_live_ledger_policy"] == "reject"
    assert telemetry["unknown_fields_rejected"] is True
    assert stage_resources["required_status"] == "PASS"
    assert stage_resources["required_completed_public_stage_count"] == 9
    assert stage_resources["raw_sample_reverification_required"] is True


def test_local_resource_measurement_keeps_live_claims_null(tmp_path: Path) -> None:
    result = measure_resources(
        Path.cwd(),
        tmp_path / "missing-dataset.jsonl",
        tmp_path / "missing-agreement.json",
        tmp_path / "missing-splits",
        tmp_path / "resource-measurement.json",
    )

    assert result["status"] == "FAIL"
    assert result["measurement_kind"] == "local_analytical_resource_plan"
    assert result["hardware_observed"] is False
    assert result["compute_hours"] is None
    assert result["billed_cost"] is None
    assert (
        result["measurement_classes"]["planning_scenarios"]["evidence"] == "resource_estimate_yaml"
    )
    requirements = {item["name"]: item for item in result["post_run_input_requirements"]}
    assert "runtime_telemetry_ledger" in requirements
    assert "public_stage_resource_measurement" in requirements
