import json
from pathlib import Path

import pytest

from vipibench.checkpoint import StageLedger
from vipibench.dataio import sha256_file, write_json
from vipibench.manifest import runtime_source_fingerprint
from vipibench.modeling import load_yaml
from vipibench.runtime_capacity import RuntimeProbe, check_runtime_profile
from vipibench.runtime_telemetry import (
    build_strict_capacity_receipt,
    strict_capacity_receipt_sha256,
)
from vipibench.runtime_transition import (
    check_analysis_cpu_runtime,
    load_bound_accelerator_preflight,
    write_cpu_analysis_transition_receipt,
)
from vipibench.stage_orchestration import complete_stage_group, load_stage_plan

PROJECT_ROOT = Path.cwd()
STAGE_PLAN_PATH = PROJECT_ROOT / "configs/resources/confirmatory_stage_plan.json"


def _cpu_probe() -> RuntimeProbe:
    return RuntimeProbe(
        compute_available=False,
        device_type=None,
        device_name=None,
        device_index=None,
        device_memory_gib=0.0,
        bf16_supported=False,
        tensor_probe_passed=False,
        system_ram_gib=12.0,
        disk_free_gib=40.0,
        compute_capability=None,
        evidence_kind="observed",
    )


def _a100_check() -> dict[str, object]:
    profile = load_yaml(PROJECT_ROOT / "configs/profiles/accelerator_80gb.yaml")
    return check_runtime_profile(
        profile,
        RuntimeProbe(
            compute_available=True,
            device_type="cuda",
            device_name="NVIDIA A100-SXM4-80GB",
            device_index=0,
            device_memory_gib=79.2,
            bf16_supported=True,
            tensor_probe_passed=True,
            system_ram_gib=53.0,
            disk_free_gib=120.0,
            compute_capability="8.0",
            evidence_kind="observed",
        ),
    )


def _build_prior_stage_chain(run_root: Path) -> tuple[dict[str, object], str]:
    plan = load_stage_plan(STAGE_PLAN_PATH)
    binding = {
        "runtime_source_fingerprint": runtime_source_fingerprint(PROJECT_ROOT),
        "launch_hashes": {"artifact_manifest": "A" * 64},
        "launch_authorization_sha256": "B" * 64,
        "stage_plan_sha256": sha256_file(STAGE_PLAN_PATH),
        "protocol_amendment": "fixture-amendment",
        "durable_lineage": "fixture-lineage",
    }
    prior_session = "20260809T010203.000000Z-17"
    prior_root = run_root / "session_evidence/runtime_sessions" / prior_session
    prior_root.mkdir(parents=True)
    preflight_path = prior_root / "prelaunch_readiness.json"
    accelerator_path = prior_root / "resource_measurement.json"
    write_json(
        preflight_path,
        {
            "status": "PASS",
            "milestone": "READY_FOR_CONFIRMATORY_LAUNCH",
            "launch_hashes": {
                **binding["launch_hashes"],
                "runtime_source_fingerprint": binding["runtime_source_fingerprint"],
            },
        },
    )
    binding["launch_hashes"] = json.loads(preflight_path.read_text())["launch_hashes"]
    accelerator = _a100_check()
    write_json(accelerator_path, accelerator)
    strict_receipt = build_strict_capacity_receipt(accelerator, project_root=PROJECT_ROOT)
    strict_path = run_root / "strict_capacity_receipt.json"
    write_json(strict_path, strict_receipt)
    launch_path = run_root / "launch_record.json"
    write_json(
        launch_path,
        {
            "schema_version": "1.0.0",
            "status": "PASS",
            "mode": "confirmatory",
            "session_id": prior_session,
            "selected_public_stage": "attack-evaluate",
            "launch_hashes": binding["launch_hashes"],
            "preflight_sha256": sha256_file(preflight_path),
            "accelerator_sha256": sha256_file(accelerator_path),
            "strict_capacity_receipt_sha256": strict_capacity_receipt_sha256(
                strict_receipt,
                project_root=PROJECT_ROOT,
            ),
        },
    )

    stage_ledger = StageLedger(run_root / "orchestration_ledger", artifact_root=run_root)
    group_ledger = StageLedger(run_root / "stage_group_ledger", artifact_root=run_root)
    for spec in plan["stages"]:
        stage_id = str(spec["id"])
        if stage_id == "analysis":
            break
        for member in spec["member_stage_ids"]:
            output = run_root / "fixture_stage_outputs" / f"{member}.json"
            write_json(output, {"status": "PASS", "stage": member})
            stage_ledger.complete(
                str(member),
                [output],
                {"command": ["fixture"], **binding},
            )
        direct_outputs = (
            [preflight_path, accelerator_path, strict_path, launch_path]
            if stage_id == "preflight"
            else []
        )
        complete_stage_group(
            plan=plan,
            stage_id=stage_id,
            stage_ledger=stage_ledger,
            group_ledger=group_ledger,
            output_root=run_root,
            run_binding=binding,
            direct_outputs=direct_outputs,
        )
    return binding, prior_session


def test_analysis_cpu_profile_accepts_only_observed_no_accelerator_runtime() -> None:
    profile = load_yaml(PROJECT_ROOT / "configs/profiles/analysis_cpu.yaml")

    passed = check_analysis_cpu_runtime(profile, _cpu_probe())

    assert passed["status"] == "PASS"
    assert passed["runtime_observed"] is True
    assert passed["accelerator_hardware_observed"] is False
    assert passed["accelerator_work_allowed"] is False

    gpu_probe = RuntimeProbe(
        **{
            **_cpu_probe().as_dict(),
            "compute_available": True,
            "device_type": "cuda",
            "device_name": "NVIDIA T4",
            "device_index": 0,
            "device_memory_gib": 15.0,
            "compute_capability": "7.5",
        }
    )
    failed = check_analysis_cpu_runtime(profile, gpu_probe)
    assert failed["status"] == "FAIL"
    assert "analysis_cpu_accelerator_present" in failed["errors"]


def test_bound_accelerator_preflight_rejects_measurement_tampering(tmp_path: Path) -> None:
    run_root = tmp_path / "run-output"
    _binding, prior_session = _build_prior_stage_chain(run_root)

    loaded = load_bound_accelerator_preflight(
        project_root=PROJECT_ROOT,
        output_root=run_root,
        selected_cpu_stage="analysis",
    )
    assert loaded["prior_accelerator_session_id"] == prior_session

    measurement_path = run_root / "session_evidence/runtime_sessions" / prior_session
    measurement_path /= "resource_measurement.json"
    measurement = json.loads(measurement_path.read_text())
    measurement["probe"]["device_name"] = "NVIDIA T4"
    write_json(measurement_path, measurement)
    with pytest.raises(ValueError, match="measurement hash mismatch"):
        load_bound_accelerator_preflight(
            project_root=PROJECT_ROOT,
            output_root=run_root,
            selected_cpu_stage="analysis",
        )


def test_cpu_analysis_transition_binds_current_environment_and_prior_stage_chain(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run-output"
    binding, prior_session = _build_prior_stage_chain(run_root)
    session_id = "20260809T020304.000000Z-23"
    session_root = run_root / "session_evidence/runtime_sessions" / session_id
    runtime_path = session_root / "analysis_runtime_measurement.json"
    environment_path = session_root / "analysis_environment_compatibility.json"
    output_path = session_root / "cpu_analysis_transition.json"
    write_json(
        runtime_path,
        check_analysis_cpu_runtime(
            load_yaml(PROJECT_ROOT / "configs/profiles/analysis_cpu.yaml"),
            _cpu_probe(),
        ),
    )
    write_json(
        environment_path,
        {
            "status": "PASS",
            "dependency_profile": "analysis-cpu",
            "runtime_source_fingerprint": runtime_source_fingerprint(PROJECT_ROOT),
            "accelerator_stack_present": False,
        },
    )

    receipt = write_cpu_analysis_transition_receipt(
        selected_stage="analysis",
        session_id=session_id,
        project_root=PROJECT_ROOT,
        output_root=run_root,
        stage_plan_path=STAGE_PLAN_PATH,
        run_binding=binding,
        runtime_check_path=runtime_path,
        environment_compatibility_path=environment_path,
        output_path=output_path,
    )

    assert receipt["status"] == "PASS"
    assert receipt["execution_role"] == "cpu_analysis_only"
    assert receipt["prior_accelerator_session_id"] == prior_session
    assert set(receipt["verified_prior_stage_receipts"]) == {
        "preflight",
        "data",
        "baselines",
        "encoder",
        "core",
        "attack-generate",
        "attack-evaluate",
    }
    assert output_path.is_file()


def test_cpu_analysis_transition_rejects_accelerator_stage(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="only for analysis or finalize"):
        write_cpu_analysis_transition_receipt(
            selected_stage="encoder",
            session_id="fixture",
            project_root=PROJECT_ROOT,
            output_root=tmp_path,
            stage_plan_path=STAGE_PLAN_PATH,
            run_binding={},
            runtime_check_path=tmp_path / "runtime.json",
            environment_compatibility_path=tmp_path / "environment.json",
            output_path=tmp_path / "receipt.json",
        )
