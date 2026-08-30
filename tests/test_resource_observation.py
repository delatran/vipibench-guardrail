import json
from pathlib import Path

import pytest

from vipibench.dataio import canonical_json
from vipibench.modeling import load_yaml
from vipibench.resource_observation import (
    StageResourceObserver,
    build_resource_measurement,
    verify_resource_measurement,
)
from vipibench.runtime_capacity import RuntimeProbe, check_runtime_profile
from vipibench.runtime_storage import create_runtime_storage_plan
from vipibench.runtime_telemetry import build_strict_capacity_receipt


def _runtime_check() -> dict[str, object]:
    profile = load_yaml(Path.cwd() / "configs" / "profiles" / "accelerator_80gb.yaml")
    assert isinstance(profile, dict)
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
            system_ram_gib=167.1,
            disk_free_gib=200.0,
            compute_capability="8.0",
            evidence_kind="observed",
        ),
    )


def _strict_receipt() -> dict[str, object]:
    return build_strict_capacity_receipt(_runtime_check(), project_root=Path.cwd())


def _sample_payload() -> dict[str, object]:
    return {
        "gpu": {
            "index": 0,
            "name": "NVIDIA A100-SXM4-80GB",
            "utilization_percent": 91.0,
            "memory_used_mib": 70123.0,
            "memory_total_mib": 81920.0,
            "power_draw_w": 312.0,
            "power_limit_w": 400.0,
            "temperature_c": 67.0,
            "sm_clock_mhz": 1410.0,
        },
        "host": {
            "cpu_percent": 68.0,
            "logical_cpu_count": 16,
            "load_1m": 7.0,
            "ram_used_gib": 48.0,
            "ram_available_gib": 119.0,
            "ram_percent": 28.7,
            "disk_read_bytes": 1000,
            "disk_write_bytes": 2000,
            "disk_read_time_ms": 10,
            "disk_write_time_ms": 20,
        },
        "paths": {
            "output": {
                "path": "/fixture/output",
                "device_id": 1,
                "total_gib": 235.0,
                "used_gib": 47.0,
                "free_gib": 188.0,
            },
            "ephemeral": {
                "path": "/fixture/scratch",
                "device_id": 2,
                "total_gib": 368.0,
                "used_gib": 1.0,
                "free_gib": 367.0,
            },
        },
        "errors": [],
    }


def _observer_fixture(tmp_path: Path, stage_id: str) -> tuple[StageResourceObserver, Path]:
    output_root = tmp_path / "run-output"
    scratch = tmp_path / "scratch"
    output_root.mkdir()
    scratch.mkdir()
    session_root = output_root / "session_evidence" / "runtime_sessions" / f"s-{stage_id}"
    storage_path = session_root / "storage_plan.json"
    create_runtime_storage_plan(
        session_id=f"s-{stage_id}",
        default_root=tmp_path,
        output_root=output_root,
        protected_roots=[Path.cwd()],
        minimum_free_gib=0.001,
        explicit_root=scratch,
        output_path=storage_path,
        partitions=[
            {"mountpoint": str(tmp_path), "device": "fixture", "fstype": "ext4"},
        ],
    )
    receipt_path = output_root / "strict_capacity_receipt.json"
    receipt_path.write_text(json.dumps(_strict_receipt()), encoding="utf-8")
    session_capacity_check_path = session_root / "accelerator_capacity_check.json"
    session_capacity_check_path.write_text(json.dumps(_runtime_check()), encoding="utf-8")
    observer = StageResourceObserver(
        stage_id=stage_id,
        session_id=f"s-{stage_id}",
        output_root=output_root,
        project_root=Path.cwd(),
        storage_plan_path=storage_path,
        session_capacity_check_path=session_capacity_check_path,
        strict_capacity_receipt_path=receipt_path,
        sample_interval_seconds=60.0,
        sample_provider=_sample_payload,
    )
    return observer, output_root


def test_stage_observation_and_measurement_are_hash_bound(tmp_path: Path) -> None:
    observer, output_root = _observer_fixture(tmp_path, "preflight")
    observer.start()
    summary = observer.stop(stage_execution_status="completed")

    assert summary["status"] == "PASS"
    assert summary["sample_count"] == 2
    assert summary["valid_gpu_sample_count"] == 2
    measurement_path = output_root / "resource_measurement.json"
    measurement = build_resource_measurement(
        output_root=output_root,
        project_root=Path.cwd(),
        expected_public_stages=["preflight"],
        output_path=measurement_path,
    )

    assert measurement["status"] == "PASS"
    assert measurement["hardware_observed"] is True
    assert measurement["completed_public_stages"] == ["preflight"]
    assert verify_resource_measurement(
        measurement,
        output_root=output_root,
        project_root=Path.cwd(),
        expected_public_stages=["preflight"],
    ) == measurement


def test_measurement_remains_partial_until_every_expected_stage_is_observed(
    tmp_path: Path,
) -> None:
    observer, output_root = _observer_fixture(tmp_path, "preflight")
    observer.start()
    observer.stop(stage_execution_status="completed")

    measurement = build_resource_measurement(
        output_root=output_root,
        project_root=Path.cwd(),
        expected_public_stages=["preflight", "data"],
    )

    assert measurement["status"] == "PARTIAL"
    assert measurement["missing_public_stages"] == ["data"]


def test_failed_stage_attempt_is_retained_without_counting_as_complete(tmp_path: Path) -> None:
    observer, output_root = _observer_fixture(tmp_path, "preflight")
    observer.start()
    observer.stop(stage_execution_status="failed")

    measurement = build_resource_measurement(
        output_root=output_root,
        project_root=Path.cwd(),
        expected_public_stages=["preflight"],
    )

    assert measurement["status"] == "PARTIAL"
    assert measurement["failed_stage_attempt_count"] == 1
    assert measurement["completed_public_stages"] == []


def test_raw_sample_tampering_invalidates_the_measurement(tmp_path: Path) -> None:
    observer, output_root = _observer_fixture(tmp_path, "preflight")
    observer.start()
    observer.stop(stage_execution_status="completed")
    measurement = build_resource_measurement(
        output_root=output_root,
        project_root=Path.cwd(),
        expected_public_stages=["preflight"],
    )
    raw_path = observer.raw_samples_path
    rows = raw_path.read_text(encoding="utf-8").splitlines()
    first = json.loads(rows[0])
    first["gpu"]["utilization_percent"] = 1.0
    rows[0] = canonical_json(first)
    raw_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="resource sample hash mismatch"):
        verify_resource_measurement(
            measurement,
            output_root=output_root,
            project_root=Path.cwd(),
            expected_public_stages=["preflight"],
        )


def test_session_capacity_check_tampering_invalidates_the_measurement(tmp_path: Path) -> None:
    observer, output_root = _observer_fixture(tmp_path, "preflight")
    observer.start()
    observer.stop(stage_execution_status="completed")
    measurement = build_resource_measurement(
        output_root=output_root,
        project_root=Path.cwd(),
        expected_public_stages=["preflight"],
    )
    capacity = json.loads(observer.session_capacity_check_path.read_text(encoding="utf-8"))
    capacity["probe"]["device_name"] = "tampered-device"
    observer.session_capacity_check_path.write_text(json.dumps(capacity), encoding="utf-8")

    with pytest.raises(ValueError, match="strict A100 profile"):
        verify_resource_measurement(
            measurement,
            output_root=output_root,
            project_root=Path.cwd(),
            expected_public_stages=["preflight"],
        )


def test_missing_gpu_observation_fails_closed(tmp_path: Path) -> None:
    observer, output_root = _observer_fixture(tmp_path, "preflight")
    observer._sample_provider = lambda: {  # noqa: SLF001 - explicit failure injection.
        "gpu": None,
        "host": {},
        "paths": {},
        "errors": ["nvidia_smi:missing"],
    }
    observer.start()
    summary = observer.stop(stage_execution_status="completed")

    assert summary["status"] == "FAIL"
    with pytest.raises(ValueError, match="summary is not PASS"):
        build_resource_measurement(
            output_root=output_root,
            project_root=Path.cwd(),
            expected_public_stages=["preflight"],
        )
