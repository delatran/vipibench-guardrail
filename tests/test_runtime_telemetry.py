import hashlib
import json
from pathlib import Path

import pytest

from vipibench.dataio import canonical_json
from vipibench.modeling import load_yaml
from vipibench.runtime_capacity import RuntimeProbe, check_runtime_profile
from vipibench.runtime_telemetry import (
    build_strict_capacity_receipt,
    build_telemetry_ledger,
    consolidate_live_telemetry_ledgers,
    record_stage_interval,
    strict_capacity_receipt_bindings,
    strict_capacity_receipt_sha256,
    verify_telemetry_ledger,
    write_telemetry_ledger,
)

INPUT_HASH = "A" * 64
OUTPUT_HASH = "B" * 64


def strict_receipt() -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "receipt_type": "strict_runtime_capacity_80gb",
        "status": "PASS",
        "errors": [],
        "profile": "accelerator_80gb",
        **strict_capacity_receipt_bindings(Path.cwd()),
        "hardware_observed": True,
        "probe": {
            "compute_available": True,
            "device_type": "cuda",
            "device_name": "NVIDIA A100-SXM4-80GB",
            "device_index": 0,
            "device_memory_gib": 79.2,
            "bf16_supported": True,
            "tensor_probe_passed": True,
            "system_ram_gib": 53.0,
            "disk_free_gib": 120.0,
            "compute_capability": "8.0",
            "evidence_kind": "observed",
        },
    }


def canonical_strict_receipt() -> dict[str, object]:
    profile = load_yaml(Path.cwd() / "configs" / "profiles" / "accelerator_80gb.yaml")
    assert isinstance(profile, dict)
    probe_payload = strict_receipt()["probe"]
    assert isinstance(probe_payload, dict)
    runtime_check = check_runtime_profile(profile, RuntimeProbe(**probe_payload))
    return build_strict_capacity_receipt(runtime_check, project_root=Path.cwd())


def interval(
    interval_id: str,
    start: float,
    end: float,
    *,
    receipt_hash: str | None,
    status: str = "completed",
    resumed_from: tuple[str, ...] = (),
    accelerator_stage: bool = True,
    run_id: str = "run-001",
) -> dict[str, object]:
    return record_stage_interval(
        stage_id="confirmatory-evaluation",
        interval_id=interval_id,
        run_id=run_id,
        attempt_id=f"attempt-{interval_id}",
        start_monotonic_seconds=start,
        end_monotonic_seconds=end,
        status=status,
        accelerator_stage=accelerator_stage,
        observed_device_receipt_sha256=receipt_hash,
        input_artifact_hashes={"input": INPUT_HASH},
        output_artifact_hashes=({"output": OUTPUT_HASH} if status == "completed" else {}),
        resumed_from_interval_ids=resumed_from,
    )


def payload_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest().upper()


def test_resume_deduplicates_replayed_intervals_and_computes_exact_hours() -> None:
    receipt = strict_receipt()
    receipt_hash = strict_capacity_receipt_sha256(receipt)
    first = interval("interval-1", 100.0, 3700.0, receipt_hash=receipt_hash)
    interrupted = interval(
        "interval-2",
        3700.0,
        5500.0,
        receipt_hash=receipt_hash,
        status="failed",
        resumed_from=("interval-1",),
    )
    resumed = interval(
        "interval-3",
        5500.0,
        7300.0,
        receipt_hash=receipt_hash,
        resumed_from=("interval-1", "interval-2"),
    )

    ledger = build_telemetry_ledger(
        [first, first, interrupted, resumed],
        strict_capacity_receipt=receipt,
        local_only=False,
    )

    assert ledger["validation_status"] == "PASS"
    assert ledger["execution_status"] == "failed"
    assert ledger["record_count"] == 4
    assert ledger["unique_interval_count"] == 3
    assert ledger["deduplicated_replay_count"] == 1
    assert ledger["deduplicated_interval_ids"] == ["interval-1"]
    assert ledger["accelerator_elapsed_seconds"] == 7200.0
    assert ledger["compute_hours"] == 2.0
    assert ledger["billed_cost"] is None
    assert ledger["hardware_observed"] is True
    assert verify_telemetry_ledger(ledger, strict_capacity_receipt=receipt) == ledger


def test_overlap_and_negative_duration_fail_closed() -> None:
    receipt = strict_receipt()
    receipt_hash = strict_capacity_receipt_sha256(receipt)
    first = interval("interval-1", 10.0, 20.0, receipt_hash=receipt_hash)
    overlap = interval("interval-2", 19.0, 30.0, receipt_hash=receipt_hash)

    with pytest.raises(ValueError, match="overlapping stage intervals"):
        build_telemetry_ledger(
            [first, overlap],
            strict_capacity_receipt=receipt,
            local_only=False,
        )
    with pytest.raises(ValueError, match="negative duration"):
        interval("negative", 20.0, 19.0, receipt_hash=receipt_hash)


def test_local_only_artifact_cannot_claim_hardware_or_compute_hours() -> None:
    local_timing = interval(
        "local-validation",
        1.0,
        11.0,
        receipt_hash=None,
        accelerator_stage=False,
    )

    ledger = build_telemetry_ledger([local_timing], local_only=True)

    assert ledger["observed_runtime_seconds"] == 10.0
    assert ledger["hardware_observed"] is False
    assert ledger["strict_capacity_receipt_sha256"] is None
    assert ledger["accelerator_elapsed_seconds"] is None
    assert ledger["compute_hours"] is None
    assert ledger["billed_cost"] is None
    with pytest.raises(ValueError, match="local-only ledger"):
        build_telemetry_ledger(
            [local_timing],
            strict_capacity_receipt=strict_receipt(),
            local_only=True,
        )


def test_mock_capacity_result_cannot_set_hardware_observed() -> None:
    receipt = strict_receipt()
    probe = receipt["probe"]
    assert isinstance(probe, dict)
    probe["evidence_kind"] = "mock_test"

    with pytest.raises(ValueError, match="capacity_probe_not_observed"):
        strict_capacity_receipt_sha256(receipt)


def test_canonical_runtime_capacity_result_is_the_only_live_receipt_producer() -> None:
    receipt = canonical_strict_receipt()
    assert receipt["receipt_type"] == "strict_runtime_capacity_80gb"
    assert receipt["status"] == "PASS"
    assert receipt["hardware_observed"] is True
    assert strict_capacity_receipt_sha256(receipt) == payload_sha256(receipt)

    profile = load_yaml(Path.cwd() / "configs" / "profiles" / "accelerator_80gb.yaml")
    assert isinstance(profile, dict)
    probe_payload = strict_receipt()["probe"]
    assert isinstance(probe_payload, dict)
    probe_payload["device_name"] = "NVIDIA unsupported accelerator"
    failed = check_runtime_profile(profile, RuntimeProbe(**probe_payload))
    assert failed["hardware_observed"] is False
    with pytest.raises(ValueError, match="not observed PASS"):
        build_strict_capacity_receipt(failed, project_root=Path.cwd())


def test_strict_receipt_requires_versioned_l4_profile_source_bindings_and_device_index() -> None:
    receipt = strict_receipt()
    probe = receipt["probe"]
    assert isinstance(probe, dict)
    probe["device_index"] = None

    with pytest.raises(ValueError, match="capacity_probe_device_index_invalid"):
        strict_capacity_receipt_sha256(receipt)

    version_mismatch = strict_receipt()
    version_mismatch["schema_version"] = "2.0.0"
    with pytest.raises(ValueError, match="schema version mismatch"):
        strict_capacity_receipt_sha256(version_mismatch)

    non_l4 = strict_receipt()
    unregistered_probe = non_l4["probe"]
    assert isinstance(unregistered_probe, dict)
    unregistered_probe["device_name"] = "NVIDIA unsupported accelerator"
    with pytest.raises(ValueError, match="device_name_mismatch"):
        strict_capacity_receipt_sha256(non_l4)

    for field in (
        "profile_sha256",
        "runtime_source_fingerprint",
        "runtime_capacity_source_sha256",
    ):
        tampered = strict_receipt()
        tampered[field] = "0" * 64
        with pytest.raises(ValueError, match=field):
            strict_capacity_receipt_sha256(tampered)


def test_live_ledger_rejects_empty_intervals_instead_of_claiming_completed_zero_hours() -> None:
    receipt = strict_receipt()

    with pytest.raises(ValueError, match="at least one accelerator interval"):
        build_telemetry_ledger(
            [],
            strict_capacity_receipt=receipt,
            local_only=False,
        )


def test_cross_run_resume_reference_fails_closed() -> None:
    receipt = strict_receipt()
    receipt_hash = strict_capacity_receipt_sha256(receipt)
    first = interval(
        "run-a-interval",
        10.0,
        20.0,
        receipt_hash=receipt_hash,
        run_id="run-a",
    )
    resumed = interval(
        "run-b-interval",
        20.0,
        30.0,
        receipt_hash=receipt_hash,
        run_id="run-b",
        resumed_from=("run-a-interval",),
    )

    with pytest.raises(ValueError, match="cross-run resume reference"):
        build_telemetry_ledger(
            [first, resumed],
            strict_capacity_receipt=receipt,
            local_only=False,
        )


def test_unknown_fields_fail_even_when_stage_or_ledger_hash_is_recomputed() -> None:
    local_timing = interval(
        "local-validation",
        1.0,
        11.0,
        receipt_hash=None,
        accelerator_stage=False,
    )
    unknown_stage = {**local_timing, "unexpected": "tampered"}
    unknown_stage["record_sha256"] = payload_sha256(
        {key: value for key, value in unknown_stage.items() if key != "record_sha256"}
    )
    with pytest.raises(ValueError, match="stage record unknown fields"):
        build_telemetry_ledger([unknown_stage], local_only=True)

    ledger = build_telemetry_ledger([local_timing], local_only=True)
    unknown_ledger = {**ledger, "unexpected": "tampered"}
    unknown_ledger["ledger_sha256"] = payload_sha256(
        {key: value for key, value in unknown_ledger.items() if key != "ledger_sha256"}
    )
    with pytest.raises(ValueError, match="ledger unknown fields"):
        verify_telemetry_ledger(unknown_ledger)


def test_unknown_strict_receipt_field_cannot_be_hashed_as_evidence() -> None:
    receipt = {**strict_receipt(), "unexpected": "tampered"}

    with pytest.raises(ValueError, match="strict capacity receipt unknown fields"):
        strict_capacity_receipt_sha256(receipt)


def test_ledger_and_stage_hashes_detect_tampering() -> None:
    local_timing = interval(
        "local-validation",
        1.0,
        11.0,
        receipt_hash=None,
        accelerator_stage=False,
    )
    ledger = build_telemetry_ledger([local_timing], local_only=True)
    tampered = {**ledger, "observed_runtime_seconds": 1000.0}

    with pytest.raises(ValueError, match="ledger hash mismatch"):
        verify_telemetry_ledger(tampered)

    tampered_record = {**local_timing, "stage_id": "other-stage"}
    with pytest.raises(ValueError, match="stage record hash mismatch"):
        build_telemetry_ledger([tampered_record], local_only=True)


def test_write_telemetry_ledger_persists_verifiable_hash_bound_artifact(
    tmp_path: Path,
) -> None:
    output = tmp_path / "runtime_telemetry.json"
    local_timing = interval(
        "local-validation",
        2.0,
        5.0,
        receipt_hash=None,
        accelerator_stage=False,
    )

    ledger = write_telemetry_ledger(output, [local_timing], local_only=True)

    assert output.is_file()
    assert len(str(ledger["ledger_sha256"])) == 64
    assert verify_telemetry_ledger(ledger) == ledger


def test_consolidated_live_telemetry_requires_one_verified_receipt(tmp_path: Path) -> None:
    receipt = canonical_strict_receipt()
    receipt_path = tmp_path / "strict_capacity_receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    receipt_hash = strict_capacity_receipt_sha256(receipt)
    first = build_telemetry_ledger(
        [interval("core-interval", 10.0, 20.0, receipt_hash=receipt_hash, run_id="core")],
        strict_capacity_receipt=receipt,
        local_only=False,
    )
    second = build_telemetry_ledger(
        [interval("attack-interval", 20.0, 35.0, receipt_hash=receipt_hash, run_id="attack")],
        strict_capacity_receipt=receipt,
        local_only=False,
    )
    first_path = tmp_path / "core.telemetry.json"
    second_path = tmp_path / "attack.telemetry.json"
    first_path.write_text(json.dumps(first), encoding="utf-8")
    second_path.write_text(json.dumps(second), encoding="utf-8")

    output = tmp_path / "runtime_telemetry.json"
    merged = consolidate_live_telemetry_ledgers(
        [first_path, second_path],
        strict_capacity_receipt_path=receipt_path,
        output_path=output,
        project_root=Path.cwd(),
    )

    assert output.is_file()
    assert merged["record_count"] == 2
    assert merged["compute_hours"] == pytest.approx(25.0 / 3600.0)
    assert verify_telemetry_ledger(merged, strict_capacity_receipt=receipt) == merged

    mismatched_receipt = canonical_strict_receipt()
    mismatched_probe = mismatched_receipt["probe"]
    assert isinstance(mismatched_probe, dict)
    mismatched_probe["disk_free_gib"] = 121.0
    mismatched_hash = strict_capacity_receipt_sha256(mismatched_receipt)
    mismatched_ledger = build_telemetry_ledger(
        [
            interval(
                "mismatched-interval",
                35.0,
                45.0,
                receipt_hash=mismatched_hash,
                run_id="mismatched",
            )
        ],
        strict_capacity_receipt=mismatched_receipt,
        local_only=False,
    )
    second_path.write_text(json.dumps(mismatched_ledger), encoding="utf-8")

    with pytest.raises(ValueError, match="stage receipt binding mismatch"):
        consolidate_live_telemetry_ledgers(
            [first_path, second_path],
            strict_capacity_receipt_path=receipt_path,
            output_path=output,
            project_root=Path.cwd(),
        )
