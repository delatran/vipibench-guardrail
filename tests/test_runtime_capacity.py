import pytest

from vipibench.runtime_capacity import (
    CapacityMeasurement,
    RuntimeProbe,
    check_runtime_profile,
    is_capacity_exhaustion,
    rank_capacity_candidates,
    select_capacity_candidate,
    validate_model_device_placement,
)

PROFILE = {
    "name": "accelerator_80gb",
    "compute_required": True,
    "minimum_device_memory_gib": 70,
    "minimum_system_ram_gib": 40,
    "minimum_disk_free_gib": 80,
    "bf16_required": True,
    "tensor_probe_required": True,
}

STRICT = {
    **PROFILE,
    "required_device_type": "cuda",
    "required_device_names": [
        "NVIDIA A100-SXM4-80GB",
        "NVIDIA A100-PCIE-80GB",
        "NVIDIA A100 80GB",
    ],
    "required_compute_capability": "8.0",
    "maximum_device_memory_gib": 82,
}


def probe(**updates: object) -> RuntimeProbe:
    values: dict[str, object] = {
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
        "evidence_kind": "mock_test",
    }
    values.update(updates)
    return RuntimeProbe(**values)


def test_host_only_runtime_fails_compute_profile() -> None:
    result = check_runtime_profile(
        PROFILE,
        probe(
            compute_available=False,
            device_type=None,
            device_index=None,
            device_memory_gib=0.0,
            bf16_supported=False,
            tensor_probe_passed=False,
            compute_capability=None,
        ),
    )
    assert result["status"] == "FAIL"
    assert "compute_device_unavailable" in result["errors"]
    assert result["hardware_observed"] is False


def test_failed_observed_probe_cannot_elevate_hardware_evidence() -> None:
    result = check_runtime_profile(
        STRICT,
        probe(device_name="NVIDIA L4", evidence_kind="observed"),
    )
    assert result["status"] == "FAIL"
    assert result["hardware_observed"] is False


def test_insufficient_device_memory_fails_closed() -> None:
    result = check_runtime_profile(PROFILE, probe(device_memory_gib=69.9))
    assert result["status"] == "FAIL"
    assert "device_memory_below_minimum" in result["errors"]


def test_wrong_device_type_fails_strict_profile() -> None:
    result = check_runtime_profile(
        STRICT,
        probe(device_type="xpu", compute_capability=None),
    )
    assert result["status"] == "FAIL"
    assert "device_type_mismatch" in result["errors"]
    assert "compute_capability_mismatch" in result["errors"]


def test_wrong_compute_capability_fails_identity_gate() -> None:
    result = check_runtime_profile(
        STRICT,
        probe(compute_capability="8.9", device_memory_gib=22.5),
    )
    assert result["status"] == "FAIL"
    assert "compute_capability_mismatch" in result["errors"]


@pytest.mark.parametrize(
    "device_name",
    ["NVIDIA GeForce RTX 4090", "NVIDIA L40", "NVIDIA L4", "Tesla T4"],
)
def test_unregistered_device_names_fail_even_when_other_boundaries_match(
    device_name: str,
) -> None:
    result = check_runtime_profile(STRICT, probe(device_name=device_name))
    assert result["status"] == "FAIL"
    assert "device_name_mismatch" in result["errors"]


def test_oversized_device_memory_fails_upper_bound() -> None:
    result = check_runtime_profile(STRICT, probe(device_memory_gib=96.0))
    assert result["status"] == "FAIL"
    assert "device_memory_above_maximum" in result["errors"]


def test_below_strict_memory_gate_fails() -> None:
    result = check_runtime_profile(STRICT, probe(device_memory_gib=69.9))
    assert result["status"] == "FAIL"
    assert "device_memory_below_minimum" in result["errors"]


def test_mock_passes_logic_but_is_not_observed_evidence() -> None:
    result = check_runtime_profile(STRICT, probe())
    assert result["status"] == "PASS"
    assert result["hardware_observed"] is False


def test_standard_profile_can_use_non_bf16_without_faking_bf16() -> None:
    standard = {
        "name": "standard_compute",
        "compute_required": True,
        "required_device_type": "cuda",
        "minimum_device_memory_gib": 15,
        "bf16_preferred": True,
        "tensor_probe_required": False,
    }
    result = check_runtime_profile(
        standard,
        probe(
            device_memory_gib=15.8,
            bf16_supported=False,
            tensor_probe_passed=False,
            compute_capability="7.5",
        ),
    )
    assert result["status"] == "PASS"
    assert result["probe"]["bf16_supported"] is False


@pytest.mark.parametrize(
    ("updates", "expected_error"),
    [
        ({"bf16_supported": False}, "bf16_probe_failed"),
        ({"tensor_probe_passed": False}, "tensor_probe_failed"),
        ({"system_ram_gib": 39.9}, "system_ram_below_minimum"),
        ({"disk_free_gib": 79.9}, "disk_free_below_minimum"),
    ],
)
def test_strict_profile_fails_every_resource_boundary(
    updates: dict[str, object],
    expected_error: str,
) -> None:
    result = check_runtime_profile(STRICT, probe(**updates))

    assert result["status"] == "FAIL"
    assert expected_error in result["errors"]


def test_capacity_selection_maximizes_measured_throughput_with_reserve() -> None:
    result = select_capacity_candidate(
        [
            CapacityMeasurement("b8", 8, 90.0, 30.0, 80.0),
            CapacityMeasurement("b16", 16, 145.0, 68.0, 80.0),
            CapacityMeasurement("b24", 24, 150.0, 75.0, 80.0),
        ],
        maximum_utilization=0.88,
    )
    assert result["status"] == "PASS"
    assert result["selected"]["candidate_id"] == "b16"


def test_capacity_selection_fails_when_every_candidate_exceeds_reserve() -> None:
    result = select_capacity_candidate(
        [CapacityMeasurement("oversized", 32, 160.0, 76.0, 80.0)],
        maximum_utilization=0.88,
    )
    assert result["status"] == "FAIL"
    assert result["selected"] is None


def test_capacity_ranking_preserves_stable_fallback_order() -> None:
    measurements = [
        CapacityMeasurement("batch-64-checkpoint-on", 64, 125.0, 12.0, 22.0),
        CapacityMeasurement("batch-32-checkpoint-on", 32, 120.0, 7.0, 22.0),
        CapacityMeasurement("batch-64-checkpoint-off", 64, 102.0, 12.1, 22.0),
        CapacityMeasurement("batch-128-checkpoint-on", 128, 0.0, 22.0, 22.0, False),
    ]

    ranked = rank_capacity_candidates(measurements, maximum_utilization=0.88)

    assert [item.candidate_id for item in ranked] == [
        "batch-64-checkpoint-on",
        "batch-32-checkpoint-on",
        "batch-64-checkpoint-off",
    ]


def test_capacity_probe_only_suppresses_allocation_failures() -> None:
    class FakeTorch:
        class OutOfMemoryError(RuntimeError):
            pass

    assert is_capacity_exhaustion(FakeTorch, FakeTorch.OutOfMemoryError("capacity"))
    assert is_capacity_exhaustion(FakeTorch, RuntimeError("out of memory"))
    assert not is_capacity_exhaustion(FakeTorch, RuntimeError("invalid tensor shape"))


class _FakeParameter:
    def __init__(self, device: str) -> None:
        self.device = device


class _FakeModel:
    def __init__(
        self,
        device_map: dict[str, object] | None = None,
        devices: list[str] | None = None,
    ) -> None:
        self.hf_device_map = device_map
        self._devices = devices or []

    def parameters(self):
        return iter(_FakeParameter(device) for device in self._devices)


def test_model_device_map_accepts_cuda_only() -> None:
    result = validate_model_device_placement(
        _FakeModel({"embed": 0, "layers": "cuda:0"}), model_label="fixture"
    )
    assert result["status"] == "PASS"
    assert result["normalized_placements"] == ["cuda"]


@pytest.mark.parametrize("placement", ["cpu", "disk", "meta", "xpu:0"])
def test_model_device_map_rejects_offload_and_unknown_devices(placement: str) -> None:
    with pytest.raises(RuntimeError, match="model_offload_or_non_cuda_placement"):
        validate_model_device_placement(
            _FakeModel({"embed": "cuda:0", "layers": placement}), model_label="fixture"
        )


def test_parameter_fallback_rejects_mixed_cpu_cuda() -> None:
    with pytest.raises(RuntimeError, match="model_offload_or_non_cuda_placement"):
        validate_model_device_placement(
            _FakeModel(devices=["cuda:0", "cpu"]), model_label="fixture"
        )
