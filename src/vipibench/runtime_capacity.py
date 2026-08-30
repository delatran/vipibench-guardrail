from __future__ import annotations

import ctypes
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from vipibench.modeling import load_yaml


@dataclass(frozen=True)
class RuntimeProbe:
    compute_available: bool
    device_type: str | None
    device_name: str | None
    device_index: int | None
    device_memory_gib: float
    bf16_supported: bool
    tensor_probe_passed: bool
    system_ram_gib: float
    disk_free_gib: float
    compute_capability: str | None = None
    evidence_kind: str = "observed"

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CapacityMeasurement:
    candidate_id: str
    batch_size: int
    samples_per_second: float
    peak_reserved_gib: float
    total_memory_gib: float
    completed: bool = True

    @property
    def utilization(self) -> float:
        if self.total_memory_gib <= 0:
            return float("inf")
        return self.peak_reserved_gib / self.total_memory_gib


def is_capacity_exhaustion(torch: Any, exc: RuntimeError) -> bool:
    """Return true only for allocation failures that a smaller candidate may resolve."""

    exhaustion_type = getattr(torch, "OutOfMemoryError", None)
    if exhaustion_type is not None and isinstance(exc, exhaustion_type):
        return True
    message = str(exc).lower()
    return "out of memory" in message or "memory allocation" in message


def select_capacity_candidate(
    measurements: list[CapacityMeasurement],
    *,
    maximum_utilization: float = 0.88,
) -> dict[str, object]:
    eligible = rank_capacity_candidates(
        measurements,
        maximum_utilization=maximum_utilization,
    )
    if not eligible:
        return {
            "status": "FAIL",
            "errors": ["no_capacity_candidate_within_reserve"],
            "maximum_utilization": maximum_utilization,
            "selected": None,
            "measurements": [_measurement_dict(item) for item in measurements],
        }
    selected = eligible[0]
    return {
        "status": "PASS",
        "errors": [],
        "maximum_utilization": maximum_utilization,
        "selected": _measurement_dict(selected),
        "measurements": [_measurement_dict(item) for item in measurements],
        "selection_rule": (
            "maximum measured throughput within the reserved-memory boundary; deterministic "
            "ties prefer lower utilization and then smaller batches"
        ),
    }


def rank_capacity_candidates(
    measurements: list[CapacityMeasurement],
    *,
    maximum_utilization: float = 0.88,
) -> list[CapacityMeasurement]:
    """Rank viable candidates so numerical canaries can reject one without leaving the ladder."""

    if not 0.5 <= maximum_utilization < 1.0:
        raise ValueError("maximum_utilization must be in [0.5, 1.0)")
    eligible = [
        item
        for item in measurements
        if item.completed
        and item.batch_size > 0
        and item.samples_per_second > 0
        and item.utilization <= maximum_utilization
    ]
    return sorted(
        eligible,
        key=lambda item: (
            item.samples_per_second,
            -item.utilization,
            -item.batch_size,
            item.candidate_id,
        ),
        reverse=True,
    )


def _measurement_dict(item: CapacityMeasurement) -> dict[str, object]:
    return {
        **asdict(item),
        "utilization": item.utilization,
    }


def _system_ram_gib() -> float:
    if os.name == "nt":

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("memory_load", ctypes.c_ulong),
                ("total_phys", ctypes.c_ulonglong),
                ("avail_phys", ctypes.c_ulonglong),
                ("total_page_file", ctypes.c_ulonglong),
                ("avail_page_file", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("avail_virtual", ctypes.c_ulonglong),
                ("avail_extended_virtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(MemoryStatus)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        return status.total_phys / (1024**3)
    pages = os.sysconf("SC_PHYS_PAGES")
    page_size = os.sysconf("SC_PAGE_SIZE")
    return pages * page_size / (1024**3)


def observe_runtime(path_for_disk: Path) -> RuntimeProbe:
    disk_free = shutil.disk_usage(path_for_disk).free / (1024**3)
    unavailable = RuntimeProbe(
        compute_available=False,
        device_type=None,
        device_name=None,
        device_index=None,
        device_memory_gib=0.0,
        bf16_supported=False,
        tensor_probe_passed=False,
        system_ram_gib=_system_ram_gib(),
        disk_free_gib=disk_free,
    )
    try:
        import torch
    except ImportError:
        return unavailable
    interface = getattr(torch, "accelerator", None)
    if interface is None or not interface.is_available():
        return unavailable
    device = interface.current_accelerator(check_available=True)
    if device is None:
        return unavailable
    device_index = int(interface.current_device_index())
    device_name = None
    compute_capability: str | None = None
    try:
        if str(device.type).casefold() == "cuda":
            device_name = str(torch.cuda.get_device_name(device_index))
            properties = torch.cuda.get_device_properties(device_index)
            total_memory = int(properties.total_memory)
            compute_capability = f"{int(properties.major)}.{int(properties.minor)}"
        else:
            _, total_memory = interface.memory.get_memory_info(device_index)
    except Exception:
        total_memory = 0
    try:
        value = torch.ones(8, device=device, dtype=torch.bfloat16)
        tensor_probe_passed = bool(float(value.sum().item()) == 8.0)
        bf16_supported = tensor_probe_passed
    except Exception:
        tensor_probe_passed = False
        bf16_supported = False
    return RuntimeProbe(
        compute_available=True,
        device_type=str(device.type),
        device_name=device_name,
        device_index=device_index,
        device_memory_gib=total_memory / (1024**3),
        bf16_supported=bf16_supported,
        tensor_probe_passed=tensor_probe_passed,
        system_ram_gib=_system_ram_gib(),
        disk_free_gib=disk_free,
        compute_capability=compute_capability,
    )


def check_runtime_profile(
    profile: dict[str, object],
    probe: RuntimeProbe,
) -> dict[str, object]:
    errors: list[str] = []
    if float(probe.system_ram_gib) < float(profile.get("minimum_system_ram_gib", 0)):
        errors.append("system_ram_below_minimum")
    if float(probe.disk_free_gib) < float(profile.get("minimum_disk_free_gib", 0)):
        errors.append("disk_free_below_minimum")
    if bool(profile.get("compute_required", False)):
        if not probe.compute_available:
            errors.append("compute_device_unavailable")
        required_device_type = str(profile.get("required_device_type", "")).strip()
        if required_device_type and required_device_type.casefold() != str(
            probe.device_type or ""
        ).casefold():
            errors.append("device_type_mismatch")
        required_capability = str(profile.get("required_compute_capability", "")).strip()
        if required_capability and required_capability != str(probe.compute_capability or ""):
            errors.append("compute_capability_mismatch")
        required_names = profile.get("required_device_names", [])
        if required_names:
            if not isinstance(required_names, list) or any(
                not isinstance(value, str) or not value.strip() for value in required_names
            ):
                errors.append("required_device_names_invalid")
            else:
                observed_name = _normalize_device_name(probe.device_name)
                allowed_names = {_normalize_device_name(value) for value in required_names}
                if observed_name not in allowed_names:
                    errors.append("device_name_mismatch")
        if probe.device_memory_gib < float(profile.get("minimum_device_memory_gib", 0)):
            errors.append("device_memory_below_minimum")
        maximum_memory = profile.get("maximum_device_memory_gib")
        if maximum_memory is not None and probe.device_memory_gib > float(maximum_memory):
            errors.append("device_memory_above_maximum")
        if bool(profile.get("bf16_required", False)) and not probe.bf16_supported:
            errors.append("bf16_probe_failed")
        if bool(profile.get("tensor_probe_required", False)) and not probe.tensor_probe_passed:
            errors.append("tensor_probe_failed")
    status = "PASS" if not errors else "FAIL"
    # A probe may describe an attempted observation on a host with no usable
    # accelerator.  It becomes observed hardware evidence only after the
    # complete requested profile has passed; otherwise a local host-only check
    # must remain an explicit non-observation.
    hardware_observed = status == "PASS" and probe.evidence_kind == "observed"
    return {
        "status": status,
        "profile": profile.get("name"),
        "errors": errors,
        "probe": probe.as_dict(),
        "hardware_observed": hardware_observed,
    }


def validate_model_device_placement(
    model: Any,
    *,
    model_label: str,
    required_device_type: str = "cuda",
) -> dict[str, object]:
    """Reject Accelerate maps or parameters placed on CPU, disk, meta, or unknown devices."""

    required = required_device_type.strip().casefold()
    if required != "cuda":
        raise ValueError("only the fail-closed CUDA placement contract is supported")
    placements: list[object]
    evidence_kind: str
    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, dict) and device_map:
        placements = list(device_map.values())
        evidence_kind = "hf_device_map"
    else:
        try:
            placements = [parameter.device for parameter in model.parameters()]
        except (AttributeError, TypeError) as exc:
            raise RuntimeError(f"{model_label} exposes no verifiable device placement") from exc
        evidence_kind = "parameters"

    normalized = [_normalize_model_placement(value) for value in placements]
    errors: list[str] = []
    if not normalized:
        errors.append("model_device_placement_empty")
    forbidden = sorted({value for value in normalized if value != required})
    if forbidden:
        errors.append("model_offload_or_non_cuda_placement")
    result = {
        "schema_version": "1.0.0",
        "status": "PASS" if not errors else "FAIL",
        "model_label": model_label,
        "required_device_type": required,
        "evidence_kind": evidence_kind,
        "normalized_placements": sorted(set(normalized)),
        "errors": errors,
    }
    if errors:
        raise RuntimeError(result)
    return result


def _normalize_device_name(value: object) -> str:
    return " ".join(str(value or "").strip().casefold().split())


def _normalize_model_placement(value: object) -> str:
    if isinstance(value, int) and not isinstance(value, bool):
        return "cuda" if value >= 0 else "invalid"
    normalized = str(value).strip().casefold()
    if normalized.isdigit() or normalized == "cuda" or normalized.startswith("cuda:"):
        return "cuda"
    if normalized.startswith("cpu"):
        return "cpu"
    if normalized.startswith("disk"):
        return "disk"
    if normalized.startswith("meta"):
        return "meta"
    return normalized or "unknown"


def check_runtime_profile_path(
    profile_path: Path,
    path_for_disk: Path,
) -> dict[str, object]:
    return check_runtime_profile(load_yaml(profile_path), observe_runtime(path_for_disk))
