from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
import threading
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

import psutil

from vipibench.dataio import canonical_json, sha256_file, write_json
from vipibench.modeling import load_yaml
from vipibench.runtime_capacity import RuntimeProbe, check_runtime_profile
from vipibench.runtime_storage import verify_runtime_storage_plan_document
from vipibench.runtime_telemetry import (
    strict_capacity_receipt_sha256,
    verify_telemetry_ledger,
)

SCHEMA_VERSION = "1.0.0"
SAMPLE_KIND = "public_stage_resource_sample"
SUMMARY_KIND = "public_stage_resource_observation"
MEASUREMENT_KIND = "observed_public_stage_resource_measurement"

EXPECTED_PUBLIC_STAGES = (
    "preflight",
    "data",
    "baselines",
    "encoder",
    "core",
    "attack-generate",
    "attack-evaluate",
    "analysis",
    "finalize",
)

_SAMPLE_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "stage_id",
        "session_id",
        "timestamp_utc",
        "monotonic_seconds",
        "gpu",
        "host",
        "paths",
        "errors",
        "sample_sha256",
    }
)
_SUMMARY_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "status",
        "stage_id",
        "session_id",
        "stage_execution_status",
        "started_at",
        "ended_at",
        "elapsed_seconds",
        "sample_count",
        "valid_gpu_sample_count",
        "raw_samples_path",
        "raw_samples_sha256",
        "storage_plan_path",
        "storage_plan_sha256",
        "session_capacity_check_path",
        "session_capacity_check_sha256",
        "strict_capacity_receipt_sha256",
        "metric_summary",
        "sample_error_counts",
        "errors",
        "claim_boundary",
        "summary_sha256",
    }
)
_MEASUREMENT_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "measurement_kind",
        "hardware_observed",
        "observed_resource_utilization",
        "strict_capacity_receipt_sha256",
        "runtime_telemetry_sha256",
        "target_compute_hours",
        "public_stage_observed_seconds",
        "public_stage_observed_hours",
        "successful_stage_observed_seconds",
        "expected_public_stages",
        "completed_public_stages",
        "missing_public_stages",
        "stage_attempts",
        "summary_file_count",
        "failed_stage_attempt_count",
        "billed_cost",
        "final_holdout_feedback_used",
        "claim_boundary",
        "measurement_sha256",
    }
)
_GPU_FIELDS = (
    "index",
    "name",
    "utilization_percent",
    "memory_used_mib",
    "memory_total_mib",
    "power_draw_w",
    "power_limit_w",
    "temperature_c",
    "sm_clock_mhz",
)


class StageResourceObserver:
    """Sample one public stage in a bounded background thread.

    The observer never controls the workload. It writes append-only raw samples, then produces a
    summary bound to the session storage plan and strict A100 capacity receipt.
    """

    def __init__(
        self,
        *,
        stage_id: str,
        session_id: str,
        output_root: Path,
        project_root: Path,
        storage_plan_path: Path,
        session_capacity_check_path: Path,
        strict_capacity_receipt_path: Path,
        sample_interval_seconds: float = 15.0,
        sample_provider: Callable[[], Mapping[str, object]] | None = None,
    ) -> None:
        if not stage_id.strip() or not session_id.strip():
            raise ValueError("stage_id and session_id must be non-empty")
        if sample_interval_seconds <= 0:
            raise ValueError("sample interval must be positive")
        self.stage_id = stage_id
        self.session_id = session_id
        self.output_root = output_root.resolve()
        self.project_root = project_root.resolve()
        self.storage_plan_path = storage_plan_path.resolve()
        self.session_capacity_check_path = session_capacity_check_path.resolve()
        self.strict_capacity_receipt_path = strict_capacity_receipt_path.resolve()
        self.sample_interval_seconds = float(sample_interval_seconds)
        self.session_root = self.storage_plan_path.parent
        self.raw_samples_path = self.session_root / "resource_observation.samples.jsonl"
        self.summary_path = self.session_root / "resource_observation.summary.json"
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = False
        self._stopped = False
        self._lock = threading.Lock()

        storage_plan = json.loads(self.storage_plan_path.read_text(encoding="utf-8"))
        if not isinstance(storage_plan, Mapping):
            raise ValueError("storage plan must be an object")
        plan = verify_runtime_storage_plan_document(storage_plan)
        monitored_paths = {
            "output": self.output_root,
            "ephemeral": Path(str(plan["ephemeral_root"])),
            "model_cache": Path(str(plan["model_cache_root"])),
        }
        self._sample_provider = sample_provider or (
            lambda: collect_live_resource_payload(monitored_paths=monitored_paths)
        )

    def start(self) -> None:
        if self._started:
            raise RuntimeError("resource observer already started")
        if self.raw_samples_path.exists() or self.summary_path.exists():
            raise FileExistsError("resource observation output already exists for this session")
        self.raw_samples_path.parent.mkdir(parents=True, exist_ok=True)
        self._started = True
        self._append_live_sample()
        self._thread = threading.Thread(
            target=self._run,
            name=f"vipibench-resource-{self.stage_id}",
            daemon=True,
        )
        self._thread.start()

    def stop(self, *, stage_execution_status: str) -> dict[str, object]:
        if not self._started:
            raise RuntimeError("resource observer was not started")
        if self._stopped:
            raw = json.loads(self.summary_path.read_text(encoding="utf-8"))
            if not isinstance(raw, Mapping):
                raise ValueError("resource observation summary invalid")
            return dict(raw)
        if stage_execution_status not in {"completed", "failed"}:
            raise ValueError("stage execution status invalid")
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(5.0, self.sample_interval_seconds + 2.0))
            if self._thread.is_alive():
                raise RuntimeError("resource observer thread did not stop")
        self._append_live_sample()
        summary = build_stage_resource_summary(
            raw_samples_path=self.raw_samples_path,
            storage_plan_path=self.storage_plan_path,
            session_capacity_check_path=self.session_capacity_check_path,
            strict_capacity_receipt_path=self.strict_capacity_receipt_path,
            output_root=self.output_root,
            project_root=self.project_root,
            stage_execution_status=stage_execution_status,
            output_path=self.summary_path,
        )
        self._stopped = True
        return summary

    def _run(self) -> None:
        while not self._stop_event.wait(self.sample_interval_seconds):
            self._append_live_sample()

    def _append_live_sample(self) -> None:
        try:
            payload = self._sample_provider()
            if not isinstance(payload, Mapping):
                raise TypeError("sample provider must return a mapping")
            gpu = payload.get("gpu")
            host = payload.get("host")
            paths = payload.get("paths", {})
            errors = payload.get("errors", [])
            if not isinstance(paths, Mapping) or not isinstance(errors, list):
                raise TypeError("sample provider paths/errors invalid")
            self._append_sample(gpu=gpu, host=host, paths=dict(paths), errors=list(errors))
        except BaseException as exc:
            self._append_failure_sample(f"collector_exception:{type(exc).__name__}:{exc}")

    def _append_failure_sample(self, message: str) -> None:
        self._append_sample(gpu=None, host=None, paths={}, errors=[message])

    def _append_sample(
        self,
        *,
        gpu: object,
        host: object,
        paths: Mapping[str, object],
        errors: Sequence[object],
    ) -> None:
        sample: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "kind": SAMPLE_KIND,
            "stage_id": self.stage_id,
            "session_id": self.session_id,
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "monotonic_seconds": time.monotonic(),
            "gpu": gpu,
            "host": host,
            "paths": dict(paths),
            "errors": [str(error) for error in errors],
        }
        sample["sample_sha256"] = _payload_sha256(sample)
        validated = _validate_sample(sample)
        line = canonical_json(validated) + "\n"
        with self._lock, self.raw_samples_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())


def collect_live_resource_payload(
    *, monitored_paths: Mapping[str, Path]
) -> dict[str, object]:
    errors: list[str] = []
    gpu: dict[str, object] | None = None
    host: dict[str, object] | None = None
    paths: dict[str, object] = {}
    try:
        gpu = _query_nvidia_smi()
    except BaseException as exc:
        errors.append(f"nvidia_smi:{type(exc).__name__}:{exc}")
    try:
        host = _query_host_metrics()
    except BaseException as exc:
        errors.append(f"host_metrics:{type(exc).__name__}:{exc}")
    for name, path in sorted(monitored_paths.items()):
        try:
            usage = shutil.disk_usage(path)
            paths[name] = {
                "path": str(path.resolve()),
                "device_id": int(path.stat().st_dev),
                "total_gib": usage.total / (1024**3),
                "used_gib": usage.used / (1024**3),
                "free_gib": usage.free / (1024**3),
            }
        except BaseException as exc:
            errors.append(f"disk_usage:{name}:{type(exc).__name__}:{exc}")
    return {"gpu": gpu, "host": host, "paths": paths, "errors": errors}


def build_stage_resource_summary(
    *,
    raw_samples_path: Path,
    storage_plan_path: Path,
    session_capacity_check_path: Path,
    strict_capacity_receipt_path: Path,
    output_root: Path,
    project_root: Path,
    stage_execution_status: str,
    output_path: Path | None = None,
) -> dict[str, object]:
    samples = _load_samples(raw_samples_path)
    if stage_execution_status not in {"completed", "failed"}:
        raise ValueError("stage execution status invalid")
    storage_raw = json.loads(storage_plan_path.read_text(encoding="utf-8"))
    if not isinstance(storage_raw, Mapping):
        raise ValueError("storage plan must be an object")
    storage_plan = verify_runtime_storage_plan_document(storage_raw)
    session_capacity_check = _verify_session_capacity_check(
        session_capacity_check_path,
        project_root,
    )
    receipt_raw = json.loads(strict_capacity_receipt_path.read_text(encoding="utf-8"))
    if not isinstance(receipt_raw, Mapping):
        raise ValueError("strict capacity receipt must be an object")
    receipt_hash = strict_capacity_receipt_sha256(receipt_raw, project_root=project_root)

    stage_ids = {str(sample["stage_id"]) for sample in samples}
    session_ids = {str(sample["session_id"]) for sample in samples}
    if len(stage_ids) != 1 or len(session_ids) != 1:
        raise ValueError("resource samples mix stage or session identities")
    valid_gpu_samples = [sample for sample in samples if _valid_gpu(sample.get("gpu"))]
    fatal_errors: list[str] = []
    if not samples:
        fatal_errors.append("resource_samples_missing")
    if not valid_gpu_samples:
        fatal_errors.append("valid_gpu_samples_missing")
    observed_device_name = str(session_capacity_check["probe"]["device_name"])
    if any(str(sample["gpu"]["name"]) != observed_device_name for sample in valid_gpu_samples):
        fatal_errors.append("gpu_identity_changed_during_stage")

    sample_errors = Counter(
        str(error)
        for sample in samples
        for error in sample.get("errors", [])
        if isinstance(error, str)
    )
    started = str(samples[0]["timestamp_utc"])
    ended = str(samples[-1]["timestamp_utc"])
    elapsed = max(
        0.0,
        float(samples[-1]["monotonic_seconds"]) - float(samples[0]["monotonic_seconds"]),
    )
    summary: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "kind": SUMMARY_KIND,
        "status": "PASS" if not fatal_errors else "FAIL",
        "stage_id": next(iter(stage_ids)),
        "session_id": next(iter(session_ids)),
        "stage_execution_status": stage_execution_status,
        "started_at": started,
        "ended_at": ended,
        "elapsed_seconds": elapsed,
        "sample_count": len(samples),
        "valid_gpu_sample_count": len(valid_gpu_samples),
        "raw_samples_path": _relative_output_path(raw_samples_path, output_root),
        "raw_samples_sha256": sha256_file(raw_samples_path),
        "storage_plan_path": _relative_output_path(storage_plan_path, output_root),
        "storage_plan_sha256": str(storage_plan["plan_sha256"]),
        "session_capacity_check_path": _relative_output_path(
            session_capacity_check_path, output_root
        ),
        "session_capacity_check_sha256": sha256_file(session_capacity_check_path),
        "strict_capacity_receipt_sha256": receipt_hash,
        "metric_summary": _metric_summary(samples),
        "sample_error_counts": dict(sorted(sample_errors.items())),
        "errors": fatal_errors,
        "claim_boundary": (
            "PASS proves hash-bound stage resource observations under one strict A100 80 GB "
            "receipt. Utilization values describe this stage attempt; they do not prove maximum "
            "possible throughput, scientific quality, or billed cost."
        ),
    }
    summary["summary_sha256"] = _payload_sha256(summary)
    validated = _validate_summary(summary)
    if output_path is not None:
        write_json(output_path, validated)
    return validated


def verify_stage_resource_summary(
    summary: Mapping[str, object],
    *,
    summary_path: Path,
    output_root: Path,
    project_root: Path,
    strict_capacity_receipt_path: Path,
) -> dict[str, object]:
    candidate = _validate_summary(summary)
    raw_path = _resolve_output_path(output_root, str(candidate["raw_samples_path"]))
    storage_path = _resolve_output_path(output_root, str(candidate["storage_plan_path"]))
    session_capacity_check_path = _resolve_output_path(
        output_root, str(candidate["session_capacity_check_path"])
    )
    rebuilt = build_stage_resource_summary(
        raw_samples_path=raw_path,
        storage_plan_path=storage_path,
        session_capacity_check_path=session_capacity_check_path,
        strict_capacity_receipt_path=strict_capacity_receipt_path,
        output_root=output_root,
        project_root=project_root,
        stage_execution_status=str(candidate["stage_execution_status"]),
    )
    if canonical_json(rebuilt) != canonical_json(candidate):
        raise ValueError(f"resource observation summary contents mismatch: {summary_path}")
    return candidate


def build_resource_measurement(
    *,
    output_root: Path,
    project_root: Path,
    expected_public_stages: Sequence[str] = EXPECTED_PUBLIC_STAGES,
    output_path: Path | None = None,
) -> dict[str, object]:
    measurement = _build_resource_measurement(
        output_root=output_root,
        project_root=project_root,
        expected_public_stages=expected_public_stages,
    )
    if output_path is not None:
        write_json(output_path, measurement)
    return measurement


def verify_resource_measurement(
    measurement: Mapping[str, object],
    *,
    output_root: Path,
    project_root: Path,
    expected_public_stages: Sequence[str] = EXPECTED_PUBLIC_STAGES,
) -> dict[str, object]:
    candidate = _validate_measurement(measurement)
    rebuilt = _build_resource_measurement(
        output_root=output_root,
        project_root=project_root,
        expected_public_stages=expected_public_stages,
    )
    if canonical_json(rebuilt) != canonical_json(candidate):
        raise ValueError("resource measurement contents mismatch")
    return candidate


def _build_resource_measurement(
    *,
    output_root: Path,
    project_root: Path,
    expected_public_stages: Sequence[str],
) -> dict[str, object]:
    root = output_root.resolve()
    receipt_path = root / "strict_capacity_receipt.json"
    receipt_raw = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not isinstance(receipt_raw, Mapping):
        raise ValueError("strict capacity receipt must be an object")
    receipt_hash = strict_capacity_receipt_sha256(receipt_raw, project_root=project_root)

    summary_paths = sorted(
        root.glob("session_evidence/runtime_sessions/*/resource_observation.summary.json")
    )
    attempts: list[dict[str, object]] = []
    for path in summary_paths:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            raise ValueError(f"resource observation summary must be an object: {path}")
        summary = verify_stage_resource_summary(
            raw,
            summary_path=path,
            output_root=root,
            project_root=project_root,
            strict_capacity_receipt_path=receipt_path,
        )
        if summary["status"] != "PASS":
            raise ValueError(f"resource observation summary is not PASS: {path}")
        if summary["strict_capacity_receipt_sha256"] != receipt_hash:
            raise ValueError(f"resource observation receipt mismatch: {path}")
        attempts.append(
            {
                "stage_id": summary["stage_id"],
                "session_id": summary["session_id"],
                "stage_execution_status": summary["stage_execution_status"],
                "started_at": summary["started_at"],
                "ended_at": summary["ended_at"],
                "elapsed_seconds": summary["elapsed_seconds"],
                "summary_path": _relative_output_path(path, root),
                "summary_sha256": summary["summary_sha256"],
                "storage_plan_sha256": summary["storage_plan_sha256"],
                "session_capacity_check_sha256": summary[
                    "session_capacity_check_sha256"
                ],
                "sample_count": summary["sample_count"],
                "valid_gpu_sample_count": summary["valid_gpu_sample_count"],
                "metric_summary": summary["metric_summary"],
            }
        )
    attempts.sort(key=lambda item: (str(item["started_at"]), str(item["session_id"])))
    expected = [str(stage) for stage in expected_public_stages]
    completed = [
        stage
        for stage in expected
        if any(
            item["stage_id"] == stage and item["stage_execution_status"] == "completed"
            for item in attempts
        )
    ]
    missing = [stage for stage in expected if stage not in completed]
    observed_seconds = sum(float(item["elapsed_seconds"]) for item in attempts)
    successful_seconds = sum(
        float(item["elapsed_seconds"])
        for item in attempts
        if item["stage_execution_status"] == "completed"
    )

    runtime_telemetry_path = root / "runtime_telemetry.json"
    runtime_telemetry_sha256: str | None = None
    target_compute_hours: float | None = None
    if runtime_telemetry_path.is_file():
        telemetry_raw = json.loads(runtime_telemetry_path.read_text(encoding="utf-8"))
        if not isinstance(telemetry_raw, Mapping):
            raise ValueError("runtime telemetry must be an object")
        telemetry = verify_telemetry_ledger(
            telemetry_raw,
            strict_capacity_receipt=receipt_raw,
            project_root=project_root,
        )
        runtime_telemetry_sha256 = sha256_file(runtime_telemetry_path)
        target_compute_hours = float(telemetry["compute_hours"])

    measurement: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS" if not missing else "PARTIAL",
        "measurement_kind": MEASUREMENT_KIND,
        "hardware_observed": True,
        "observed_resource_utilization": bool(attempts),
        "strict_capacity_receipt_sha256": receipt_hash,
        "runtime_telemetry_sha256": runtime_telemetry_sha256,
        "target_compute_hours": target_compute_hours,
        "public_stage_observed_seconds": observed_seconds,
        "public_stage_observed_hours": observed_seconds / 3600.0,
        "successful_stage_observed_seconds": successful_seconds,
        "expected_public_stages": expected,
        "completed_public_stages": completed,
        "missing_public_stages": missing,
        "stage_attempts": attempts,
        "summary_file_count": len(summary_paths),
        "failed_stage_attempt_count": sum(
            item["stage_execution_status"] == "failed" for item in attempts
        ),
        "billed_cost": None,
        "final_holdout_feedback_used": False,
        "claim_boundary": (
            "PASS requires at least one completed, hash-bound resource observation for every "
            "locked public stage under the current strict A100 80 GB receipt. It does not claim "
            "that utilization is globally maximal or that resource use improves model quality."
        ),
    }
    measurement["measurement_sha256"] = _payload_sha256(measurement)
    return _validate_measurement(measurement)


def _verify_session_capacity_check(path: Path, project_root: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"session capacity check missing or unsafe: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("session capacity check must be an object")
    if set(raw) != {"status", "profile", "errors", "probe", "hardware_observed"}:
        raise ValueError("session capacity check fields invalid")
    probe_raw = raw.get("probe")
    if not isinstance(probe_raw, Mapping):
        raise ValueError("session capacity check probe invalid")
    expected_probe_fields = set(RuntimeProbe.__dataclass_fields__)
    if set(probe_raw) != expected_probe_fields:
        raise ValueError("session capacity check probe fields invalid")
    probe = RuntimeProbe(**{field: probe_raw[field] for field in expected_probe_fields})
    profile = load_yaml(project_root / "configs" / "profiles" / "accelerator_80gb.yaml")
    if not isinstance(profile, dict):
        raise ValueError("strict accelerator profile must be an object")
    rebuilt = check_runtime_profile(profile, probe)
    if canonical_json(rebuilt) != canonical_json(raw):
        raise ValueError("session capacity check does not match the strict A100 profile")
    if rebuilt["status"] != "PASS" or rebuilt["hardware_observed"] is not True:
        raise ValueError("session capacity check is not an observed PASS")
    return dict(rebuilt)


def _query_nvidia_smi() -> dict[str, object]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        raise FileNotFoundError("nvidia-smi")
    fields = (
        "index,name,utilization.gpu,memory.used,memory.total,power.draw,power.limit,"
        "temperature.gpu,clocks.current.sm"
    )
    result = subprocess.run(
        [executable, f"--query-gpu={fields}", "--format=csv,noheader,nounits", "--id=0"],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"exit_{result.returncode}")
    line = next((item.strip() for item in result.stdout.splitlines() if item.strip()), "")
    values = [item.strip() for item in line.split(",")]
    if len(values) != len(_GPU_FIELDS):
        raise ValueError("unexpected nvidia-smi field count")
    parsed: dict[str, object] = {
        "index": int(values[0]),
        "name": values[1],
    }
    for field, value in zip(_GPU_FIELDS[2:], values[2:], strict=True):
        parsed[field] = _optional_float(value)
    if not _valid_gpu(parsed):
        raise ValueError("required nvidia-smi values are unavailable")
    return parsed


def _query_host_metrics() -> dict[str, object]:
    memory = psutil.virtual_memory()
    disk = psutil.disk_io_counters()
    try:
        load_1m = float(os.getloadavg()[0])
    except (AttributeError, OSError):
        load_1m = None
    return {
        "cpu_percent": float(psutil.cpu_percent(interval=None)),
        "logical_cpu_count": psutil.cpu_count(logical=True),
        "load_1m": load_1m,
        "ram_used_gib": memory.used / (1024**3),
        "ram_available_gib": memory.available / (1024**3),
        "ram_percent": float(memory.percent),
        "disk_read_bytes": int(disk.read_bytes) if disk is not None else None,
        "disk_write_bytes": int(disk.write_bytes) if disk is not None else None,
        "disk_read_time_ms": int(disk.read_time) if disk is not None else None,
        "disk_write_time_ms": int(disk.write_time) if disk is not None else None,
    }


def _load_samples(path: Path) -> list[dict[str, object]]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(path)
    samples: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"resource sample JSONL invalid at line {line_number}") from exc
        if not isinstance(raw, Mapping):
            raise ValueError(f"resource sample must be an object at line {line_number}")
        samples.append(_validate_sample(raw))
    if not samples:
        raise ValueError("resource sample file is empty")
    monotonic = [float(sample["monotonic_seconds"]) for sample in samples]
    if monotonic != sorted(monotonic):
        raise ValueError("resource samples are not monotonic")
    return samples


def _validate_sample(sample: Mapping[str, object]) -> dict[str, object]:
    candidate = dict(sample)
    _require_exact_fields(candidate, _SAMPLE_FIELDS, "resource sample")
    if candidate["schema_version"] != SCHEMA_VERSION or candidate["kind"] != SAMPLE_KIND:
        raise ValueError("resource sample schema mismatch")
    _require_nonempty_string(candidate["stage_id"], "stage_id")
    _require_nonempty_string(candidate["session_id"], "session_id")
    _require_nonempty_string(candidate["timestamp_utc"], "timestamp_utc")
    _finite_number(candidate["monotonic_seconds"], "monotonic_seconds")
    if candidate["gpu"] is not None and not isinstance(candidate["gpu"], Mapping):
        raise ValueError("resource sample gpu invalid")
    if candidate["host"] is not None and not isinstance(candidate["host"], Mapping):
        raise ValueError("resource sample host invalid")
    if not isinstance(candidate["paths"], Mapping):
        raise ValueError("resource sample paths invalid")
    if not isinstance(candidate["errors"], list) or any(
        not isinstance(item, str) for item in candidate["errors"]
    ):
        raise ValueError("resource sample errors invalid")
    observed_hash = _require_sha256(candidate["sample_sha256"], "sample_sha256")
    if _payload_sha256(_without(candidate, "sample_sha256")) != observed_hash:
        raise ValueError("resource sample hash mismatch")
    return candidate


def _validate_summary(summary: Mapping[str, object]) -> dict[str, object]:
    candidate = dict(summary)
    _require_exact_fields(candidate, _SUMMARY_FIELDS, "resource summary")
    if candidate["schema_version"] != SCHEMA_VERSION or candidate["kind"] != SUMMARY_KIND:
        raise ValueError("resource summary schema mismatch")
    if candidate["status"] not in {"PASS", "FAIL"}:
        raise ValueError("resource summary status invalid")
    if candidate["stage_execution_status"] not in {"completed", "failed"}:
        raise ValueError("resource summary execution status invalid")
    for field in (
        "stage_id",
        "session_id",
        "started_at",
        "ended_at",
        "raw_samples_path",
        "storage_plan_path",
        "session_capacity_check_path",
        "claim_boundary",
    ):
        _require_nonempty_string(candidate[field], field)
    _finite_number(candidate["elapsed_seconds"], "elapsed_seconds")
    if not isinstance(candidate["sample_count"], int) or candidate["sample_count"] <= 0:
        raise ValueError("resource summary sample count invalid")
    if (
        not isinstance(candidate["valid_gpu_sample_count"], int)
        or candidate["valid_gpu_sample_count"] < 0
    ):
        raise ValueError("resource summary valid GPU sample count invalid")
    for field in (
        "raw_samples_sha256",
        "storage_plan_sha256",
        "session_capacity_check_sha256",
        "strict_capacity_receipt_sha256",
    ):
        _require_sha256(candidate[field], field)
    for field in ("metric_summary", "sample_error_counts"):
        if not isinstance(candidate[field], Mapping):
            raise ValueError(f"resource summary {field} invalid")
    if not isinstance(candidate["errors"], list):
        raise ValueError("resource summary errors invalid")
    observed_hash = _require_sha256(candidate["summary_sha256"], "summary_sha256")
    if _payload_sha256(_without(candidate, "summary_sha256")) != observed_hash:
        raise ValueError("resource summary hash mismatch")
    return candidate


def _validate_measurement(measurement: Mapping[str, object]) -> dict[str, object]:
    candidate = dict(measurement)
    _require_exact_fields(candidate, _MEASUREMENT_FIELDS, "resource measurement")
    if candidate["schema_version"] != SCHEMA_VERSION:
        raise ValueError("resource measurement schema mismatch")
    if candidate["measurement_kind"] != MEASUREMENT_KIND:
        raise ValueError("resource measurement kind mismatch")
    if candidate["status"] not in {"PASS", "PARTIAL"}:
        raise ValueError("resource measurement status invalid")
    if candidate["hardware_observed"] is not True:
        raise ValueError("resource measurement hardware must be observed")
    _require_sha256(
        candidate["strict_capacity_receipt_sha256"],
        "strict_capacity_receipt_sha256",
    )
    runtime_hash = candidate["runtime_telemetry_sha256"]
    if runtime_hash is not None:
        _require_sha256(runtime_hash, "runtime_telemetry_sha256")
    for field in (
        "expected_public_stages",
        "completed_public_stages",
        "missing_public_stages",
        "stage_attempts",
    ):
        if not isinstance(candidate[field], list):
            raise ValueError(f"resource measurement {field} invalid")
    observed_hash = _require_sha256(candidate["measurement_sha256"], "measurement_sha256")
    if _payload_sha256(_without(candidate, "measurement_sha256")) != observed_hash:
        raise ValueError("resource measurement hash mismatch")
    return candidate


def _metric_summary(samples: Sequence[Mapping[str, object]]) -> dict[str, object]:
    return {
        "gpu_utilization_percent": _summarize_values(
            _nested_numbers(samples, "gpu", "utilization_percent")
        ),
        "gpu_memory_used_mib": _summarize_values(
            _nested_numbers(samples, "gpu", "memory_used_mib")
        ),
        "gpu_power_draw_w": _summarize_values(
            _nested_numbers(samples, "gpu", "power_draw_w")
        ),
        "cpu_percent": _summarize_values(_nested_numbers(samples, "host", "cpu_percent")),
        "ram_used_gib": _summarize_values(_nested_numbers(samples, "host", "ram_used_gib")),
        "disk_read_bytes_delta": _counter_delta(samples, "disk_read_bytes"),
        "disk_write_bytes_delta": _counter_delta(samples, "disk_write_bytes"),
        "minimum_free_gib_by_path": _minimum_free_by_path(samples),
    }


def _nested_numbers(
    samples: Sequence[Mapping[str, object]], parent: str, field: str
) -> list[float]:
    values: list[float] = []
    for sample in samples:
        container = sample.get(parent)
        if not isinstance(container, Mapping):
            continue
        value = container.get(field)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            number = float(value)
            if math.isfinite(number):
                values.append(number)
    return values


def _summarize_values(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "minimum": None, "median": None, "p95": None, "maximum": None}
    ordered = sorted(float(value) for value in values)
    return {
        "count": len(ordered),
        "minimum": ordered[0],
        "median": float(statistics.median(ordered)),
        "p95": ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)],
        "maximum": ordered[-1],
    }


def _counter_delta(samples: Sequence[Mapping[str, object]], field: str) -> int | None:
    values = _nested_numbers(samples, "host", field)
    if len(values) < 2:
        return None
    return max(0, int(values[-1] - values[0]))


def _minimum_free_by_path(samples: Sequence[Mapping[str, object]]) -> dict[str, float]:
    observed: dict[str, list[float]] = {}
    for sample in samples:
        paths = sample.get("paths")
        if not isinstance(paths, Mapping):
            continue
        for name, payload in paths.items():
            if not isinstance(name, str) or not isinstance(payload, Mapping):
                continue
            value = payload.get("free_gib")
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                observed.setdefault(name, []).append(float(value))
    return {name: min(values) for name, values in sorted(observed.items()) if values}


def _valid_gpu(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    required = ("index", "name", "utilization_percent", "memory_used_mib", "memory_total_mib")
    if any(field not in value for field in required):
        return False
    if not isinstance(value.get("name"), str) or not str(value["name"]).strip():
        return False
    for field in required[2:]:
        item = value.get(field)
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            return False
        if not math.isfinite(float(item)):
            return False
    return True


def _optional_float(value: str) -> float | None:
    if not value or value.upper() in {"N/A", "NA", "[NOT SUPPORTED]"}:
        return None
    return float(value)


def _relative_output_path(path: Path, output_root: Path) -> str:
    resolved = path.resolve()
    root = output_root.resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(f"resource artifact escapes output root: {resolved}") from exc


def _resolve_output_path(output_root: Path, relative: str) -> Path:
    candidate = (output_root.resolve() / relative).resolve()
    if output_root.resolve() not in candidate.parents:
        raise ValueError("resource artifact path escapes output root")
    return candidate


def _payload_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest().upper()


def _without(value: Mapping[str, object], key: str) -> dict[str, object]:
    return {name: item for name, item in value.items() if name != key}


def _require_exact_fields(
    value: Mapping[str, object], expected: frozenset[str], label: str
) -> None:
    unknown = sorted(set(value).difference(expected))
    missing = sorted(expected.difference(value))
    if unknown:
        raise ValueError(f"{label} unknown fields: {','.join(unknown)}")
    if missing:
        raise ValueError(f"{label} missing fields: {','.join(missing)}")


def _require_nonempty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} invalid")
    return value


def _require_sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.upper()
        or any(character not in "0123456789ABCDEF" for character in value)
    ):
        raise ValueError(f"{label} invalid")
    return value


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} invalid")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} invalid")
    return number
