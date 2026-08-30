from __future__ import annotations

import hashlib
import os
import shutil
from collections.abc import Iterable, Mapping
from pathlib import Path

import psutil

from vipibench.cache_contract import CACHE_MARKER_NAME, CACHE_MARKER_TEXT
from vipibench.dataio import canonical_json, write_json

SCHEMA_VERSION = "1.0.0"
STORAGE_PLAN_KIND = "verified_ephemeral_storage_plan"
OWNERSHIP_MARKER = ".vipibench-owned-ephemeral"

_REMOTE_OR_VIRTUAL_FILESYSTEMS = frozenset(
    {
        "9p",
        "cifs",
        "fuse",
        "fuse.drivefs",
        "fuseblk",
        "nfs",
        "nfs4",
        "smbfs",
        "sshfs",
    }
)
_KNOWN_SCRATCH_ROOTS = (
    Path("/local-scratch"),
    Path("/scratch"),
    Path("/mnt/local-scratch"),
)
_PLAN_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "status",
        "session_id",
        "selection_mode",
        "minimum_free_gib",
        "default_root",
        "selected_mount_root",
        "selected_device",
        "selected_filesystem",
        "selected_device_id",
        "same_device_as_output",
        "free_gib_at_selection",
        "ephemeral_root",
        "model_cache_root",
        "session_temp_root",
        "torch_cache_root",
        "xdg_cache_root",
        "ownership_marker",
        "considered_candidates",
        "claim_boundary",
        "plan_sha256",
    }
)


def create_runtime_storage_plan(
    *,
    session_id: str,
    default_root: Path,
    output_root: Path,
    protected_roots: Iterable[Path],
    minimum_free_gib: float,
    explicit_root: Path | None = None,
    output_path: Path | None = None,
    partitions: Iterable[Mapping[str, object]] | None = None,
) -> dict[str, object]:
    """Select one owned cache/temp root without moving durable experiment outputs.

    An explicit root is fail-closed. Automatic selection considers only the configured default,
    known scratch paths, and local mounted filesystems whose path identifies them as scratch.
    The selected mount itself is never treated as owned; all writes stay under a marker-bound child.
    """

    if not session_id.strip():
        raise ValueError("session_id must be non-empty")
    if minimum_free_gib <= 0:
        raise ValueError("minimum_free_gib must be positive")

    default = _existing_directory(default_root, "default_root")
    output = output_root.resolve()
    protected = [path.resolve() for path in protected_roots]
    partition_rows = _partition_rows(partitions)
    roots = _candidate_roots(default, explicit_root, partition_rows)
    considered = [
        _evaluate_candidate(
            root,
            default_root=default,
            output_root=output,
            protected_roots=protected,
            minimum_free_gib=minimum_free_gib,
            explicit=explicit_root is not None and root == explicit_root.resolve(),
            partitions=partition_rows,
        )
        for root in roots
    ]

    if explicit_root is not None:
        explicit_candidates = [item for item in considered if item["explicit"] is True]
        if len(explicit_candidates) != 1 or explicit_candidates[0]["eligible"] is not True:
            errors = explicit_candidates[0]["rejection_reasons"] if explicit_candidates else []
            raise ValueError(f"explicit scratch root rejected: {errors}")
        selected = explicit_candidates[0]
        selection_mode = "explicit_verified_root"
    else:
        eligible = [item for item in considered if item["eligible"] is True]
        if not eligible:
            raise RuntimeError("no safe ephemeral storage candidate meets the free-space gate")
        selected = sorted(eligible, key=_candidate_rank)[0]
        selection_mode = str(selected["selection_class"])

    selected_root = Path(str(selected["path"]))
    ephemeral_candidate = selected_root / "vipibench-ephemeral"
    if ephemeral_candidate.is_symlink():
        raise ValueError("ephemeral root must not be a symlink")
    ephemeral_root = ephemeral_candidate.resolve()
    _reject_overlap(ephemeral_root, [output, *protected])
    marker = _claim_owned_root(ephemeral_root)
    model_cache_root = _claim_model_cache_root(
        ephemeral_root / "model-cache",
        owner_marker=marker,
    )
    session_temp_root = ephemeral_root / "tmp" / session_id
    torch_cache_root = ephemeral_root / "torch-cache"
    xdg_cache_root = ephemeral_root / "xdg-cache"
    huggingface_hub_root = model_cache_root / "hub"
    for path in (
        model_cache_root,
        huggingface_hub_root,
        session_temp_root,
        torch_cache_root,
        xdg_cache_root,
    ):
        path.mkdir(parents=True, exist_ok=True)

    plan: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "kind": STORAGE_PLAN_KIND,
        "status": "PASS",
        "session_id": session_id,
        "selection_mode": selection_mode,
        "minimum_free_gib": float(minimum_free_gib),
        "default_root": str(default),
        "selected_mount_root": str(selected_root),
        "selected_device": selected["device"],
        "selected_filesystem": selected["filesystem"],
        "selected_device_id": int(selected["device_id"]),
        "same_device_as_output": int(selected["device_id"]) == _device_id(output),
        "free_gib_at_selection": float(selected["free_gib"]),
        "ephemeral_root": str(ephemeral_root),
        "model_cache_root": str(model_cache_root),
        "session_temp_root": str(session_temp_root),
        "torch_cache_root": str(torch_cache_root),
        "xdg_cache_root": str(xdg_cache_root),
        "ownership_marker": str(marker),
        "considered_candidates": considered,
        "claim_boundary": (
            "PASS proves only that cache and temporary paths use one writable, marker-owned child "
            "with sufficient observed free space. Durable outputs, checkpoints, and research "
            "evidence remain on the existing checksum-bound output path."
        ),
    }
    plan["plan_sha256"] = _payload_sha256(plan)
    validated = verify_runtime_storage_plan(plan, output_root=output, protected_roots=protected)
    if output_path is not None:
        write_json(output_path, validated)
    return validated


def verify_runtime_storage_plan(
    plan: Mapping[str, object],
    *,
    output_root: Path,
    protected_roots: Iterable[Path],
) -> dict[str, object]:
    candidate = verify_runtime_storage_plan_document(plan)

    ephemeral_root = _existing_directory(Path(str(candidate["ephemeral_root"])), "ephemeral_root")
    marker = Path(str(candidate["ownership_marker"])).resolve()
    if marker != ephemeral_root / OWNERSHIP_MARKER or not marker.is_file() or marker.is_symlink():
        raise ValueError("ephemeral ownership marker missing or unsafe")
    if marker.read_text(encoding="utf-8").strip() != STORAGE_PLAN_KIND:
        raise ValueError("ephemeral ownership marker content mismatch")
    _reject_overlap(
        ephemeral_root,
        [output_root.resolve(), *(path.resolve() for path in protected_roots)],
    )
    model_cache_root: Path | None = None
    for field in (
        "model_cache_root",
        "session_temp_root",
        "torch_cache_root",
        "xdg_cache_root",
    ):
        path = _existing_directory(Path(str(candidate[field])), field)
        if path != ephemeral_root and ephemeral_root not in path.parents:
            raise ValueError(f"{field} escapes the owned ephemeral root")
        if field == "model_cache_root":
            model_cache_root = path
    if model_cache_root is None:
        raise ValueError("model cache root missing from storage plan")
    model_cache_marker = model_cache_root / CACHE_MARKER_NAME
    if (
        model_cache_marker.is_symlink()
        or not model_cache_marker.is_file()
        or model_cache_marker.read_text(encoding="utf-8") != CACHE_MARKER_TEXT
    ):
        raise ValueError("model-cache ownership marker missing or unsafe")
    if int(candidate["selected_device_id"]) != _device_id(ephemeral_root):
        raise ValueError("selected device id changed")
    return candidate


def verify_runtime_storage_plan_document(plan: Mapping[str, object]) -> dict[str, object]:
    """Validate the durable plan document without requiring the ephemeral mount to survive."""

    candidate = dict(plan)
    _require_exact_fields(candidate, _PLAN_FIELDS, "storage plan")
    if candidate["schema_version"] != SCHEMA_VERSION:
        raise ValueError("storage plan schema version mismatch")
    if candidate["kind"] != STORAGE_PLAN_KIND or candidate["status"] != "PASS":
        raise ValueError("storage plan is not a PASS artifact")
    observed_hash = candidate["plan_sha256"]
    if not _is_sha256(observed_hash):
        raise ValueError("storage plan hash invalid")
    if _payload_sha256(_without(candidate, "plan_sha256")) != observed_hash:
        raise ValueError("storage plan hash mismatch")
    return candidate


def runtime_storage_environment(plan: Mapping[str, object]) -> dict[str, str]:
    """Return child-process cache/temp bindings without changing the caller environment."""

    return {
        "VIPIBENCH_EPHEMERAL_MODEL_CACHE_ROOT": str(plan["model_cache_root"]),
        "HF_HOME": str(plan["model_cache_root"]),
        "HF_HUB_CACHE": str(Path(str(plan["model_cache_root"])) / "hub"),
        "TMPDIR": str(plan["session_temp_root"]),
        "TMP": str(plan["session_temp_root"]),
        "TEMP": str(plan["session_temp_root"]),
        "TORCH_HOME": str(plan["torch_cache_root"]),
        "XDG_CACHE_HOME": str(plan["xdg_cache_root"]),
    }


def _candidate_roots(
    default_root: Path,
    explicit_root: Path | None,
    partitions: list[dict[str, str]],
) -> list[Path]:
    roots: list[Path] = []
    if explicit_root is not None:
        roots.append(explicit_root.resolve())
    else:
        for row in partitions:
            mountpoint = Path(row["mountpoint"])
            if _looks_like_scratch(mountpoint):
                roots.append(mountpoint)
        roots.extend(path for path in _KNOWN_SCRATCH_ROOTS if path.is_dir())
        disk_parent = Path("/mnt/disks")
        if disk_parent.is_dir() and not disk_parent.is_symlink():
            roots.extend(path for path in disk_parent.iterdir() if path.is_dir())
        roots.append(default_root)
    deduplicated: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        try:
            resolved = root.resolve(strict=True)
        except OSError:
            resolved = root.resolve()
        key = os.path.normcase(str(resolved))
        if key not in seen:
            seen.add(key)
            deduplicated.append(resolved)
    return deduplicated


def _evaluate_candidate(
    path: Path,
    *,
    default_root: Path,
    output_root: Path,
    protected_roots: list[Path],
    minimum_free_gib: float,
    explicit: bool,
    partitions: list[dict[str, str]],
) -> dict[str, object]:
    reasons: list[str] = []
    if not path.is_dir():
        reasons.append("missing_directory")
    if path.is_symlink():
        reasons.append("symlink_root")
    if path == Path(path.anchor):
        reasons.append("filesystem_root_not_allowed")
    if not os.access(path, os.R_OK | os.W_OK | os.X_OK):
        reasons.append("not_writable")
    device, filesystem = _partition_identity(path, partitions)
    if filesystem.lower() in _REMOTE_OR_VIRTUAL_FILESYSTEMS:
        reasons.append(f"remote_or_virtual_filesystem:{filesystem}")

    free_gib = 0.0
    device_id = -1
    if path.is_dir():
        try:
            free_gib = shutil.disk_usage(path).free / (1024**3)
            device_id = _device_id(path)
        except OSError as exc:
            reasons.append(f"storage_probe_failed:{type(exc).__name__}")
    if free_gib < minimum_free_gib:
        reasons.append("free_space_below_minimum")

    owned_child = (path / "vipibench-ephemeral").resolve()
    try:
        _reject_overlap(owned_child, [output_root, *protected_roots])
    except ValueError:
        reasons.append("owned_child_overlaps_protected_path")

    if explicit:
        selection_class = "explicit_verified_root"
    elif _looks_like_scratch(path):
        selection_class = "verified_named_local_scratch"
    elif device_id >= 0 and device_id != _device_id(default_root):
        selection_class = "verified_separate_local_mount"
    else:
        selection_class = "verified_default_local_volume"
    return {
        "path": str(path),
        "device": device,
        "filesystem": filesystem,
        "device_id": device_id,
        "free_gib": free_gib,
        "explicit": explicit,
        "selection_class": selection_class,
        "eligible": not reasons,
        "rejection_reasons": reasons,
    }


def _candidate_rank(candidate: Mapping[str, object]) -> tuple[int, float, str]:
    classes = {
        "explicit_verified_root": 0,
        "verified_named_local_scratch": 1,
        "verified_separate_local_mount": 2,
        "verified_default_local_volume": 3,
    }
    return (
        classes.get(str(candidate["selection_class"]), 99),
        -float(candidate["free_gib"]),
        str(candidate["path"]),
    )


def _partition_rows(partitions: Iterable[Mapping[str, object]] | None) -> list[dict[str, str]]:
    if partitions is None:
        raw = (
            {"mountpoint": item.mountpoint, "device": item.device, "fstype": item.fstype}
            for item in psutil.disk_partitions(all=True)
        )
    else:
        raw = partitions
    rows: list[dict[str, str]] = []
    for item in raw:
        mountpoint = item.get("mountpoint")
        if not isinstance(mountpoint, str) or not mountpoint:
            continue
        rows.append(
            {
                "mountpoint": str(Path(mountpoint).resolve()),
                "device": str(item.get("device", "unknown")),
                "fstype": str(item.get("fstype", "unknown")),
            }
        )
    return rows


def _partition_identity(path: Path, partitions: list[dict[str, str]]) -> tuple[str, str]:
    matches: list[tuple[int, dict[str, str]]] = []
    resolved = path.resolve()
    for row in partitions:
        mountpoint = Path(row["mountpoint"])
        if resolved == mountpoint or mountpoint in resolved.parents:
            matches.append((len(mountpoint.parts), row))
    if not matches:
        return "unknown", "unknown"
    row = max(matches, key=lambda item: item[0])[1]
    return row["device"], row["fstype"]


def _looks_like_scratch(path: Path) -> bool:
    normalized = path.as_posix().lower()
    return "scratch" in normalized or normalized.startswith("/mnt/disks/")


def _claim_owned_root(path: Path) -> Path:
    if path.exists() and (path.is_symlink() or not path.is_dir()):
        raise ValueError("ephemeral root exists but is not a safe directory")
    path.mkdir(parents=True, exist_ok=True)
    marker = path / OWNERSHIP_MARKER
    existing_items = [item for item in path.iterdir() if item.name != OWNERSHIP_MARKER]
    if marker.exists():
        if marker.is_symlink() or not marker.is_file():
            raise ValueError("ephemeral ownership marker is unsafe")
        if marker.read_text(encoding="utf-8").strip() != STORAGE_PLAN_KIND:
            raise ValueError("ephemeral ownership marker content mismatch")
    elif existing_items:
        raise ValueError("refusing to claim non-empty unowned ephemeral root")
    else:
        marker.write_text(STORAGE_PLAN_KIND + "\n", encoding="utf-8", newline="\n")
    return marker.resolve()


def _claim_model_cache_root(path: Path, *, owner_marker: Path) -> Path:
    """Initialize the nested cleanup marker under an already owned scratch root.

    A failed preflight from the original storage planner can leave exactly one empty
    ``hub`` directory before the notebook checks the nested marker. That deterministic
    bootstrap residue is safe to adopt because the parent ownership marker is valid.
    Any other unmarked content remains fail-closed.
    """

    owner_root = owner_marker.parent.resolve()
    expected_owner_marker = owner_root / OWNERSHIP_MARKER
    if (
        owner_marker != expected_owner_marker
        or owner_marker.is_symlink()
        or not owner_marker.is_file()
        or owner_marker.read_text(encoding="utf-8").strip() != STORAGE_PLAN_KIND
    ):
        raise ValueError("ephemeral ownership marker missing or unsafe")
    if path.is_symlink() or (path.exists() and not path.is_dir()):
        raise ValueError("model cache root exists but is not a safe directory")
    path.mkdir(parents=True, exist_ok=True)
    root = path.resolve()
    if root == owner_root or owner_root not in root.parents:
        raise ValueError("model cache root escapes the owned ephemeral root")

    marker = root / CACHE_MARKER_NAME
    if marker.exists():
        if marker.is_symlink() or not marker.is_file():
            raise ValueError("model-cache ownership marker is unsafe")
        if marker.read_text(encoding="utf-8") != CACHE_MARKER_TEXT:
            raise ValueError("model-cache ownership marker content mismatch")
        return root

    existing_items = list(root.iterdir())
    legacy_hub = root / "hub"
    safe_bootstrap_residue = (
        not existing_items
        or (
            existing_items == [legacy_hub]
            and legacy_hub.is_dir()
            and not legacy_hub.is_symlink()
            and not any(legacy_hub.iterdir())
        )
    )
    if not safe_bootstrap_residue:
        raise ValueError("refusing to claim unexpected unmarked model cache content")
    marker.write_text(CACHE_MARKER_TEXT, encoding="utf-8")
    return root


def _reject_overlap(path: Path, protected_roots: Iterable[Path]) -> None:
    resolved = path.resolve()
    for protected in protected_roots:
        candidate = protected.resolve()
        if resolved == candidate or resolved in candidate.parents or candidate in resolved.parents:
            raise ValueError(f"ephemeral root overlaps protected path: {candidate}")


def _existing_directory(path: Path, label: str) -> Path:
    if path.is_symlink():
        raise ValueError(f"{label} is a symlink: {path}")
    resolved = path.resolve()
    if not resolved.is_dir():
        raise ValueError(f"{label} is missing or unsafe: {resolved}")
    return resolved


def _device_id(path: Path) -> int:
    return int(path.stat().st_dev)


def _payload_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest().upper()


def _without(value: Mapping[str, object], key: str) -> dict[str, object]:
    return {name: item for name, item in value.items() if name != key}


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.upper()
        and all(character in "0123456789ABCDEF" for character in value)
    )


def _require_exact_fields(
    value: Mapping[str, object], expected: frozenset[str], label: str
) -> None:
    unknown = sorted(set(value).difference(expected))
    missing = sorted(expected.difference(value))
    if unknown:
        raise ValueError(f"{label} unknown fields: {','.join(unknown)}")
    if missing:
        raise ValueError(f"{label} missing fields: {','.join(missing)}")
