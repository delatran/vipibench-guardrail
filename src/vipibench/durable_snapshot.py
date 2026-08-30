from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import tempfile
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import BinaryIO

from vipibench.dataio import write_json

CONTENT_STORE_SCHEMA_VERSION = "2.0.0"
CONTENT_STORE_MODE = "content_addressed_v1"
CONFIRMATORY_PROJECTION_PROFILE = "confirmatory_resume_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _snapshot_files(source_root: Path) -> list[Path]:
    if not source_root.is_dir():
        return []
    return sorted(
        (
            path
            for path in source_root.rglob("*")
            if path.is_file()
            and not path.is_symlink()
            and not path.name.endswith((".tmp", ".part"))
        ),
        key=lambda path: path.relative_to(source_root).as_posix(),
    )


def create_content_addressed_snapshot(
    source_root: Path,
    store_root: Path,
    manifest_path: Path,
    *,
    projection_profile: str = CONFIRMATORY_PROJECTION_PROFILE,
) -> dict[str, object]:
    """Publish a two-generation content-addressed snapshot without recopying stable blobs."""

    source = source_root.resolve()
    store = store_root.resolve()
    manifest = manifest_path.resolve()
    store.mkdir(parents=True, exist_ok=True)
    blobs = store / "blobs"
    blobs.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)

    all_files = _snapshot_files(source)
    files, projection = _project_snapshot_files(
        source,
        all_files,
        projection_profile=projection_profile,
    )
    previous_pointer = _load_json_object(manifest)
    previous_index = _load_active_content_index(store, previous_pointer)
    cached_entries = {
        str(entry.get("path")): entry
        for entry in previous_index.get("entries", [])
        if isinstance(entry, dict)
    }

    entries: list[dict[str, object]] = []
    reused_blob_count = 0
    written_blob_count = 0
    for path in files:
        relative = path.relative_to(source).as_posix()
        stat = path.stat()
        cached = cached_entries.get(relative)
        if _cached_blob_is_reusable(cached, stat, blobs):
            entries.append(dict(cached))
            reused_blob_count += 1
            continue
        entry, wrote_blob = _materialize_content_blob(path, relative, blobs)
        entries.append(entry)
        written_blob_count += int(wrote_blob)

    entries.sort(key=lambda item: str(item["path"]))
    state_rows = [
        {
            "path": entry["path"],
            "size_bytes": entry["size_bytes"],
            "mtime_ns": entry["mtime_ns"],
        }
        for entry in entries
    ]
    total_size = sum(int(entry["size_bytes"]) for entry in entries)
    index_payload = {
        "schema_version": CONTENT_STORE_SCHEMA_VERSION,
        "status": "PASS",
        "storage_mode": CONTENT_STORE_MODE,
        "created_at": datetime.now(UTC).isoformat(),
        "source_root_name": source.name,
        "projection_profile": projection_profile,
        "source_state_fingerprint": hashlib.sha256(
            _json_bytes(state_rows)
        ).hexdigest().upper(),
        "file_count": len(entries),
        "total_size_bytes": total_size,
        "entries": entries,
        "projection": projection,
        "claim_boundary": (
            "This index binds one projected resume generation to immutable content-addressed "
            "blobs. Restore re-hashes every blob and StageLedger still verifies stage outputs."
        ),
    }
    index_a, index_b = _content_index_slots(store)
    current_name = (
        previous_pointer.get("index_name") if isinstance(previous_pointer, dict) else None
    )
    published_index = index_b if current_name == index_a.name else index_a
    _write_json_atomic(published_index, index_payload)
    index_sha256 = _sha256(published_index)
    pointer_payload = {
        "schema_version": CONTENT_STORE_SCHEMA_VERSION,
        "status": "PASS",
        "storage_mode": CONTENT_STORE_MODE,
        "created_at": index_payload["created_at"],
        "store_name": store.name,
        "index_name": published_index.name,
        "index_sha256": index_sha256,
        "projection_profile": projection_profile,
        "source_state_fingerprint": index_payload["source_state_fingerprint"],
        "file_count": len(entries),
        "total_size_bytes": total_size,
        "projection": projection,
        "claim_boundary": (
            "PASS proves a locally published pointer and checksum-bound content index. It does "
            "not prove asynchronous Google Drive server propagation or hosted execution."
        ),
    }
    _write_json_atomic(manifest, pointer_payload)
    garbage_collected = _garbage_collect_content_blobs(store)
    return {
        **pointer_payload,
        "reused_blob_count": reused_blob_count,
        "written_blob_count": written_blob_count,
        "garbage_collected_blob_count": garbage_collected,
    }


def restore_verified_content_snapshot(
    store_root: Path,
    manifest_path: Path,
    target_root: Path,
) -> dict[str, object]:
    """Restore a content-addressed generation into an empty target after full hash checks."""

    store = store_root.resolve()
    manifest = manifest_path.resolve()
    target = target_root.resolve()
    if target.exists() and any(target.iterdir()):
        return {
            "schema_version": CONTENT_STORE_SCHEMA_VERSION,
            "status": "SKIP",
            "reason": "local_output_nonempty",
            "restored": False,
        }
    if not manifest.is_file():
        return {
            "schema_version": CONTENT_STORE_SCHEMA_VERSION,
            "status": "SKIP",
            "reason": "durable_snapshot_absent",
            "restored": False,
        }
    pointer = _require_content_pointer(manifest)
    if pointer.get("store_name") != store.name:
        raise ValueError("durable content store name mismatch")
    index_name = pointer.get("index_name")
    allowed_names = {path.name for path in _content_index_slots(store)}
    if index_name not in allowed_names:
        raise ValueError("durable content index name mismatch")
    index_path = store / str(index_name)
    if not index_path.is_file():
        raise FileNotFoundError("durable content index is missing")
    if pointer.get("index_sha256") != _sha256(index_path):
        raise ValueError("durable content index hash mismatch")
    index = _load_json_object(index_path)
    _validate_content_index(index, pointer)
    entries = index["entries"]

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(prefix=".vipibench-restore-", dir=str(target.parent))
    ).resolve()
    restored_files = 0
    try:
        for entry in entries:
            relative = _safe_snapshot_relative(str(entry["path"]))
            destination = (temporary_root / Path(*relative.parts)).resolve()
            destination.relative_to(temporary_root)
            blob = store / "blobs" / str(entry["sha256"])
            if not blob.is_file():
                raise FileNotFoundError(f"durable content blob missing: {entry['sha256']}")
            if blob.stat().st_size != int(entry["size_bytes"]):
                raise ValueError(f"durable content blob size mismatch: {entry['path']}")
            if _sha256(blob) != entry["sha256"]:
                raise ValueError(f"durable content blob hash mismatch: {entry['path']}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(blob, destination)
            mtime_ns = int(entry["mtime_ns"])
            os.utime(destination, ns=(mtime_ns, mtime_ns))
            restored_files += 1
        if target.exists():
            target.rmdir()
        os.replace(temporary_root, target)
    except Exception:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise
    return {
        "schema_version": CONTENT_STORE_SCHEMA_VERSION,
        "status": "PASS",
        "restored": True,
        "restored_file_count": restored_files,
        "index_sha256": pointer["index_sha256"],
        "projection": pointer["projection"],
        "target_root": str(target),
    }


def _project_snapshot_files(
    source: Path,
    files: list[Path],
    *,
    projection_profile: str,
) -> tuple[list[Path], dict[str, object]]:
    if projection_profile != CONFIRMATORY_PROJECTION_PROFILE:
        raise ValueError(f"unsupported durable projection profile: {projection_profile}")
    encoder_root = source / "mdeberta"
    completed_runs = _verified_encoder_development_runs(encoder_root)
    latest_checkpoints = _latest_complete_encoder_checkpoints(encoder_root, completed_runs)
    matrix_marker = source / "orchestration_ledger" / "encoder-matrix.complete.json"
    matrix_complete = _verified_stage_marker(source, matrix_marker) is not None
    selected_run = _selected_encoder_run(encoder_root) if matrix_complete else None
    if matrix_complete and selected_run is None:
        matrix_complete = False

    included: list[Path] = []
    reason_counts: dict[str, int] = {}
    excluded_size = 0
    for path in files:
        relative = path.relative_to(source)
        parts = relative.parts
        reason = None
        if len(parts) >= 4 and parts[0] == "mdeberta" and parts[2] == "checkpoints":
            run_id = parts[1]
            checkpoint_name = parts[3]
            if run_id in completed_runs:
                reason = "completed_encoder_checkpoint"
            elif latest_checkpoints.get(run_id) != checkpoint_name:
                reason = "superseded_or_incomplete_encoder_checkpoint"
        if (
            reason is None
            and matrix_complete
            and len(parts) >= 3
            and parts[0] == "mdeberta"
            and parts[2] == "model"
            and parts[1] != selected_run
        ):
            reason = "nonselected_encoder_model_after_verified_matrix"
        if reason is None:
            included.append(path)
        else:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
            excluded_size += path.stat().st_size
    return included, {
        "profile": projection_profile,
        "source_file_count": len(files),
        "included_file_count": len(included),
        "excluded_file_count": len(files) - len(included),
        "excluded_size_bytes": excluded_size,
        "exclusion_reason_counts": dict(sorted(reason_counts.items())),
        "encoder_matrix_stage_verified": matrix_complete,
        "selected_encoder_run": selected_run,
        "verified_completed_encoder_runs": sorted(completed_runs),
        "retained_latest_encoder_checkpoints": dict(sorted(latest_checkpoints.items())),
        "resume_boundary": (
            "Before outer encoder-matrix completion, verified completed models and only the "
            "newest complete checkpoint for each incomplete run are retained. After the outer "
            "marker is hash-valid, only the selected model weights remain in the durable "
            "projection; all small evidence and orchestration markers remain retained."
        ),
    }


def _verified_encoder_development_runs(encoder_root: Path) -> set[str]:
    ledger_root = encoder_root / "stage_ledger"
    completed: set[str] = set()
    if not ledger_root.is_dir():
        return completed
    for marker in ledger_root.glob("development-*.complete.json"):
        payload = _verified_stage_marker(encoder_root, marker)
        stage_id = payload.get("stage_id") if payload is not None else None
        if isinstance(stage_id, str) and stage_id.startswith("development-"):
            completed.add(stage_id.removeprefix("development-"))
    return completed


def _latest_complete_encoder_checkpoints(
    encoder_root: Path,
    completed_runs: set[str],
) -> dict[str, str]:
    latest: dict[str, str] = {}
    if not encoder_root.is_dir():
        return latest
    for run_dir in encoder_root.iterdir():
        if not run_dir.is_dir() or run_dir.name in completed_runs:
            continue
        checkpoint_root = run_dir / "checkpoints"
        candidates = []
        if checkpoint_root.is_dir():
            for candidate in checkpoint_root.iterdir():
                if candidate.is_dir() and _complete_encoder_checkpoint(candidate):
                    candidates.append(candidate)
        if candidates:
            selected = max(candidates, key=lambda path: int(path.name.split("-", 1)[1]))
            latest[run_dir.name] = selected.name
    return latest


def _complete_encoder_checkpoint(path: Path) -> bool:
    if not path.name.startswith("checkpoint-"):
        return False
    try:
        step = int(path.name.split("-", 1)[1])
    except ValueError:
        return False
    if step <= 0:
        return False
    required = ("trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth")
    if any(not (path / name).is_file() for name in required):
        return False
    model_files = [
        candidate
        for candidate in (path / "model.safetensors", path / "pytorch_model.bin")
        if candidate.is_file()
    ]
    return len(model_files) == 1


def _selected_encoder_run(encoder_root: Path) -> str | None:
    selection = _load_json_object(encoder_root / "model_selection.json")
    selected = selection.get("selected") if isinstance(selection, dict) else None
    run_id = selected.get("run_id") if isinstance(selected, dict) else None
    if not isinstance(run_id, str) or not (encoder_root / run_id / "model").is_dir():
        return None
    return run_id


def _verified_stage_marker(artifact_root: Path, marker: Path) -> dict[str, object] | None:
    payload = _load_json_object(marker)
    if not isinstance(payload, dict) or payload.get("status") != "PASS":
        return None
    outputs = payload.get("outputs")
    if not isinstance(outputs, dict) or not outputs:
        return None
    for value, expected_sha256 in outputs.items():
        try:
            relative = _safe_snapshot_relative(str(value))
            output = (artifact_root / Path(*relative.parts)).resolve()
            output.relative_to(artifact_root.resolve())
        except ValueError:
            return None
        if (
            not output.is_file()
            or not isinstance(expected_sha256, str)
            or _sha256(output) != expected_sha256.upper()
        ):
            return None
    return payload


def _materialize_content_blob(
    source: Path,
    relative: str,
    blob_root: Path,
) -> tuple[dict[str, object], bool]:
    before = source.stat()
    temporary = blob_root / f".{uuid.uuid4().hex}.tmp"
    digest = hashlib.sha256()
    size = 0
    try:
        with source.open("rb") as input_handle, temporary.open("xb") as output_handle:
            for block in iter(lambda: input_handle.read(1024 * 1024), b""):
                output_handle.write(block)
                digest.update(block)
                size += len(block)
        after = source.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f"snapshot source changed while copying: {relative}")
        if size != after.st_size:
            raise RuntimeError(f"snapshot source size changed while copying: {relative}")
        sha256 = digest.hexdigest().upper()
        destination = blob_root / sha256
        wrote_blob = False
        if destination.is_file():
            if destination.stat().st_size != size or _sha256(destination) != sha256:
                os.replace(temporary, destination)
                wrote_blob = True
            else:
                temporary.unlink(missing_ok=True)
        else:
            os.replace(temporary, destination)
            wrote_blob = True
        return (
            {
                "path": relative,
                "size_bytes": size,
                "mtime_ns": after.st_mtime_ns,
                "sha256": sha256,
            },
            wrote_blob,
        )
    finally:
        temporary.unlink(missing_ok=True)


def _cached_blob_is_reusable(cached: object, stat: os.stat_result, blob_root: Path) -> bool:
    if not isinstance(cached, dict):
        return False
    sha256 = cached.get("sha256")
    if not isinstance(sha256, str) or not _is_sha256(sha256):
        return False
    blob = blob_root / sha256
    return (
        cached.get("size_bytes") == stat.st_size
        and cached.get("mtime_ns") == stat.st_mtime_ns
        and blob.is_file()
        and blob.stat().st_size == stat.st_size
    )


def _require_content_pointer(path: Path) -> dict[str, object]:
    payload = _load_json_object(path)
    if not isinstance(payload, dict):
        raise ValueError("durable content pointer is not a JSON object")
    if payload.get("schema_version") != CONTENT_STORE_SCHEMA_VERSION:
        raise ValueError("durable content pointer schema mismatch")
    if payload.get("status") != "PASS":
        raise ValueError("durable content pointer status is not PASS")
    if payload.get("storage_mode") != CONTENT_STORE_MODE:
        raise ValueError("durable content pointer storage mode mismatch")
    return payload


def _validate_content_index(index: object, pointer: dict[str, object]) -> None:
    if not isinstance(index, dict):
        raise ValueError("durable content index is not a JSON object")
    if index.get("schema_version") != CONTENT_STORE_SCHEMA_VERSION:
        raise ValueError("durable content index schema mismatch")
    if index.get("status") != "PASS" or index.get("storage_mode") != CONTENT_STORE_MODE:
        raise ValueError("durable content index status or mode mismatch")
    if index.get("projection_profile") != pointer.get("projection_profile"):
        raise ValueError("durable content projection profile mismatch")
    entries = index.get("entries")
    if not isinstance(entries, list):
        raise ValueError("durable content index entries missing")
    seen: set[str] = set()
    total = 0
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {
            "path",
            "size_bytes",
            "mtime_ns",
            "sha256",
        }:
            raise ValueError("durable content index entry fields invalid")
        relative = _safe_snapshot_relative(str(entry["path"])).as_posix()
        if relative in seen:
            raise ValueError("durable content index contains duplicate paths")
        seen.add(relative)
        size = entry["size_bytes"]
        mtime_ns = entry["mtime_ns"]
        sha256 = entry["sha256"]
        if (
            not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
            or not isinstance(mtime_ns, int)
            or isinstance(mtime_ns, bool)
            or mtime_ns < 0
            or not isinstance(sha256, str)
            or not _is_sha256(sha256)
        ):
            raise ValueError("durable content index entry value invalid")
        total += size
    if index.get("file_count") != len(entries) or pointer.get("file_count") != len(entries):
        raise ValueError("durable content index file count mismatch")
    if index.get("total_size_bytes") != total or pointer.get("total_size_bytes") != total:
        raise ValueError("durable content index total size mismatch")
    if index.get("projection") != pointer.get("projection"):
        raise ValueError("durable content projection metadata mismatch")


def _safe_snapshot_relative(value: str) -> PurePosixPath:
    relative = PurePosixPath(value)
    if not value or relative.is_absolute() or ".." in relative.parts or "\\" in value:
        raise ValueError(f"unsafe durable snapshot path: {value}")
    return relative


def _content_index_slots(store: Path) -> tuple[Path, Path]:
    return store / "index.a.json", store / "index.b.json"


def _load_active_content_index(
    store: Path,
    pointer: dict[str, object] | None,
) -> dict[str, object]:
    if not isinstance(pointer, dict) or pointer.get("storage_mode") != CONTENT_STORE_MODE:
        return {}
    index_name = pointer.get("index_name")
    allowed = {path.name for path in _content_index_slots(store)}
    if index_name not in allowed:
        return {}
    path = store / str(index_name)
    if not path.is_file() or pointer.get("index_sha256") != _sha256(path):
        return {}
    payload = _load_json_object(path)
    return payload if isinstance(payload, dict) else {}


def _garbage_collect_content_blobs(store: Path) -> int:
    retained: set[str] = set()
    for index_path in _content_index_slots(store):
        payload = _load_json_object(index_path)
        entries = payload.get("entries", []) if isinstance(payload, dict) else []
        for entry in entries:
            if isinstance(entry, dict) and isinstance(entry.get("sha256"), str):
                retained.add(str(entry["sha256"]))
    removed = 0
    blob_root = store / "blobs"
    if not blob_root.is_dir():
        return removed
    for path in blob_root.iterdir():
        if path.is_file() and not path.name.startswith(".") and path.name not in retained:
            path.unlink()
            removed += 1
    return removed


def _load_json_object(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789ABCDEF" for character in value)


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
            + b"\n"
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def create_atomic_snapshot(
    source_root: Path,
    archive_path: Path,
    manifest_path: Path,
) -> dict[str, object]:
    """Create a checksum-bound archive and replace only after the new archive is complete."""

    source = source_root.resolve()
    archive = archive_path.resolve()
    manifest = manifest_path.resolve()
    archive.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    slot_a, slot_b = _archive_slots(archive)
    current_name = None
    if manifest.is_file():
        try:
            import json

            current = json.loads(manifest.read_text(encoding="utf-8"))
            current_name = current.get("archive_name")
        except (OSError, ValueError):
            current_name = None
    published_archive = slot_b if current_name == slot_a.name else slot_a
    temporary_archive = archive.with_name(f".{published_archive.name}.{token}.tmp")
    files = _snapshot_files(source)
    try:
        with tarfile.open(temporary_archive, mode="w:gz", format=tarfile.PAX_FORMAT) as handle:
            for path in files:
                relative = path.relative_to(source).as_posix()
                handle.add(path, arcname=relative, recursive=False)
        archive_sha256 = _sha256(temporary_archive)
        archive_size = temporary_archive.stat().st_size
        os.replace(temporary_archive, published_archive)
        payload = {
            "schema_version": "1.0.0",
            "status": "PASS",
            "created_at": datetime.now(UTC).isoformat(),
            "archive_name": published_archive.name,
            "archive_sha256": archive_sha256,
            "archive_size_bytes": archive_size,
            "file_count": len(files),
            "source_root_name": source.name,
            "ignored_suffixes": [".tmp", ".part"],
            "claim_boundary": (
                "PASS proves one atomically published archive and checksum. Restored stage "
                "outputs remain subject to StageLedger hash verification."
            ),
        }
        write_json(manifest, payload)
        return payload
    finally:
        temporary_archive.unlink(missing_ok=True)


def restore_verified_snapshot(
    archive_path: Path,
    manifest_path: Path,
    target_root: Path,
) -> dict[str, object]:
    """Restore a verified archive into an empty/missing target without unsafe tar extraction."""

    archive = archive_path.resolve()
    manifest_path = manifest_path.resolve()
    target = target_root.resolve()
    if target.exists() and any(target.iterdir()):
        return {
            "schema_version": "1.0.0",
            "status": "SKIP",
            "reason": "local_output_nonempty",
            "restored": False,
        }
    if not manifest_path.is_file():
        return {
            "schema_version": "1.0.0",
            "status": "SKIP",
            "reason": "durable_snapshot_absent",
            "restored": False,
        }
    import json

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("status") != "PASS":
        raise ValueError("durable snapshot manifest status is not PASS")
    archive_name = payload.get("archive_name")
    allowed_names = {path.name for path in _archive_slots(archive)}
    if archive_name not in allowed_names:
        raise ValueError("durable snapshot archive name mismatch")
    published_archive = archive.parent / str(archive_name)
    if not published_archive.is_file():
        raise FileNotFoundError("durable snapshot archive named by the manifest is missing")
    if payload.get("archive_sha256") != _sha256(published_archive):
        raise ValueError("durable snapshot archive hash mismatch")
    if int(payload.get("archive_size_bytes", -1)) != published_archive.stat().st_size:
        raise ValueError("durable snapshot archive size mismatch")

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(prefix=".vipibench-restore-", dir=str(target.parent))
    ).resolve()
    restored_files = 0
    try:
        with tarfile.open(published_archive, mode="r:gz") as handle:
            members = handle.getmembers()
            for member in members:
                relative = PurePosixPath(member.name)
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError(f"unsafe durable snapshot member: {member.name}")
                if member.issym() or member.islnk() or not (member.isfile() or member.isdir()):
                    raise ValueError(f"unsupported durable snapshot member: {member.name}")
                destination = (temporary_root / Path(*relative.parts)).resolve()
                try:
                    destination.relative_to(temporary_root)
                except ValueError as exc:
                    raise ValueError(f"unsafe durable snapshot member: {member.name}") from exc
                if member.isdir():
                    destination.mkdir(parents=True, exist_ok=True)
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                source: BinaryIO | None = handle.extractfile(member)
                if source is None:
                    raise ValueError(f"cannot read durable snapshot member: {member.name}")
                with source, destination.open("wb") as output:
                    shutil.copyfileobj(source, output)
                restored_files += 1
        if restored_files != int(payload.get("file_count", -1)):
            raise ValueError("durable snapshot file count mismatch")
        if target.exists():
            target.rmdir()
        os.replace(temporary_root, target)
    except Exception:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise
    return {
        "schema_version": "1.0.0",
        "status": "PASS",
        "restored": True,
        "restored_file_count": restored_files,
        "archive_sha256": payload["archive_sha256"],
        "target_root": str(target),
    }


def _archive_slots(base: Path) -> tuple[Path, Path]:
    suffixes = "".join(base.suffixes)
    stem = base.name[: -len(suffixes)] if suffixes else base.name
    return (
        base.with_name(f"{stem}.a{suffixes}"),
        base.with_name(f"{stem}.b{suffixes}"),
    )


class PeriodicContentAddressedSnapshot:
    """Periodically publish bounded incremental snapshots with bounded shutdown."""

    def __init__(
        self,
        source_root: Path,
        store_root: Path,
        manifest_path: Path,
        *,
        interval_seconds: float,
        stop_timeout_seconds: float,
        projection_profile: str = CONFIRMATORY_PROJECTION_PROFILE,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("snapshot interval must be positive")
        if stop_timeout_seconds <= 0:
            raise ValueError("snapshot stop timeout must be positive")
        self.source_root = source_root
        self.store_root = store_root
        self.manifest_path = manifest_path
        self.interval_seconds = interval_seconds
        self.stop_timeout_seconds = stop_timeout_seconds
        self.projection_profile = projection_profile
        self._stop = threading.Event()
        self._snapshot_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self.last_error: str | None = None
        self.completed_snapshots = 0

    def snapshot_now(self) -> dict[str, object]:
        with self._snapshot_lock:
            result = create_content_addressed_snapshot(
                self.source_root,
                self.store_root,
                self.manifest_path,
                projection_profile=self.projection_profile,
            )
            self.completed_snapshots += 1
            self.last_error = None
            return result

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("periodic content snapshot already started")
        self.snapshot_now()
        self._thread = threading.Thread(
            target=self._run,
            name="vipibench-content-snapshot",
            daemon=True,
        )
        self._thread.start()

    def stop(self, *, final_snapshot: bool = True) -> dict[str, object] | None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.stop_timeout_seconds)
            if self._thread.is_alive():
                raise TimeoutError(
                    "periodic content snapshot did not stop within "
                    f"{self.stop_timeout_seconds:g} seconds; final snapshot was not started"
                )
        result = self.snapshot_now() if final_snapshot else None
        if self.last_error is not None:
            raise RuntimeError(f"periodic content snapshot failed: {self.last_error}")
        return result

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            try:
                self.snapshot_now()
            except Exception as exc:  # pragma: no cover - integration failure path
                self.last_error = f"{type(exc).__name__}:{exc}"


class PeriodicSnapshot:
    """Keep a last-known-good Drive snapshot while the notebook process is running."""

    def __init__(
        self,
        source_root: Path,
        archive_path: Path,
        manifest_path: Path,
        *,
        interval_seconds: float,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("snapshot interval must be positive")
        self.source_root = source_root
        self.archive_path = archive_path
        self.manifest_path = manifest_path
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._snapshot_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self.last_error: str | None = None
        self.completed_snapshots = 0

    def snapshot_now(self) -> dict[str, object]:
        with self._snapshot_lock:
            result = create_atomic_snapshot(
                self.source_root,
                self.archive_path,
                self.manifest_path,
            )
            self.completed_snapshots += 1
            return result

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("periodic snapshot already started")
        self.snapshot_now()
        self._thread = threading.Thread(
            target=self._run,
            name="vipibench-durable-snapshot",
            daemon=True,
        )
        self._thread.start()

    def stop(self, *, final_snapshot: bool = True) -> dict[str, object] | None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        result = self.snapshot_now() if final_snapshot else None
        if self.last_error is not None:
            raise RuntimeError(f"periodic durable snapshot failed: {self.last_error}")
        return result

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            try:
                self.snapshot_now()
            except Exception as exc:  # pragma: no cover - exercised via bootstrap failure path
                self.last_error = f"{type(exc).__name__}:{exc}"
