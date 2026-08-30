import json
import threading
import time
from pathlib import Path

import pytest

from vipibench import durable_snapshot
from vipibench.checkpoint import StageLedger
from vipibench.durable_snapshot import (
    PeriodicContentAddressedSnapshot,
    PeriodicSnapshot,
    _project_snapshot_files,
    _snapshot_files,
    create_atomic_snapshot,
    create_content_addressed_snapshot,
    restore_verified_content_snapshot,
    restore_verified_snapshot,
)


def _active_archive(base: Path, manifest: Path) -> Path:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    return base.parent / payload["archive_name"]


def test_snapshot_restores_verified_files_and_ignores_partial_files(tmp_path: Path) -> None:
    source = tmp_path / "run-output"
    (source / "stage").mkdir(parents=True)
    (source / "stage" / "result.json").write_text('{"ok": true}\n', encoding="utf-8")
    (source / "stage" / "unfinished.tmp").write_text("partial", encoding="utf-8")
    archive = tmp_path / "drive" / "run-output.tar.gz"
    manifest = tmp_path / "drive" / "run-output.snapshot.json"

    snapshot = create_atomic_snapshot(source, archive, manifest)
    assert snapshot["status"] == "PASS"
    assert snapshot["file_count"] == 1
    assert _active_archive(archive, manifest).is_file()

    target = tmp_path / "new-session" / "run-output"
    result = restore_verified_snapshot(archive, manifest, target)
    assert result["status"] == "PASS"
    assert (target / "stage" / "result.json").read_bytes() == (
        source / "stage" / "result.json"
    ).read_bytes()
    assert not (target / "stage" / "unfinished.tmp").exists()


def test_snapshot_restore_rejects_archive_tampering(tmp_path: Path) -> None:
    source = tmp_path / "run-output"
    source.mkdir()
    (source / "result.json").write_text("{}\n", encoding="utf-8")
    archive = tmp_path / "run-output.tar.gz"
    manifest = tmp_path / "run-output.snapshot.json"
    create_atomic_snapshot(source, archive, manifest)
    active = _active_archive(archive, manifest)
    active.write_bytes(active.read_bytes() + b"tamper")

    with pytest.raises(ValueError, match="archive hash mismatch"):
        restore_verified_snapshot(archive, manifest, tmp_path / "restore")


def test_snapshot_restore_does_not_overwrite_nonempty_local_output(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "result.json").write_text("{}\n", encoding="utf-8")
    archive = tmp_path / "run-output.tar.gz"
    manifest = tmp_path / "run-output.snapshot.json"
    create_atomic_snapshot(source, archive, manifest)
    target = tmp_path / "target"
    target.mkdir()
    (target / "newer.json").write_text("{}\n", encoding="utf-8")

    result = restore_verified_snapshot(archive, manifest, target)
    assert result == {
        "schema_version": "1.0.0",
        "status": "SKIP",
        "reason": "local_output_nonempty",
        "restored": False,
    }
    assert (target / "newer.json").is_file()


def test_snapshot_restore_rejects_manifest_hash_tampering(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "result.json").write_text("{}\n", encoding="utf-8")
    archive = tmp_path / "run-output.tar.gz"
    manifest = tmp_path / "run-output.snapshot.json"
    create_atomic_snapshot(source, archive, manifest)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["archive_sha256"] = "0" * 64
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="archive hash mismatch"):
        restore_verified_snapshot(archive, manifest, tmp_path / "restore")


def test_snapshot_uses_two_slots_so_old_manifest_remains_restorable(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "result.json").write_text('{"version": 1}\n', encoding="utf-8")
    archive = tmp_path / "run-output.tar.gz"
    manifest = tmp_path / "run-output.snapshot.json"
    create_atomic_snapshot(source, archive, manifest)
    first_manifest = manifest.read_bytes()
    first_active = _active_archive(archive, manifest)

    (source / "result.json").write_text('{"version": 2}\n', encoding="utf-8")
    create_atomic_snapshot(source, archive, manifest)
    second_active = _active_archive(archive, manifest)
    assert second_active != first_active
    assert first_active.is_file() and second_active.is_file()

    manifest.write_bytes(first_manifest)
    target = tmp_path / "restored-old"
    restore_verified_snapshot(archive, manifest, target)
    assert json.loads((target / "result.json").read_text(encoding="utf-8")) == {
        "version": 1
    }


def test_periodic_snapshot_serializes_overlapping_requests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active = 0
    maximum_active = 0
    guard = threading.Lock()
    start = threading.Barrier(3)

    def fake_snapshot(source_root: Path, archive_path: Path, manifest_path: Path):
        nonlocal active, maximum_active
        with guard:
            active += 1
            maximum_active = max(maximum_active, active)
        time.sleep(0.1)
        with guard:
            active -= 1
        return {"status": "PASS"}

    def invoke_snapshot() -> None:
        start.wait(timeout=2)
        worker.snapshot_now()

    monkeypatch.setattr(durable_snapshot, "create_atomic_snapshot", fake_snapshot)
    worker = PeriodicSnapshot(
        tmp_path / "source",
        tmp_path / "archive.tar.gz",
        tmp_path / "manifest.json",
        interval_seconds=1,
    )
    first = threading.Thread(target=invoke_snapshot)
    second = threading.Thread(target=invoke_snapshot)
    first.start()
    second.start()
    start.wait(timeout=2)
    first.join(timeout=2)
    second.join(timeout=2)

    assert not first.is_alive()
    assert not second.is_alive()
    assert maximum_active == 1
    assert worker.completed_snapshots == 2


def test_content_snapshot_reuses_unchanged_blobs_and_restores(tmp_path: Path) -> None:
    source = tmp_path / "run-output"
    source.mkdir()
    result_path = source / "result.json"
    result_path.write_text('{"version": 1}\n', encoding="utf-8")
    store = tmp_path / "drive" / "run-output.snapshot-store"
    manifest = tmp_path / "drive" / "run-output.snapshot.json"

    first = create_content_addressed_snapshot(source, store, manifest)
    second = create_content_addressed_snapshot(source, store, manifest)

    assert first["written_blob_count"] == 1
    assert second["written_blob_count"] == 0
    assert second["reused_blob_count"] == 1
    assert len(list((store / "blobs").iterdir())) == 1
    target = tmp_path / "restored" / "run-output"
    restored = restore_verified_content_snapshot(store, manifest, target)
    assert restored["status"] == "PASS"
    assert (target / "result.json").read_bytes() == result_path.read_bytes()


def test_content_snapshot_two_generations_bound_blob_retention(tmp_path: Path) -> None:
    source = tmp_path / "run-output"
    source.mkdir()
    result_path = source / "result.json"
    result_path.write_text('{"version": 1}\n', encoding="utf-8")
    store = tmp_path / "store"
    manifest = tmp_path / "snapshot.json"
    create_content_addressed_snapshot(source, store, manifest)

    result_path.write_text('{"version": 2}\n', encoding="utf-8")
    create_content_addressed_snapshot(source, store, manifest)
    assert len(list((store / "blobs").iterdir())) == 2

    create_content_addressed_snapshot(source, store, manifest)
    assert len(list((store / "blobs").iterdir())) == 1


def test_content_snapshot_restore_rejects_blob_tampering(tmp_path: Path) -> None:
    source = tmp_path / "run-output"
    source.mkdir()
    (source / "result.json").write_text("{}\n", encoding="utf-8")
    store = tmp_path / "store"
    manifest = tmp_path / "snapshot.json"
    create_content_addressed_snapshot(source, store, manifest)
    blob = next((store / "blobs").iterdir())
    blob.write_bytes(blob.read_bytes() + b"tamper")

    with pytest.raises(ValueError, match="blob size mismatch|blob hash mismatch"):
        restore_verified_content_snapshot(store, manifest, tmp_path / "restore")


def _write_complete_checkpoint(path: Path, *, complete: bool = True) -> None:
    path.mkdir(parents=True)
    for name in ("trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth"):
        (path / name).write_text(name, encoding="utf-8")
    if complete:
        (path / "model.safetensors").write_bytes(b"weights")


def test_projection_retains_only_resumable_encoder_state(tmp_path: Path) -> None:
    source = tmp_path / "run-output"
    encoder = source / "mdeberta"
    completed = encoder / "mdeberta-text-role-s17"
    completed_model = completed / "model" / "model.safetensors"
    completed_model.parent.mkdir(parents=True)
    completed_model.write_bytes(b"selected-weights")
    development_ledger = StageLedger(encoder / "stage_ledger", artifact_root=encoder)
    development_ledger.complete(
        "development-mdeberta-text-role-s17",
        [completed_model],
        {"source": "fixture"},
    )
    _write_complete_checkpoint(completed / "checkpoints" / "checkpoint-1")

    active = encoder / "mdeberta-text-role-s29"
    active_model = active / "model" / "model.safetensors"
    active_model.parent.mkdir(parents=True)
    active_model.write_bytes(b"other-weights")
    _write_complete_checkpoint(active / "checkpoints" / "checkpoint-1")
    _write_complete_checkpoint(active / "checkpoints" / "checkpoint-2")
    _write_complete_checkpoint(active / "checkpoints" / "checkpoint-3", complete=False)

    before, before_projection = _project_snapshot_files(
        source,
        _snapshot_files(source),
        projection_profile="confirmatory_resume_v1",
    )
    before_paths = {path.relative_to(source).as_posix() for path in before}
    assert "mdeberta/mdeberta-text-role-s17/model/model.safetensors" in before_paths
    assert not any("mdeberta-text-role-s17/checkpoints" in path for path in before_paths)
    assert any("mdeberta-text-role-s29/checkpoints/checkpoint-2" in path for path in before_paths)
    assert not any(
        "mdeberta-text-role-s29/checkpoints/checkpoint-1" in path for path in before_paths
    )
    assert not any(
        "mdeberta-text-role-s29/checkpoints/checkpoint-3" in path for path in before_paths
    )
    assert before_projection["encoder_matrix_stage_verified"] is False

    selection = encoder / "model_selection.json"
    selection.write_text(
        json.dumps(
            {
                "status": "PASS",
                "selected": {"run_id": "mdeberta-text-role-s29"},
            }
        ),
        encoding="utf-8",
    )
    test_manifest = active / "test_manifest.json"
    test_manifest.write_text('{"status": "PASS"}\n', encoding="utf-8")
    orchestration = StageLedger(source / "orchestration_ledger", artifact_root=source)
    orchestration.complete("encoder-matrix", [selection, test_manifest], {"source": "fixture"})

    after, after_projection = _project_snapshot_files(
        source,
        _snapshot_files(source),
        projection_profile="confirmatory_resume_v1",
    )
    after_paths = {path.relative_to(source).as_posix() for path in after}
    assert "mdeberta/mdeberta-text-role-s17/model/model.safetensors" not in after_paths
    assert "mdeberta/mdeberta-text-role-s29/model/model.safetensors" in after_paths
    assert after_projection["encoder_matrix_stage_verified"] is True
    assert after_projection["selected_encoder_run"] == "mdeberta-text-role-s29"


def test_content_snapshot_stop_is_bounded_and_does_not_duplicate_final_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "result.json").write_text("{}\n", encoding="utf-8")
    worker = PeriodicContentAddressedSnapshot(
        source,
        tmp_path / "store",
        tmp_path / "snapshot.json",
        interval_seconds=0.01,
        stop_timeout_seconds=0.05,
    )
    worker.start()
    entered = threading.Event()
    release = threading.Event()
    calls = 0

    def blocked_snapshot(*args, **kwargs):
        nonlocal calls
        calls += 1
        entered.set()
        release.wait(timeout=2)
        return {"status": "PASS"}

    monkeypatch.setattr(durable_snapshot, "create_content_addressed_snapshot", blocked_snapshot)
    assert entered.wait(timeout=1)
    with pytest.raises(TimeoutError, match="final snapshot was not started"):
        worker.stop(final_snapshot=True)
    assert calls == 1
    release.set()
