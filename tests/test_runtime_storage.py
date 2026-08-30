import json
from pathlib import Path

import pytest

from vipibench.cache_contract import CACHE_MARKER_NAME, CACHE_MARKER_TEXT
from vipibench.runtime_storage import (
    OWNERSHIP_MARKER,
    STORAGE_PLAN_KIND,
    create_runtime_storage_plan,
    runtime_storage_environment,
    verify_runtime_storage_plan,
    verify_runtime_storage_plan_document,
)


def test_explicit_scratch_uses_only_an_owned_child_and_binds_environment(tmp_path: Path) -> None:
    default = tmp_path / "default"
    scratch = tmp_path / "local-scratch"
    output = default / "work" / "run-output"
    project = tmp_path / "bundle" / "project"
    for path in (default, scratch, output, project):
        path.mkdir(parents=True)
    plan_path = output / "session_evidence" / "storage_plan.json"

    plan = create_runtime_storage_plan(
        session_id="session-001",
        default_root=default,
        output_root=output,
        protected_roots=[project],
        minimum_free_gib=0.001,
        explicit_root=scratch,
        output_path=plan_path,
        partitions=[
            {"mountpoint": str(tmp_path), "device": "fixture", "fstype": "ext4"},
        ],
    )

    ephemeral = scratch / "vipibench-ephemeral"
    assert plan["status"] == "PASS"
    assert plan["selection_mode"] == "explicit_verified_root"
    assert Path(str(plan["ephemeral_root"])) == ephemeral.resolve()
    assert Path(str(plan["ownership_marker"])).read_text(encoding="utf-8").strip()
    assert verify_runtime_storage_plan(
        plan,
        output_root=output,
        protected_roots=[project],
    ) == plan
    assert json.loads(plan_path.read_text(encoding="utf-8"))["plan_sha256"] == plan[
        "plan_sha256"
    ]
    environment = runtime_storage_environment(plan)
    assert environment["VIPIBENCH_EPHEMERAL_MODEL_CACHE_ROOT"] == str(
        ephemeral / "model-cache"
    )
    assert environment["HF_HUB_CACHE"] == str(ephemeral / "model-cache" / "hub")
    assert environment["TMPDIR"] == str(ephemeral / "tmp" / "session-001")


def test_auto_selection_prefers_a_named_scratch_mount_over_default(tmp_path: Path) -> None:
    default = tmp_path / "content"
    scratch = tmp_path / "local-scratch"
    output = default / "work" / "run-output"
    project = tmp_path / "bundle" / "project"
    for path in (default, scratch, output, project):
        path.mkdir(parents=True)

    plan = create_runtime_storage_plan(
        session_id="session-002",
        default_root=default,
        output_root=output,
        protected_roots=[project],
        minimum_free_gib=0.001,
        partitions=[
            {"mountpoint": str(default), "device": "default", "fstype": "ext4"},
            {"mountpoint": str(scratch), "device": "scratch", "fstype": "ext4"},
        ],
    )

    assert plan["selection_mode"] == "verified_named_local_scratch"
    assert plan["selected_mount_root"] == str(scratch.resolve())


def test_owned_retry_initializes_the_notebook_model_cache_marker(tmp_path: Path) -> None:
    default = tmp_path / "content"
    scratch = tmp_path / "local-scratch"
    output = default / "work" / "run-output"
    project = tmp_path / "bundle" / "project"
    for path in (default, scratch, output, project):
        path.mkdir(parents=True)

    ephemeral = scratch / "vipibench-ephemeral"
    model_cache = ephemeral / "model-cache"
    (model_cache / "hub").mkdir(parents=True)
    (ephemeral / OWNERSHIP_MARKER).write_text(
        STORAGE_PLAN_KIND + "\n",
        encoding="utf-8",
        newline="\n",
    )

    plan = create_runtime_storage_plan(
        session_id="session-owned-retry",
        default_root=default,
        output_root=output,
        protected_roots=[project],
        minimum_free_gib=0.001,
        explicit_root=scratch,
        partitions=[
            {"mountpoint": str(tmp_path), "device": "fixture", "fstype": "ext4"},
        ],
    )

    marker = model_cache / CACHE_MARKER_NAME
    assert plan["model_cache_root"] == str(model_cache.resolve())
    assert marker.read_text(encoding="utf-8") == CACHE_MARKER_TEXT


def test_owned_parent_rejects_unexpected_unmarked_model_cache_content(tmp_path: Path) -> None:
    default = tmp_path / "content"
    scratch = tmp_path / "local-scratch"
    output = default / "work" / "run-output"
    for path in (default, scratch, output):
        path.mkdir(parents=True)

    ephemeral = scratch / "vipibench-ephemeral"
    model_cache = ephemeral / "model-cache"
    model_cache.mkdir(parents=True)
    (model_cache / "foreign.bin").write_text("preserve", encoding="utf-8")
    (ephemeral / OWNERSHIP_MARKER).write_text(
        STORAGE_PLAN_KIND + "\n",
        encoding="utf-8",
        newline="\n",
    )

    with pytest.raises(ValueError, match="unexpected unmarked model cache content"):
        create_runtime_storage_plan(
            session_id="session-unmarked-foreign-content",
            default_root=default,
            output_root=output,
            protected_roots=[],
            minimum_free_gib=0.001,
            explicit_root=scratch,
            partitions=[
                {"mountpoint": str(tmp_path), "device": "fixture", "fstype": "ext4"},
            ],
        )

    assert (model_cache / "foreign.bin").read_text(encoding="utf-8") == "preserve"
    assert not (model_cache / CACHE_MARKER_NAME).exists()


def test_explicit_scratch_rejects_overlap_and_unowned_content(tmp_path: Path) -> None:
    default = tmp_path / "default"
    output = default / "run-output"
    output.mkdir(parents=True)

    with pytest.raises(ValueError, match="explicit scratch root rejected"):
        create_runtime_storage_plan(
            session_id="session-overlap",
            default_root=default,
            output_root=output,
            protected_roots=[],
            minimum_free_gib=0.001,
            explicit_root=output,
            partitions=[
                {"mountpoint": str(tmp_path), "device": "fixture", "fstype": "ext4"},
            ],
        )

    scratch = tmp_path / "scratch"
    owned_child = scratch / "vipibench-ephemeral"
    owned_child.mkdir(parents=True)
    (owned_child / "foreign.txt").write_text("not owned", encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty unowned"):
        create_runtime_storage_plan(
            session_id="session-unowned",
            default_root=default,
            output_root=output,
            protected_roots=[],
            minimum_free_gib=0.001,
            explicit_root=scratch,
            partitions=[
                {"mountpoint": str(tmp_path), "device": "fixture", "fstype": "ext4"},
            ],
        )


def test_storage_plan_document_hash_detects_tampering(tmp_path: Path) -> None:
    default = tmp_path / "default"
    scratch = tmp_path / "scratch"
    output = default / "run-output"
    for path in (default, scratch, output):
        path.mkdir(parents=True)
    plan = create_runtime_storage_plan(
        session_id="session-003",
        default_root=default,
        output_root=output,
        protected_roots=[],
        minimum_free_gib=0.001,
        explicit_root=scratch,
        partitions=[
            {"mountpoint": str(tmp_path), "device": "fixture", "fstype": "ext4"},
        ],
    )
    tampered = {**plan, "free_gib_at_selection": 999999.0}

    with pytest.raises(ValueError, match="storage plan hash mismatch"):
        verify_runtime_storage_plan_document(tampered)


def test_live_storage_plan_verification_requires_the_model_cache_marker(tmp_path: Path) -> None:
    default = tmp_path / "default"
    scratch = tmp_path / "scratch"
    output = default / "run-output"
    for path in (default, scratch, output):
        path.mkdir(parents=True)
    plan = create_runtime_storage_plan(
        session_id="session-marker-tamper",
        default_root=default,
        output_root=output,
        protected_roots=[],
        minimum_free_gib=0.001,
        explicit_root=scratch,
        partitions=[
            {"mountpoint": str(tmp_path), "device": "fixture", "fstype": "ext4"},
        ],
    )
    marker = Path(str(plan["model_cache_root"])) / CACHE_MARKER_NAME
    marker.unlink()

    with pytest.raises(ValueError, match="model-cache ownership marker missing or unsafe"):
        verify_runtime_storage_plan(plan, output_root=output, protected_roots=[])
