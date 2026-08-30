import json
from pathlib import Path

import pytest

from vipibench.manifest import build_manifest, readiness_manifest_paths, verify_manifest


def test_artifact_manifest_detects_bound_file_drift(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    source.write_text("version one\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    result = build_manifest(tmp_path, [Path("source.txt")], manifest)
    assert result["status"] == "PASS"
    assert verify_manifest(tmp_path, manifest)["status"] == "PASS"
    source.write_text("version two\n", encoding="utf-8")
    verification = verify_manifest(tmp_path, manifest)
    assert verification["status"] == "FAIL"
    assert verification["errors"] == ["hash_mismatch:source.txt"]


def test_manifest_rejects_empty_subset_and_unsafe_artifact_entries(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    source.write_text("bound\n", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    payload = build_manifest(tmp_path, [Path("source.txt")], manifest_path)

    payload["artifacts"] = []
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    empty = verify_manifest(tmp_path, manifest_path)
    assert empty["status"] == "FAIL"
    assert "manifest_artifacts_missing_or_empty" in empty["errors"]

    payload = build_manifest(tmp_path, [Path("source.txt")], manifest_path)
    payload["artifacts"][0]["path"] = "../source.txt"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    unsafe = verify_manifest(tmp_path, manifest_path)
    assert unsafe["status"] == "FAIL"
    assert "artifact_path_unsafe:../source.txt" in unsafe["errors"]


@pytest.mark.private_integration
def test_readiness_manifest_requires_the_complete_canonical_path_set(tmp_path: Path) -> None:
    root = Path.cwd()
    source_manifest = json.loads((root / "artifact_manifest.json").read_text(encoding="utf-8"))
    source_manifest["artifacts"] = source_manifest["artifacts"][:-1]
    forged = tmp_path / "artifact_manifest.json"
    forged.write_text(json.dumps(source_manifest), encoding="utf-8")

    result = verify_manifest(root, forged, expected_paths=readiness_manifest_paths(root))

    assert result["status"] == "FAIL"
    assert "manifest_artifact_paths_mismatch" in result["errors"]


def test_readiness_manifest_binds_the_runtime_postrun_schema() -> None:
    assert Path("outputs/postrun_audit.schema.json") in readiness_manifest_paths(Path.cwd())
