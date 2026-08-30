from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from vipibench.coverage import audit_proposal_coverage


def _load_live_manifest(project_root: Path) -> dict[str, object]:
    value = yaml.safe_load(
        (project_root / "docs/proposal_coverage.yaml").read_text(encoding="utf-8")
    )
    assert isinstance(value, dict)
    return value


def _write_manifest(path: Path, value: dict[str, object]) -> None:
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


@pytest.mark.private_integration
def test_live_proposal_coverage_manifest_passes() -> None:
    project_root = Path.cwd()
    manifest = _load_live_manifest(project_root)
    result = audit_proposal_coverage(project_root, project_root / "docs/proposal_coverage.yaml")
    assert "proposal" not in manifest
    assert result["status"] == "PASS", result["errors"]
    assert "proposal_sha256" not in result
    assert result["required_requirement_count"] == 25
    assert result["coverage_complete"] is True
    assert result["readiness_inferred"] is False


def test_proposal_coverage_rejects_deleted_requirement(tmp_path: Path) -> None:
    project_root = Path.cwd()
    manifest = deepcopy(_load_live_manifest(project_root))
    requirements = manifest["requirements"]
    assert isinstance(requirements, list)
    manifest["requirements"] = [item for item in requirements if item["id"] != "DATA-007"]
    path = tmp_path / "coverage.yaml"
    _write_manifest(path, manifest)
    result = audit_proposal_coverage(project_root, path)
    assert result["status"] == "FAIL"
    assert any("DATA-007" in error for error in result["errors"])


def test_proposal_coverage_rejects_missing_implemented_evidence(tmp_path: Path) -> None:
    project_root = Path.cwd()
    manifest = deepcopy(_load_live_manifest(project_root))
    requirements = manifest["requirements"]
    assert isinstance(requirements, list)
    source = next(item for item in requirements if item["id"] == "SOURCE-001")
    source["implemented_evidence"] = ["docs/does-not-exist.md"]
    path = tmp_path / "coverage.yaml"
    _write_manifest(path, manifest)
    result = audit_proposal_coverage(project_root, path)
    assert result["status"] == "FAIL"
    assert any("SOURCE-001" in error for error in result["errors"])


def test_proposal_coverage_requires_every_external_gate_owner(tmp_path: Path) -> None:
    project_root = Path.cwd()
    manifest = deepcopy(_load_live_manifest(project_root))
    requirements = manifest["requirements"]
    assert isinstance(requirements, list)
    compute = next(item for item in requirements if item["id"] == "COMPUTE-001")
    compute["blocked_by"] = []
    path = tmp_path / "coverage.yaml"
    _write_manifest(path, manifest)
    result = audit_proposal_coverage(project_root, path)
    assert result["status"] == "FAIL"
    assert any("resource_measurement" in error for error in result["errors"])


def test_rag_case_study_is_post_run_not_launch_preflight() -> None:
    project_root = Path.cwd()
    manifest = _load_live_manifest(project_root)
    requirements = manifest["requirements"]
    assert isinstance(requirements, list)
    rag = next(item for item in requirements if item["id"] == "GUARDRAIL-003")
    assert rag["phase"] == "post_run"
    assert rag["status"] == "DEFERRED_POST_RUN"
    assert rag["blocked_by"] == []
