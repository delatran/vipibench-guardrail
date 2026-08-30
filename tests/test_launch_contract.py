from pathlib import Path

import pytest

from vipibench.launch_contract import validate_new_run_output_root


def test_new_run_output_root_allows_only_required_smoke_evidence(tmp_path: Path) -> None:
    output_root = tmp_path / "run"
    evidence_root = output_root / "session_evidence"
    evidence_root.mkdir(parents=True)
    (evidence_root / "pre_launch_readiness.json").write_text("{}\n", encoding="utf-8")

    validate_new_run_output_root(output_root, evidence_root, resume_existing=False)


def test_new_run_output_root_rejects_unexpected_preexisting_output(tmp_path: Path) -> None:
    output_root = tmp_path / "run"
    evidence_root = output_root / "session_evidence"
    evidence_root.mkdir(parents=True)
    (output_root / "stale-run-manifest.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="data other than the required smoke"):
        validate_new_run_output_root(output_root, evidence_root, resume_existing=False)


def test_resume_keeps_existing_output_root_behavior(tmp_path: Path) -> None:
    output_root = tmp_path / "run"
    output_root.mkdir()
    (output_root / "bound-run-output.json").write_text("{}\n", encoding="utf-8")

    validate_new_run_output_root(
        output_root,
        output_root / "session_evidence",
        resume_existing=True,
    )
