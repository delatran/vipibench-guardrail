from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from vipibench.exec_splits import (
    audit_exec_splits,
    freeze_exec_splits,
    seal_frozen_split_package,
    verify_frozen_split_package,
)

DATASET_PATH = Path("data/processed/vipibench_exec.jsonl")
CONFIG_PATH = Path("configs/benchmark/exec_catalog.yaml")


@pytest.fixture(scope="module")
def frozen_dir(tmp_path_factory) -> Path:
    output = tmp_path_factory.mktemp("exec-splits") / "frozen"
    freeze_exec_splits(DATASET_PATH, CONFIG_PATH, output)
    return output


def test_frozen_split_and_holdout_manifests_pass(frozen_dir: Path) -> None:
    manifest = json.loads((frozen_dir / "split_manifest.json").read_text(encoding="utf-8"))
    audit = json.loads((frozen_dir / "split_audit.json").read_text(encoding="utf-8"))
    holdouts = json.loads((frozen_dir / "holdout_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == audit["status"] == holdouts["status"] == "PASS"
    assert manifest["family_split_counts"] == {"train": 48, "dev": 16, "test": 16}
    assert audit["near_duplicate_audit"]["violation_count"] == 0
    assert audit["near_duplicate_audit"]["maximum_cross_split_similarity"] < 0.90
    assert len(holdouts["domain_leave_one_out_folds"]) == 4
    assert len(holdouts["surface_realization_leave_one_out_folds"]) == 3


def test_freeze_refuses_to_overwrite(frozen_dir: Path) -> None:
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        freeze_exec_splits(DATASET_PATH, CONFIG_PATH, frozen_dir)


def test_frozen_package_detects_and_repairs_stale_evidence_binding(
    frozen_dir: Path,
    tmp_path: Path,
) -> None:
    copied = tmp_path / "copied"
    shutil.copytree(frozen_dir, copied)
    manifest_path = copied / "split_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["split_audit_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    failed = verify_frozen_split_package(copied, DATASET_PATH, CONFIG_PATH)
    assert failed["status"] == "FAIL"
    assert "manifest_split_audit_binding_mismatch" in failed["errors"]
    repaired = seal_frozen_split_package(copied, DATASET_PATH, CONFIG_PATH)
    assert repaired["status"] == "PASS", repaired["errors"]


def test_audit_detects_cross_split_duplicate(frozen_dir: Path, tmp_path: Path) -> None:
    tampered = tmp_path / "tampered"
    shutil.copytree(frozen_dir, tampered)
    dev_line = (tampered / "dev.jsonl").read_text(encoding="utf-8").splitlines()[0]
    with (tampered / "train.jsonl").open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(dev_line + "\n")
    result = audit_exec_splits(tampered, DATASET_PATH, CONFIG_PATH)
    assert result["status"] == "FAIL"
    assert "split_episode_counts_mismatch" in result["errors"]
    assert any(item.startswith("cross_split_family") for item in result["errors"])
    assert any(
        item.startswith("cross_split_exact_context_duplicate")
        for item in result["errors"]
    )
