from pathlib import Path

from vipibench.exec_detector_data import detector_text, load_executable_episodes
from vipibench.provenance_contrast import (
    audit_provenance_contrast_path,
    compile_provenance_contrast,
)


def test_provenance_contrast_compiler_is_paired_and_auditable(tmp_path: Path) -> None:
    dataset = tmp_path / "provenance_contrast.jsonl"
    manifest = tmp_path / "provenance_contrast_manifest.json"
    result = compile_provenance_contrast(
        Path("configs/benchmark/provenance_contrast.yaml"),
        dataset,
        manifest,
    )
    assert result["status"] == "PASS", result["errors"]
    assert result["episode_count"] == 2400
    assert result["pair_count"] == 1200
    assert result["split_pair_counts"] == {"dev": 200, "test": 400, "train": 600}
    assert result["condition_pair_counts"]["canonical"] == 1000
    assert manifest.is_file()
    assert audit_provenance_contrast_path(dataset)["status"] == "PASS"


def test_text_ablation_is_identical_but_bound_provenance_is_identifiable(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "provenance_contrast.jsonl"
    compile_provenance_contrast(
        Path("configs/benchmark/provenance_contrast.yaml"),
        dataset,
        tmp_path / "manifest.json",
    )
    episodes = load_executable_episodes(dataset)
    benign = episodes[0]
    injection = episodes[1]
    assert benign.metadata.matched_pair_id == injection.metadata.matched_pair_id
    assert detector_text(benign, "text_only") == detector_text(injection, "text_only")
    assert detector_text(benign, "role_only") == detector_text(injection, "role_only")
    assert detector_text(benign, "text_role") != detector_text(injection, "text_role")
