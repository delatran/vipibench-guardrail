from __future__ import annotations

import json
from pathlib import Path

from vipibench.dataio import canonical_json
from vipibench.episode import bind_content_hash
from vipibench.exec_validation import validate_exec_benchmark

CONFIG_PATH = Path("configs/benchmark/exec_catalog.yaml")


DATASET_PATH = Path("data/processed/vipibench_exec.jsonl")
TEMPLATE_PATH = Path("data/processed/vipibench_exec_templates.jsonl")


def test_independent_validator_passes_locked_catalog() -> None:
    result = validate_exec_benchmark(DATASET_PATH, CONFIG_PATH, TEMPLATE_PATH)
    assert result["status"] == "PASS"
    assert result["schema_validated_episode_count"] == 2400
    assert result["unique_episode_hash_count"] == 2400
    assert result["complete_matched_pair_count"] == 200
    assert set(result["generator_counts"].values()) == {800}
    assert set(result["generator_label_counts"].values()) == {400}


def test_independent_validator_rejects_cross_split_family(
    tmp_path: Path,
) -> None:
    lines = DATASET_PATH.read_text(encoding="utf-8").splitlines()
    payload = json.loads(lines[0])
    payload["metadata"]["split"] = "test"
    replacement = bind_content_hash(payload)
    tampered = tmp_path / "tampered.jsonl"
    lines[0] = canonical_json(replacement)
    tampered.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    result = validate_exec_benchmark(tampered, CONFIG_PATH, TEMPLATE_PATH)
    assert result["status"] == "FAIL"
    assert "family_crosses_split" in result["errors"]
    assert "family_split_assignment_mismatch" in result["errors"]


def test_independent_validator_rejects_hash_tamper(
    tmp_path: Path,
) -> None:
    lines = DATASET_PATH.read_text(encoding="utf-8").splitlines()
    lines[0] = lines[0].replace("pending", "tampered", 1)
    tampered = tmp_path / "hash-tampered.jsonl"
    tampered.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    result = validate_exec_benchmark(tampered, CONFIG_PATH, TEMPLATE_PATH)
    assert result["status"] == "FAIL"
    assert result["errors"][0].startswith("input_validation_error:ValueError")
